#!/usr/bin/env python
import os
from common.basedir import BASEDIR
os.environ['BASEDIR'] = BASEDIR
import argparse
import zmq
import pygame
import numpy as np
import cv2
import sys
from collections import namedtuple
from selfdrive.messaging import sub_sock, recv_one_or_none, recv_one
from common.transformations.camera import eon_intrinsics, FULL_FRAME_SIZE
from common.transformations.model import MODEL_CX, MODEL_CY, MODEL_INPUT_SIZE
from selfdrive.config import UIParams as UP
from selfdrive.services import service_list
from selfdrive.controls.lib.radar_helpers import RDR_TO_LDR
from selfdrive.controls.lib.vehicle_model import VehicleModel
from selfdrive.controls.lib.latcontrol_helpers import calc_desired_path, compute_path_pinv, model_polyfit
from openpilot_tools.lib.lazy_property import lazy_property
from openpilot_tools.replay.lib.ui_helpers import to_lid_pt, draw_path, draw_steer_path, draw_mpc, \
                                                  draw_lead_car, draw_lead_on, init_plots, warp_points, find_color
from selfdrive.car.toyota.interface import CarInterface as ToyotaInterface
try:
  from selfdrive.visiond.visiontest import VisionTest
  vision_test = True
except ImportError:
  vision_test = False

HOR = os.getenv("HORIZONTAL") is not None

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

_PATH_X = np.arange(101.)
_PATH_XD = np.arange(101.)
_PATH_PINV = compute_path_pinv(50)
#_BB_OFFSET = 290, 332
_BB_OFFSET = 0,0
_BB_SCALE = 1164/640.
_BB_TO_FULL_FRAME = np.asarray([
    [_BB_SCALE, 0., _BB_OFFSET[0]],
    [0., _BB_SCALE, _BB_OFFSET[1]],
    [0., 0.,   1.]])
_FULL_FRAME_TO_BB = np.linalg.inv(_BB_TO_FULL_FRAME)

ModelUIData = namedtuple("ModelUIData", ["cpath", "lpath", "rpath", "lead", "lead_std", "freepath"])


class CalibrationTransformsForWarpMatrix(object):
  def __init__(self, model_to_full_frame, K, E):
    self._model_to_full_frame = model_to_full_frame
    self._K = K
    self._E = E

  @property
  def model_to_bb(self):
    return _FULL_FRAME_TO_BB.dot(self._model_to_full_frame)

  @lazy_property
  def model_to_full_frame(self):
    return self._model_to_full_frame

  @lazy_property
  def car_to_model(self):
    return np.linalg.inv(self._model_to_full_frame).dot(self._K).dot(
      self._E[:, [0, 1, 3]])

  @lazy_property
  def car_to_bb(self):
    return _BB_TO_FULL_FRAME.dot(self._K).dot(self._E[:, [0, 1, 3]])


def pygame_modules_have_loaded():
  return pygame.display.get_init() and pygame.font.get_init()

def draw_var(y, x, var, color, img, calibration, top_down):
  # otherwise drawing gets stupid
  var = max(1e-1, min(var, 0.7))

  varcolor = tuple(np.array(color)*0.5)
  draw_path(y - var, x, varcolor, img, calibration, top_down)
  draw_path(y + var, x, varcolor, img, calibration, top_down)


class ModelPoly(object):
  def __init__(self, model_path):
    if len(model_path.points) == 0:
      self.valid = False
      return

    self.poly = model_polyfit(model_path.points, _PATH_PINV)
    self.prob = model_path.prob
    self.std = model_path.std
    self.y = np.polyval(self.poly, _PATH_XD)
    self.valid = True

def extract_model_data(md):
  return ModelUIData(
    cpath=ModelPoly(md.model.path),
    lpath=ModelPoly(md.model.leftLane),
    rpath=ModelPoly(md.model.rightLane),
    lead=md.model.lead.dist,
    lead_std=md.model.lead.std,
    freepath=md.model.freePath)

def plot_model(m, VM, v_ego, curvature, imgw, calibration, top_down, top_down_color=216):
  # Draw bar representing position and distribution of lead car from unfiltered vision model
  if top_down is not None:
    _, _ = to_lid_pt(m.lead, 0)
    _, py_top = to_lid_pt(m.lead + m.lead_std, 0)
    px, py_bottom = to_lid_pt(m.lead - m.lead_std, 0)
    top_down[1][int(round(px - 4)):int(round(px + 4)), py_top:py_bottom] = top_down_color

  if calibration is None:
    return

  if m.cpath.valid:
    draw_path(m.cpath.y, _PATH_XD, YELLOW, imgw, calibration, top_down, YELLOW)
    draw_var(m.cpath.y, _PATH_XD, m.cpath.std, YELLOW, imgw, calibration, top_down)

    dpath_poly, _, _ = calc_desired_path(m.lpath.poly, m.rpath.poly, m.cpath.poly,
                                         m.lpath.prob, m.rpath.prob, m.cpath.prob, v_ego)
    dpath_poly = np.array(dpath_poly)

    dpath_y = np.polyval(dpath_poly, _PATH_X)
    draw_path(dpath_y, _PATH_X, RED, imgw, calibration, top_down, RED)

  if m.lpath.valid:
    color = (0, int(255 * m.lpath.prob), 0)
    draw_path(m.lpath.y, _PATH_XD, color, imgw, calibration, top_down, YELLOW)
    draw_var(m.lpath.y, _PATH_XD, m.lpath.std, color, imgw, calibration, top_down)

  if m.rpath.valid:
    color = (0, int(255 * m.rpath.prob), 0)
    draw_path(m.rpath.y, _PATH_XD, color, imgw, calibration, top_down, YELLOW)
    draw_var(m.rpath.y, _PATH_XD, m.rpath.std, color, imgw, calibration, top_down)

  if len(m.freepath) > 0:
    for i, p in enumerate(m.freepath):
      d = i*2
      px, py = to_lid_pt(d, 0)
      cols = [36, 73, 109, 146, 182, 219, 255]
      if p >= 0.4:
        top_down[1][int(round(px - 4)):int(round(px + 4)), int(round(py - 4)):int(round(py + 4))] = find_color(top_down[0], (0, cols[int((p-0.4)*10)], 0))
      elif p <= 0.2:
        top_down[1][int(round(px - 4)):int(round(px + 4)), int(round(py - 4)):int(round(py + 4))] = 192 #find_color(top_down[0], (192, 0, 0))

  # draw user path from curvature
  draw_steer_path(v_ego, curvature, BLUE, imgw, calibration, top_down, VM, BLUE)


def maybe_update_radar_points(lt, lid_overlay):
  ar_pts = []
  if lt is not None:
    ar_pts = {}
    for track in lt.liveTracks:
      ar_pts[track.trackId] = [track.dRel, track.yRel, track.vRel, track.aRel, track.oncoming, track.stationary]
  for ids, pt in ar_pts.viewitems():
    px, py = to_lid_pt(pt[0], pt[1])
    if px != -1:
      if pt[-1]:
        color = 240
      elif pt[-2]:
        color = 230
      else:
        color = 255
      if int(ids) == 1:
        lid_overlay[px - 2:px + 2, py - 10:py + 10] = 100
      else:
        lid_overlay[px - 2:px + 2, py - 2:py + 2] = color

def get_blank_lid_overlay(UP):
  lid_overlay = np.zeros((UP.lidar_x, UP.lidar_y), 'uint8')
  # Draw the car.
  lid_overlay[int(round(UP.lidar_car_x - UP.car_hwidth)):int(
    round(UP.lidar_car_x + UP.car_hwidth)), int(round(UP.lidar_car_y -
                                                      UP.car_front))] = UP.car_color
  lid_overlay[int(round(UP.lidar_car_x - UP.car_hwidth)):int(
    round(UP.lidar_car_x + UP.car_hwidth)), int(round(UP.lidar_car_y +
                                                      UP.car_back))] = UP.car_color
  lid_overlay[int(round(UP.lidar_car_x - UP.car_hwidth)), int(
    round(UP.lidar_car_y - UP.car_front)):int(round(
      UP.lidar_car_y + UP.car_back))] = UP.car_color
  lid_overlay[int(round(UP.lidar_car_x + UP.car_hwidth)), int(
    round(UP.lidar_car_y - UP.car_front)):int(round(
      UP.lidar_car_y + UP.car_back))] = UP.car_color
  return lid_overlay


def ui_thread(addr, frame_address):
  context = zmq.Context()

  # TODO: Detect car from replay and use that to select carparams
  CP = ToyotaInterface.get_params("TOYOTA PRIUS 2017", {})
  VM = VehicleModel(CP)

  CalP = np.asarray([[0, 0], [MODEL_INPUT_SIZE[0], 0], [MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]], [0, MODEL_INPUT_SIZE[1]]])
  vanishing_point = np.asarray([[MODEL_CX, MODEL_CY]])

  pygame.init()
  pygame.font.init()
  assert pygame_modules_have_loaded()

  if HOR:
    size = (640+384+640, 960)
    write_x = 5
    write_y = 680
  else:
    size = (640+384, 960+300)
    write_x = 645
    write_y = 970

  pygame.display.set_caption("openpilot debug UI")
  screen = pygame.display.set_mode(size, pygame.DOUBLEBUF)

  alert1_font = pygame.font.SysFont("arial", 30)
  alert2_font = pygame.font.SysFont("arial", 20)
  info_font = pygame.font.SysFont("arial", 15)

  camera_surface = pygame.surface.Surface((640, 480), 0, 24).convert()
  cameraw_surface = pygame.surface.Surface(MODEL_INPUT_SIZE, 0, 24).convert()
  cameraw_test_surface = pygame.surface.Surface(MODEL_INPUT_SIZE, 0, 24)
  top_down_surface = pygame.surface.Surface((UP.lidar_x, UP.lidar_y),0,8)

  frame = context.socket(zmq.SUB)
  frame.connect(frame_address or "tcp://%s:%d" % (addr, service_list['frame'].port))
  frame.setsockopt(zmq.SUBSCRIBE, "")

  carState = sub_sock(context, service_list['carState'].port, addr=addr, conflate=True)
  plan = sub_sock(context, service_list['plan'].port, addr=addr, conflate=True)
  carControl = sub_sock(context, service_list['carControl'].port, addr=addr, conflate=True)
  live20 = sub_sock(context, service_list['live20'].port, addr=addr, conflate=True)
  liveCalibration = sub_sock(context, service_list['liveCalibration'].port, addr=addr, conflate=True)
  live100 = sub_sock(context, service_list['live100'].port, addr=addr, conflate=True)
  liveTracks = sub_sock(context, service_list['liveTracks'].port, addr=addr, conflate=True)
  model = sub_sock(context, service_list['model'].port, addr=addr, conflate=True)
  test_model = sub_sock(context, 8040, addr=addr, conflate=True)
  liveMpc = sub_sock(context, service_list['liveMpc'].port, addr=addr, conflate=True)
  liveParameters = sub_sock(context, service_list['liveParameters'].port, addr=addr, conflate=True)

  v_ego, angle_steers, angle_steers_des, model_bias = 0., 0., 0., 0.
  params_ao, params_ao_average, params_stiffness, params_sr = None, None, None, None

  enabled = False

  gas = 0.
  accel_override = 0.
  computer_gas = 0.
  brake = 0.
  steer_torque = 0.
  curvature = 0.
  computer_brake = 0.
  plan_source = 'none'
  long_control_state = 'none'

  model_data = None
  test_model_data = None
  a_ego = 0.0
  a_target = 0.0

  d_rel, y_rel, lead_status  = 0., 0., False
  d_rel2, y_rel2, lead_status2 = 0., 0., False

  v_ego, v_pid, v_cruise, v_override = 0., 0., 0., 0.
  brake_lights = False

  alert_text1, alert_text2 = "", ""

  intrinsic_matrix = None

  calibration = None
  #img = np.zeros((FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0], 3), dtype='uint8')
  img = np.zeros((480, 640, 3), dtype='uint8')
  imgff = np.zeros((FULL_FRAME_SIZE[1], FULL_FRAME_SIZE[0], 3), dtype=np.uint8)
  imgw = np.zeros((160, 320, 3), dtype=np.uint8)  # warped image
  good_lt = None
  lid_overlay_blank = get_blank_lid_overlay(UP)
  img_offset = (0, 0)
  if vision_test:
    visiontest = VisionTest(FULL_FRAME_SIZE, MODEL_INPUT_SIZE, None)

  # plots
  name_to_arr_idx = { "gas": 0,
                      "computer_gas": 1,
                      "user_brake": 2,
                      "computer_brake": 3,
                      "v_ego": 4,
                      "v_pid": 5,
                      "angle_steers_des": 6,
                      "angle_steers": 7,
                      "steer_torque": 8,
                      "v_override": 9,
                      "v_cruise": 10,
                      "a_ego": 11,
                      "a_target": 12,
                      "accel_override": 13}

  plot_arr = np.zeros((100, len(name_to_arr_idx.values())))

  plot_xlims = [(0, plot_arr.shape[0]), (0, plot_arr.shape[0]), (0, plot_arr.shape[0]), (0, plot_arr.shape[0])]
  plot_ylims = [(-0.1, 1.1), (-5., 5.), (0., 75.), (-3.0, 2.0)]
  plot_names = [["gas", "computer_gas", "user_brake", "computer_brake", "accel_override"],
                ["angle_steers", "angle_steers_des", "steer_torque"],
                ["v_ego", "v_override", "v_pid", "v_cruise"],
                ["a_ego", "a_target"]]
  plot_colors = [["b", "b", "g", "r", "y"],
                 ["b", "g", "r"],
                 ["b", "g", "r", "y"],
                 ["b", "r"]]
  plot_styles = [["-", "-", "-", "-", "-"],
                 ["-", "-", "-"],
                 ["-", "-", "-", "-"],
                 ["-", "-"]]

  draw_plots = init_plots(plot_arr, name_to_arr_idx, plot_xlims, plot_ylims, plot_names, plot_colors, plot_styles, bigplots=True)

  while 1:
    list(pygame.event.get())

    screen.fill((64,64,64))
    lid_overlay = lid_overlay_blank.copy()
    top_down = top_down_surface, lid_overlay

    # ***** frame *****
    fpkt = recv_one(frame)
    yuv_img = fpkt.frame.image

    if fpkt.frame.transform:
      yuv_transform = np.array(fpkt.frame.transform).reshape(3,3)
    else:
      # assume frame is flipped
      yuv_transform = np.array([
        [-1.0,  0.0, FULL_FRAME_SIZE[0]-1],
        [ 0.0, -1.0, FULL_FRAME_SIZE[1]-1],
        [ 0.0,  0.0, 1.0]
      ])


    if yuv_img and len(yuv_img) == FULL_FRAME_SIZE[0] * FULL_FRAME_SIZE[1] * 3 // 2:
      yuv_np = np.frombuffer(yuv_img, dtype=np.uint8).reshape(FULL_FRAME_SIZE[1] * 3 // 2, -1)

      cv2.cvtColor(yuv_np, cv2.COLOR_YUV2RGB_I420, dst=imgff)
      cv2.warpAffine(imgff, np.dot(yuv_transform, _BB_TO_FULL_FRAME)[:2],
        (img.shape[1], img.shape[0]), dst=img, flags=cv2.WARP_INVERSE_MAP)

      intrinsic_matrix = eon_intrinsics
    else:
      img.fill(0)
      intrinsic_matrix = np.eye(3)

    if calibration is not None and yuv_img and vision_test:
      model_input_yuv = visiontest.transform_contiguous(yuv_img,
        np.dot(yuv_transform, calibration.model_to_full_frame).reshape(-1).tolist())
      cv2.cvtColor(
        np.frombuffer(model_input_yuv, dtype=np.uint8).reshape(MODEL_INPUT_SIZE[1] * 3 // 2, -1),
        cv2.COLOR_YUV2RGB_I420,
        dst=imgw)
    else:
      imgw.fill(0)
    imgw_test_model = imgw.copy()


    # ***** live100 *****
    l100 = recv_one_or_none(live100)
    if l100 is not None:
      v_ego = l100.live100.vEgo
      angle_steers = l100.live100.angleSteers
      model_bias = l100.live100.angleModelBias
      curvature = l100.live100.curvature
      v_pid = l100.live100.vPid
      enabled = l100.live100.enabled
      alert_text1 = l100.live100.alertText1
      alert_text2 = l100.live100.alertText2
      long_control_state = l100.live100.longControlState

    cs = recv_one_or_none(carState)
    if cs is not None:
      gas = cs.carState.gas
      brake_lights = cs.carState.brakeLights
      a_ego = cs.carState.aEgo
      brake = cs.carState.brake
      v_cruise = cs.carState.cruiseState.speed

    cc = recv_one_or_none(carControl)
    if cc is not None:
      v_override = cc.carControl.cruiseControl.speedOverride
      computer_brake = cc.carControl.actuators.brake
      computer_gas = cc.carControl.actuators.gas
      steer_torque = cc.carControl.actuators.steer * 5.
      angle_steers_des = cc.carControl.actuators.steerAngle
      accel_override = cc.carControl.cruiseControl.accelOverride

    p = recv_one_or_none(plan)
    if p is not None:
      a_target = p.plan.aTarget
      plan_source = p.plan.longitudinalPlanSource

    plot_arr[:-1] = plot_arr[1:]
    plot_arr[-1, name_to_arr_idx['angle_steers']] = angle_steers
    plot_arr[-1, name_to_arr_idx['angle_steers_des']] = angle_steers_des
    plot_arr[-1, name_to_arr_idx['gas']] = gas
    plot_arr[-1, name_to_arr_idx['computer_gas']] = computer_gas
    plot_arr[-1, name_to_arr_idx['user_brake']] = brake
    plot_arr[-1, name_to_arr_idx['steer_torque']] = steer_torque
    plot_arr[-1, name_to_arr_idx['computer_brake']] = computer_brake
    plot_arr[-1, name_to_arr_idx['v_ego']] = v_ego
    plot_arr[-1, name_to_arr_idx['v_pid']] = v_pid
    plot_arr[-1, name_to_arr_idx['v_override']] = v_override
    plot_arr[-1, name_to_arr_idx['v_cruise']] = v_cruise
    plot_arr[-1, name_to_arr_idx['a_ego']] = a_ego
    plot_arr[-1, name_to_arr_idx['a_target']] = a_target
    plot_arr[-1, name_to_arr_idx['accel_override']] = accel_override

    # ***** model ****

    # live model
    md = recv_one_or_none(model)
    if md:
      model_data = extract_model_data(md)

    if model_data:
      plot_model(model_data, VM, v_ego, curvature, imgw, calibration,
                 top_down)

    if test_model is not None:
      test_md = recv_one_or_none(test_model)
      if test_md:
        test_model_data = extract_model_data(test_md)

    if test_model_data:
      plot_model(test_model_data, VM, v_ego, curvature, imgw_test_model,
                 calibration, top_down, 215)

    # MPC
    mpc = recv_one_or_none(liveMpc)
    if mpc:
      draw_mpc(mpc, top_down)

    # LiveParams
    params = recv_one_or_none(liveParameters)
    if params:
      params_ao = params.liveParameters.angleOffset
      params_ao_average = params.liveParameters.angleOffsetAverage
      params_stiffness = params.liveParameters.stiffnessFactor
      params_sr = params.liveParameters.steerRatio
    # **** tracks *****

    # draw all radar points
    lt = recv_one_or_none(liveTracks)
    if lt is not None:
      good_lt = lt
    if good_lt is not None:
      maybe_update_radar_points(good_lt, top_down[1])

    # ***** live20 *****

    # live l20 from drived
    l20 = recv_one_or_none(live20)
    if l20 is not None:
      d_rel = l20.live20.leadOne.dRel + RDR_TO_LDR
      y_rel = l20.live20.leadOne.yRel
      lead_status = l20.live20.leadOne.status
      d_rel2 = l20.live20.leadTwo.dRel + RDR_TO_LDR
      y_rel2 = l20.live20.leadTwo.yRel
      lead_status2 = l20.live20.leadTwo.status

    lcal = recv_one_or_none(liveCalibration)
    if lcal is not None:
      calibration_message = lcal.liveCalibration
      extrinsic_matrix = np.asarray(calibration_message.extrinsicMatrix).reshape(3, 4)

      warp_matrix = np.asarray(calibration_message.warpMatrix2).reshape(3, 3)
      calibration = CalibrationTransformsForWarpMatrix(warp_matrix, intrinsic_matrix, extrinsic_matrix)

    # draw red pt for lead car in the main img
    if lead_status:
      if calibration is not None:
        dx, dy = draw_lead_on(img, d_rel, y_rel, img_offset, calibration, color=(192,0,0))
      # draw red line for lead car
      draw_lead_car(d_rel, top_down)

    # draw red pt for lead car2 in the main img
    if lead_status2:
      if calibration is not None:
        dx2, dy2 = draw_lead_on(img, d_rel2, y_rel2, img_offset, calibration, color=(192,0,0))
      # draw red line for lead car
      draw_lead_car(d_rel2, top_down)

    # *** blits ***
    pygame.surfarray.blit_array(camera_surface, img.swapaxes(0,1))
    screen.blit(camera_surface, (0, 0))

    # display alerts
    alert_line1 = alert1_font.render(alert_text1, True, (255,0,0))
    alert_line2 = alert2_font.render(alert_text2, True, (255,0,0))
    screen.blit(alert_line1, (180, 150))
    screen.blit(alert_line2, (180, 190))

    if calibration is not None and img is not None:
      cpw = warp_points(CalP, calibration.model_to_bb)
      vanishing_pointw = warp_points(vanishing_point, calibration.model_to_bb)
      pygame.draw.polygon(screen, BLUE, tuple(map(tuple, cpw)), 1)
      pygame.draw.circle(screen, BLUE, map(int, map(round, vanishing_pointw[0])), 2)

    if HOR:
      screen.blit(draw_plots(plot_arr), (640+384, 0))
    else:
      screen.blit(draw_plots(plot_arr), (0, 600))

    pygame.surfarray.blit_array(cameraw_surface, imgw.swapaxes(0, 1))
    screen.blit(cameraw_surface, (320, 480))

    pygame.surfarray.blit_array(cameraw_test_surface, imgw_test_model.swapaxes(0, 1))
    screen.blit(cameraw_test_surface, (0, 480))

    pygame.surfarray.blit_array(*top_down)
    screen.blit(top_down[0], (640,0))

    i = 0
    SPACING = 25

    # enabled
    enabled_line = info_font.render("ENABLED", True, GREEN if enabled else BLACK)
    screen.blit(enabled_line, (write_x, write_y + i * SPACING))
    i += 1

    # brake lights
    brake_lights_line = info_font.render("BRAKE LIGHTS", True, RED if brake_lights else BLACK)
    screen.blit(brake_lights_line, (write_x, write_y + i * SPACING))
    i += 1

    # speed
    v_ego_line = info_font.render("SPEED: " + str(round(v_ego, 1)) + " m/s", True, YELLOW)
    screen.blit(v_ego_line, (write_x, write_y + i * SPACING))
    i += 1

    # angle offset
    model_bias_line = info_font.render("MODEL BIAS: " + str(round(model_bias, 2)) + " deg", True, YELLOW)
    screen.blit(model_bias_line, (write_x, write_y + i * SPACING))
    i += 1

    # long control state
    long_control_state_line = info_font.render("LONG CONTROL STATE: " + str(long_control_state), True, YELLOW)
    screen.blit(long_control_state_line, (write_x, write_y + i * SPACING))
    i += 1

    # long mpc source
    plan_source_line = info_font.render("LONG MPC SOURCE: " + str(plan_source), True, YELLOW)
    screen.blit(plan_source_line, (write_x, write_y + i * SPACING))
    i += 1

    if params_ao is not None:
      i += 1
      angle_offset_avg_line = info_font.render("ANGLE OFFSET (AVG): " + str(round(params_ao_average, 2)) + " deg", True, YELLOW)
      screen.blit(angle_offset_avg_line, (write_x, write_y + i * SPACING))
      i += 1

      angle_offset_line = info_font.render("ANGLE OFFSET (INSTANT): " + str(round(params_ao, 2)) + " deg", True, YELLOW)
      screen.blit(angle_offset_line, (write_x, write_y + i * SPACING))
      i += 1

      angle_offset_line = info_font.render("STIFFNESS: " + str(round(params_stiffness * 100., 2)) + " %", True, YELLOW)
      screen.blit(angle_offset_line, (write_x, write_y + i * SPACING))
      i += 1

      steer_ratio_line = info_font.render("STEER RATIO: " + str(round(params_sr, 2)), True, YELLOW)
      screen.blit(steer_ratio_line, (write_x, write_y + i * SPACING))
      i += 1

    # this takes time...vsync or something
    pygame.display.flip()

def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Show replay data in a UI.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("ip_address", nargs="?", default="127.0.0.1",
                      help="The ip address on which to receive zmq messages.")

  parser.add_argument("--frame-address", default=None,
                      help="The frame address (fully qualified ZMQ endpoint for frames) on which to receive zmq messages.")
  return parser

if __name__ == "__main__":
  args = get_arg_parser().parse_args(sys.argv[1:])
  ui_thread(args.ip_address, args.frame_address)
