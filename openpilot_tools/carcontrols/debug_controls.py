#!/usr/bin/env python
import struct
import zmq
import time
from common.numpy_fast import clip
from copy import copy
from selfdrive.services import service_list
from cereal import car
import selfdrive.messaging as messaging
from selfdrive.car.car_helpers import get_car


def steer_thread():
  context = zmq.Context()
  poller = zmq.Poller()

  logcan = messaging.sub_sock(context, service_list['can'].port)
  joystick_sock = messaging.sub_sock(context, service_list['testJoystick'].port, conflate=True, poller=poller)

  carstate = messaging.pub_sock(context, service_list['carState'].port)
  carcontrol = messaging.pub_sock(context, service_list['carControl'].port)
  sendcan = messaging.pub_sock(context, service_list['sendcan'].port)

  button_1_last = 0
  enabled = False

  CI, CP = get_car(logcan, sendcan, None)

  CC = car.CarControl.new_message()
  joystick = messaging.recv_one(joystick_sock)

  while True:

    # send
    for socket, event in poller.poll(0):
      if socket is joystick_sock:
        joystick = messaging.recv_one(socket)

    CS = CI.update(CC)

    # Usually axis run in pairs, up/down for one, and left/right for
    # the other.
    actuators = car.CarControl.Actuators.new_message()

    axis_3 = clip(-joystick.testJoystick.axes[3] * 1.05, -1., 1.)          # -1 to 1
    actuators.steer = axis_3
    actuators.steerAngle = axis_3 * 43.   # deg
    axis_1 = clip(-joystick.testJoystick.axes[1] * 1.05, -1., 1.)          # -1 to 1
    actuators.gas = max(axis_1, 0.)
    actuators.brake = max(-axis_1, 0.)

    pcm_cancel_cmd = joystick.testJoystick.buttons[0]
    button_1 = joystick.testJoystick.buttons[1]
    if button_1 and not button_1_last:
      enabled = not enabled

    button_1_last = button_1

    #print "enable", enabled, "steer", actuators.steer, "accel", actuators.gas - actuators.brake

    hud_alert = 0
    audible_alert = 0
    if joystick.testJoystick.buttons[2]:
      audible_alert = "beepSingle"
    if joystick.testJoystick.buttons[3]:
      audible_alert = "chimeRepeated"
      hud_alert = "steerRequired"

    CC.actuators.gas = actuators.gas
    CC.actuators.brake = actuators.brake
    CC.actuators.steer = actuators.steer
    CC.actuators.steerAngle = actuators.steerAngle
    CC.hudControl.visualAlert = hud_alert
    CC.hudControl.setSpeed = 20
    CC.cruiseControl.cancel = pcm_cancel_cmd
    CC.enabled = enabled
    CI.apply(CC)

    # broadcast carState
    cs_send = messaging.new_message()
    cs_send.init('carState')
    cs_send.carState = copy(CS)
    carstate.send(cs_send.to_bytes())
  
    # broadcast carControl
    cc_send = messaging.new_message()
    cc_send.init('carControl')
    cc_send.carControl = copy(CC)
    carcontrol.send(cc_send.to_bytes())


    # Limit to 100 frames per second
    time.sleep(0.01)


if __name__ == "__main__":
  steer_thread()
