# comma-speed-challenge
1. Background Research
   * Prior speed challenge work
   * Video classification
2. Experiments
   * Full-fat CNN-LSTM
   * 224-224 transfer learning
   * Optical-flow CNN-LSTM
   * Optical-flow flattened CNN
   * Learning frame ordering
   * Crop and Op-Flow flattened CNN

3. Data Gathering
      * Extracting files
      * Identifying data of interest
        * hevc
        * rlog
       * Extracting data from files of interest
        * Struggles with  Capnp on Mac
        * Docker Attempt
        * Ubuntu VM on Windows
        * FFMPEG
        * Log Reader port to Python 3
        * Resampling Speed
        * Op-flow
        * Augmentation
      * Results
        * Data Quantity
        * Formatting
        * HDF5 Array
4. Data observations
	* Bi/Tri-modal Speed distribution
	* Varying brightness
	* FFMPEG compression aberations
	* ~24fps vs 20fps
	* Frames less frequent than .rlogs
5. Transfering to Comma Data
6. Validation Strategies
   * Shuffled train-test split
   * Unshuffled train-test split 
   * Unshuffled K-folds
7. Results
8. Paths to improvement
