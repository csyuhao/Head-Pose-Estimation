'''
using Kalman Filter as a point stabilizer to stabilize a 2D point
图像滤波：
    在尽量保留图像细节特征的条件下对目标图像的噪声进行抑制。
'''

import numpy as np
import cv2


class Stabilizer:
    ''' using Kalman Filter as a point stabilizer. '''

    def __init__(self, state_num=4, measure_num=2, cov_process=0.0001, cov_measure=0.1):
        ''' Initializetion '''
        # Currently we only support scalar and point, so check usr input first.
        assert state_num == 4 or state_num == 2, "only scalar and point supported, check state_num please."
        # store parameters.
        self.state_num = state_num
        self.measure_num = measure_num

        # The filter itself.
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)

        # Store the parameters.
        self.state = np.zeros((state_num, 1), dtype=np.float32)

        # Store the measurement result.
        self.measurement = np.zeros((measure_num, 1), dtype=np.float32)

        # Store the prediction.
        self.prediction = np.zeros((state_num, 1), dtype=np.float32)

        # Kalman parameters setup for scalar.
        if self.measure_num == 1:
            self.filter.transitionMatrix = np.array([[1, 1], [0, 1]], dtype=np.float32)
            self.filter.measurementMatrix = np.array([[1, 1]], dtype=np.float32)
            self.filter.processNoiseCov = np.array([[1, 0], [0, 1]], dtype=np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1]], dtype=np.float32) * cov_measure

        # Kalman parameters setup for point.
        if self.measure_num == 2:
            self.filter.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
            self.filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1, 0], [0, 1]], dtype=np.float32) * cov_measure

    def update(self, measurement):
        '''update the filter'''
        # make kalman prediction
        self.prediction = self.filter.predict()

        # get new measurement
        if self.measure_num == 1:
            self.measurement = np.array([[np.float32(measurement[0])]])
        else:
            self.measurement = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])

        # correct according to measurement
        self.filter.correct(self.measurement)

        # update state value.
        self.state = self.filter.statePost
