'''
    using Kalman Filter as a point stabilizer to stabilize a 2D point
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