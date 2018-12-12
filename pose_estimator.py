''' Estimator head pose according to the facial landmarks '''
import numpy as np 
import cv2

class PoseEstimator:
    ''' Estimator head pose according to the facial landmarks '''

    def __init__(self, img_size=(480,640)):
        self.size = img_size

        # 3D model points
        self.model_points = np.array([
            (0.0, 0.0, 0.0),            # nose tip
            (0.0, -330.0, -65.0),       # chin
            (-225.0, 170.0, -135.0),    # left eye left corner
            (225.0, 170.0, -135.0),     # right eye right corner
            (-150.0, -150.0, -125.0),   # left mouth  corner
            (150.0, -150.0, -125.0)     # right mouth corner   
        ])/4.5

        self.model_points68 = self._get_full_model_points()

        # camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2 , self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
            [0, self.focal_length, self.camera_center[1]],
            [0,0,1]], dtype='double'
        )

        # assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013],[0.08560084],[-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383],[-2053.03596872]])

    def _get_full_model_points(self, filename='assets/model.txt'):
        ''' Get all 68 3D model points from file.  '''
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append((line))
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T 
        model_points [:-1] *= -1

        return model_points