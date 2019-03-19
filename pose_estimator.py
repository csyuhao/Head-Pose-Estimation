''' Estimator head pose according to the facial landmarks '''
import numpy as np
import cv2
import math


class PoseEstimator:
    ''' Estimator head pose according to the facial landmarks '''

    def __init__(self, img_size=(480, 640)):
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

        self.model_points_68 = self._get_full_model_points()

        # camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [
                [self.focal_length, 0, self.camera_center[0]],
                [0, self.focal_length, self.camera_center[1]],
                [0, 0, 1]
            ], dtype='double')

        # assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        self.t_vec = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])
        '''
        the detail imformation listed in https://www.cnblogs.com/Jessica-jie/p/6596450.html
        '''

    def _get_full_model_points(self, filename='assets/model.txt'):
        ''' Get all 68 3D model points from file.  '''
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append((line))
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        model_points[:-1] *= -1

        return model_points

    def solve_pose_by_68_points(self, image_points):
        '''
        solve pose from all the 68 image points.
        return (rotation_vector, translation_vector) as pose.
        '''

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs
            )
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68, image_points, self.camera_matrix, self.dist_coeefs,
            rvec=self.r_vec, tvec=self.t_vec, useExtrinsicGuess=True
        )
        rotation_vector.dtype = 'float32'
        translation_vector.dtype = 'float32'
        '''
        bool solvePnP(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess = false,int flags = SOLVEPNP_ITERATIVE)
        params:
            objectPoints:   object points array in world coordinate space.
            imagePoints:    the points in image.
            cameraMatrix:   the camera matrix.
            distCoeffs:     distortion coefficient of 4, 5, 8 or 12 elements.
            rvec:           Output rotation vector (see  Rodrigues() ) that, together with  tvec , brings points from the model coordinate system to the camera coordinate system.
            tvec:           Output translation vector.
            useExtrinsicGuess:  If true (1), the function uses the provided  rvec and  tvec values as initial approximations of the rotation and translation vectors, respectively, and further optimizes them.
        '''
        return (rotation_vector, translation_vector)

    def get_warp_affined_image(self, frame, box, marks, angle):

        img_box = frame[box[1]: box[3], box[0]: box[2]]
        cols = box[2] - box[0]
        rows = box[3] - box[1]
        angle = angle * 180 / math.pi
        print(angle)
        if angle < 0:
            angle = 90 + angle
        elif angle > 90:
            angle = angle - 90
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)

        img_box = cv2.warpAffine(img_box, M, (cols, rows))

        frame[box[1]: box[3], box[0]: box[2]] = img_box
