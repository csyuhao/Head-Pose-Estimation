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

        self.model_points_68 = self._get_full_model_points()

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
        '''
        the datail imformation listed in https://www.cnblogs.com/Jessica-jie/p/6596450.html
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
        
        return (rotation_vector, translation_vector)

    def draw_annotion_box(self, image, rotation_vector, translation_vector, color = (255,255,255), line_width = 2):
        '''Draw a 3D box as annoation of pose'''

        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # map to 2d image points.
        (point_2d, _) = cv2.projectPoints(point_3d, rotation_vector, translation_vector, self.camera_matrix,self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(point_2d[8]), color, line_width, cv2.LINE_AA)
