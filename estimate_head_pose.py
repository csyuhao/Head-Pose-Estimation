''' Demo code shows how to estimate hunman head pose.
Firstly, human face is detected by a dectector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark detection.
The facial landmark detection is done by a custom Convolutional Neural
Network trained with TensorFlow. After that, head pose is estimated by
solving a PnP problem.
'''
from multiprocessing import Process, Queue

import cv2
import numpy as np

from mark_detector import MarkDetector
from os_detector import detect_os
from stabilizer import Stabilizer
from pose_estimator import PoseEstimator

# multiprocessing may not work on Windows and macos, check OS for safety.
# detect_os()

CNN_INPUT_SIZE = 128

def get_face(detector, img_queue, box_queue):
    '''Get face from image queue. this function is used for multiprocessing. '''

    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        '''
        the box of face.
        '''
        box_queue.put(box)

def main():

    ''' main function'''

    # video source form webcam or video file.
    video_src = 0
    cam = cv2.VideoCapture(video_src)
    _, sample_frame = cam.read()

    # introduce mark_detector to detect landmarks
    mark_detector = MarkDetector()

    # setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(mark_detector, img_queue, box_queue))

    box_process.start()

    # introduce pos estimator to solve pose. Get one frames to setup the
    # estimator according to the images size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # introduce scalar stablilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num = 2,
        measure_num = 1,
        cov_process = 0.1,
        cov_measure = 0.1) for _ in range(6)]
    '''
    pose_stabilizers has six the same elements.
    '''
    
    while True:
        # Read frame, crop it, flip it, suits your needs.
        frame_got, frame = cam.read()
        if frame_got is False:
            break
        
        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # If frame comes from webcam, flip it so it looks like a mirror.
        if video_src == 0:
            # flipcode > 0: horizontal flip; flipcode = 0: vertical flip; flipcode < 0: horizental and vertical flip
            frame = cv2.flip(frame, 2)

            # Pose estimation by 3 steps.
            # 1. detect faces.
            # 2. detect landmarks.
            # 3. estimate pose.

            # feed frame to image queue.
            img_queue.put(frame)

            # get face from box queue.
            facebox = box_queue.get()

            if facebox is not None:
                # detect landmarks from image 128 * 128
                face_img = frame[facebox[1]: facebox[3], facebox[0] : facebox[2]]
                '''
                cut off the area of face.
                '''
                face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                '''
                BGR -> RGB
                '''
                marks = mark_detector.detect_marks(face_img)

                # covert the marks locations from local CNN to global image.
                marks *= (facebox[2] - facebox[0])  # the length of square.
                marks[:, 0] += facebox[0]           
                marks[:, 1] += facebox[1]

                # uncomment following line to show raw marks.
                # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))

                # try pose estimation with 68 points.
                pose = pose_estimator.solve_pose_by_68_points(marks)

                # stabilize the pose.
                stabile_pose = []
                pose_np = np.array(pose).flatten()

                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    stabile_pose.append(ps_stb.state[0])
                '''
                zip() will transfer two lists to tuples whose size is equal to the shorter of two lists 
                '''
                stabile_pose = np.reshape(stabile_pose, (-1, 3))

                # uncomment following line to draw pose annotaion on frame.
                # pose_estimator.draw_annotation_box(frame, pose[0], pose[1], color=(255, 128, 128))

                # uncomment following line to draw stabile pose annotation on frame.
                pose_estimator.draw_annotion_box(frame, stabile_pose[0], stabile_pose[1], color=(128, 255, 128))
        # show preview
        cv2.imshow("Preview", frame)
        if cv2.waitKey(10) == 27:
            break
    
    # clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()

if __name__ == "__main__":
    main()
