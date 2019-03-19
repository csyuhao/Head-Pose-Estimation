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
import util

from mark_detector import MarkDetector
from pose_estimator import PoseEstimator

CNN_INPUT_SIZE = 128


def get_face(img_queue, box_queue):
    '''
        Get face from image queue. this function is used for multiprocessing.
        introduce mark_detector to detect landmarks
    '''
    mark_detector = MarkDetector()
    while True:
        image = img_queue.get()
        boxes = mark_detector.extract_cnn_facebox(image)
        box_queue.put(boxes)  # the boxes of faces.


def main():

    ''' main function'''

    img_dir = "img_dir/"     # image file dir

    # setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_list = []
    for filename in os.listdir(img_dir):
        frame = cv2.imread(img_dir + filename)
        img_list.append(frame)
        img_queue.put(frame)
    box_process = Process(target=get_face, args=(img_queue, box_queue))
    box_process.start()

    # introduce mark_detector to detect landmarks
    mark_detector = MarkDetector()

    while True:
        # Crop it if frame is larger than expected.
        # frame = frame[0:480, 300:940]

        # flipcode > 0: horizontal flip; flipcode = 0: vertical flip; flipcode < 0: horizental and vertical flip
        # frame = cv2.flip(frame, 2)

        if len(img_list) != 0:
            frame = img_list.pop(0)
            raw_frame = frame.copy()
            # introduce pos estimator to solve pose. Get one frames to setup the
            # estimator according to the images size.
            height, width = frame.shape[:2]
            pose_estimator = PoseEstimator(img_size=(height, width))
        else:
            break

        # Pose estimation by 3 steps.
        # 1. detect faces.
        # 2. detect landmarks.
        # 3. estimate pose.

        # get face from box queue.
        faceboxes = box_queue.get()
        # print("the length of facebox is " + str(len(faceboxes)))
        for facebox in faceboxes:
            # detect landmarks from image 128 * 128
            face_img = frame[facebox[1]: facebox[3], facebox[0]: facebox[2]]    # cut off the area of face.

            face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)    # BGR -> RGB
            marks = mark_detector.detect_marks(face_img)

            # covert the marks locations from local CNN to global image.
            marks[:, 0] *= (facebox[2] - facebox[0]) # the width of rectangle.
            marks[:, 1] *=  (facebox[3] - facebox[1])  # the length of rectangle.
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            # uncomment following line to show raw marks.
            # util.draw_marks(frame, marks, color=(0, 255, 0))
            # util.draw_faceboxes(frame, facebox)

            # try pose estimation with 68 points.
            pose = pose_estimator.solve_pose_by_68_points(marks)
           
            # pose[0] is rotation_vector, and pose[1] is translation_vector
            
            pose_np = np.array(pose).flatten()
            pose = np.reshape(pose_np, (-1, 3))

            angle = util.get_angle(pose[0])
            pose_estimator.get_warp_affined_image(frame, facebox, marks, angle[2])

        # show preview
        cv2.imshow("Preview", frame)
        cv2.imshow("raw_frame", raw_frame)
        if cv2.waitKey(0) == 27:
            continue

    # clean up the multiprocessing process.
    box_process.terminate()
    box_process.join()


if __name__ == "__main__":
    main()
