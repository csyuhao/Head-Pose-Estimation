''' Demo code shows how to estimate hunman head pose.
Firstly, human face is detected by a dectector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark detection.
The facial landmark detection is done by a custom Convolutional Neural
Network trained with TensorFlow. After that, head pose is estimated by
solving a PnP problem.
'''
import numpy as np 
import cv2

from multiprocessing import Process, Queue 
from mark_detector import MarkDetector
from os_detector import detect_os
from sta


# multiprocessing may not work on Windows and macos, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128

def get_face(detector, img_queue, box_queue):
    '''Get face from image queue. this function is used for multiprocessing. '''

    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
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

    # introduce scalar stablilizers for opse.

    pose_stabilizers = [Stabilizer(
        state_num = 2,
        measure_num = 1,
        cov_process = 0.1,
        cov_measure = 0.1) for _ in range(6)]
    
if __name__ == "__main__":
    main()