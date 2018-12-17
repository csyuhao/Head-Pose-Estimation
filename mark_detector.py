''' Human facial landmark detector based on Convolutional Neural Network '''
import tensorflow as tf 
import numpy as np 
import cv2
import util

class FaceDetector:
    
    ''' Detect human face from image '''

    def __init__(self, 
                dnn_proto_text=r'assets/deploy.prototxt', 
                dnn_model=r'assets/res10_300x300_ssd_iter_140000.caffemodel'):
        '''
        Initialization 
        model: https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
        '''
        
        self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_text, dnn_model)
        self.detection_result = None

    def get_faceboxes(self, image, threshold=0.5):
        '''
            Get the bounding box of faces in image using dnn.
        '''
        rows, cols, _ = image.shape

        confidences = []
        faceboxes = []

        self.face_net.setInput(cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0), False, False))
        '''
            note:
                blobFromImage(const std::vector< Mat > & 	images,
                                double 	scalefactor = 1.0,
                                Size 	size = Size(),
                                const Scalar & 	mean = Scalar(),
                                bool 	swapRB = true,
                                bool 	crop = false,
                            )	
                parameters:
                    images:         input image (with 1-, 3- or 4-channels).
                    scalefactor:	multiplier for image values.
                    size:	        spatial size for output image
                    mean:	        scalar with mean values which are subtracted from channels. Values are intended to be in (mean-R, mean-G, mean-B) order if image has BGR ordering and swapRB is true.
                    swapRB:	        flag which indicates that swap first and last channels in 3-channel image is necessary.
                    crop:   	    flag which indicates whether image will be cropped after resize or not
            params from https://github.com/opencv/opencv/tree/master/samples/dnn
        '''
        detections = self.face_net.forward()

        for result in detections[0,0,:,:]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)
                confidences.append(confidence)
                faceboxes.append([x_left_bottom, y_left_bottom, x_right_top, y_right_top])
        
        self.detection_result = [faceboxes, confidences]
        return confidences, faceboxes

class MarkDetector:

    ''' Facial landmark detector by Convolutional Neural Network '''

    def __init__(self, mark_model=r'assets/frozen_inference_graph.pb'):
        ''' Intialization '''
        # a face detector is required for mark detection 
        self.face_detector = FaceDetector()

        self.cnn_input_size = 128
        self.marks = None

        # Get a Tensorflow session to ready to do landmark detection
        # load a (frozen) Tensorflow model to memory.
        detection_graph = tf.Graph()

        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(mark_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph = detection_graph
        self.sess = tf.Session(graph=detection_graph)

    def extract_cnn_facebox(self, image):
        ''' Extract face area from image. '''
        _, raw_boxes = self.face_detector.get_faceboxes(image=image, threshold=0.9)
        '''
            raw_boxes: the location of boxes.
        '''

        for box in raw_boxes:
            # Move box down.
            # height: box[3] - box[1] width: box[2] - box[0]
            # diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            # offset_y = int(abs(diff_height_width / 2))
            # box_moved = util.move_box(box, [0, offset_y])

            # Move box square.
            facebox = util.get_square_box(box)

            if util.box_in_image(facebox, image):
                return facebox
        return None
    
    def detect_marks(self, image_np):
        '''detect marks from image'''
        # get result tensor by its name.
        logits_tensor = self.graph.get_tensor_by_name("logits/BiasAdd:0")

        # actual detection.
        predictions = self.sess.run(
            logits_tensor,
            feed_dict={"input_image_tensor:0": image_np}
        )

        # convert predictions to landmarks.
        marks = np.array(predictions).flatten()
        marks = np.reshape(marks, (-1, 2))
        '''
        note:
        flatten function:
            return a copy of the array collapsed into one dimension.
            default param 'C' means to flatten in row-major order(1 row).
        reshape function:
            shape = (-1, 2) : the number of column is 2, row's is undefined.
            shape = (2, -1) : the number of row is 2, colum's is undefined.
        '''
        return marks