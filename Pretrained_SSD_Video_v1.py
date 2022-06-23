import cv2
import numpy as np

# we are not going to bother with objects less than 50% probability
THRESHOLD = 0.2
# the lower the value: the fewer bounding boxes will remain
SUPPRESSION_THRESHOLD = 0.1
SSD_INPUT_SIZE = 300


# read the class labels
def construct_class_names(file_name='class_names'):
    with open(file_name, 'rt') as file:                                    # rt means read as text
        names = file.read().rstrip('\n').split('\n')                       # rstrip removes whitespace , split helps reduce string obtained to a list

    return names

def show_detected_objects(img, boxes_to_keep, all_bounding_boxes, object_names, class_ids):
    for index in boxes_to_keep:
        box = all_bounding_boxes[0]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=3)
        cv2.putText(img, object_names[class_ids[0] - 1].upper(), (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)


class_names = construct_class_names()

capture = cv2.VideoCapture('OrangesVideo.mp4')                                   # for reading video

neural_network = cv2.dnn_DetectionModel('ssd_weights.pb', 'ssd_mobilenet_coco_cfg.pbtxt')
# define whether we run the algorithm with CPU or with GPU
# WE ARE GOING TO USE CPU !!!
neural_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)             # Backend = Open CV
neural_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)                  # Running on CPU , ofc GPU faster but my laptop doesn't have it :)
neural_network.setInputSize(SSD_INPUT_SIZE, SSD_INPUT_SIZE)                 # Resizing
neural_network.setInputScale(1.0/127.5)                                     # Normalization
neural_network.setInputMean((127.5, 127.5, 127.5))                          # OpenCV deals with BGR , while our image is RGB so we swap R and G channels
neural_network.setInputSwapRB(True)


### Reading and Analysing the video
while True:

    is_grabbed, frame = capture.read()

    if not is_grabbed:
        break                                                                 # break when no frames left in video

    class_label_ids, confidences, bbox = neural_network.detect(frame)         # applying ssd on each frame
    bbox = list(bbox)                                                         # bbox is numpy array , while NMS (non max suppression method) needs list
    confidences = np.array(confidences).reshape(1, -1).tolist()[0]            # we print confidences and observe it is stored in numpy 2d array
                                                                              # we reshape it and convert it to a list

    # these are the indexes of the bounding boxes we have to keep
    box_to_keep = cv2.dnn.NMSBoxes(bbox, confidences, THRESHOLD, SUPPRESSION_THRESHOLD)
    show_detected_objects(frame, box_to_keep, bbox, class_names, class_label_ids)
    org = (00,100)
    cv2.putText(frame, str(class_names[class_label_ids[0] - 1].upper()) + " = " + str((class_label_ids.tolist()).count(class_label_ids[0])), org,  # counting objects
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 255, 0), 1)

    cv2.imshow('SSD Algorithm', frame)
    cv2.waitKey(1)

capture.release()
cv2.destroyAllWindows()
