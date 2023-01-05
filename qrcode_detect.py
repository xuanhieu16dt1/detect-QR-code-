# Import libraries
import cv2
import numpy as np
import time
import pyrealsense2
from realsense_camera import *

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers
def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    label = str(classes[class_id])
    color = (255,0,255)
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label + " %.2f"%confidence , (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
def drawBox(image, points):
    height, width = image.shape[:2]
    for (label, xi,yi, wi, hi) in points:
        center_x = int(xi * width)
        center_y = int(yi * height)
        w = int(wi * width)
        h = int(hi * height)
        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 1)
    return
def savePredict(name, text):
    textName = name + '.txt'
    with open(textName, 'w+') as groundTruth:
        groundTruth.write(text)
        groundTruth.close()
def loadWeight(weights = "./yolo-config_10000.weights" ,cfg ="./yolo-config.cfg", class_name = "./obj.names" ):
    net = cv2.dnn.readNet(weights,cfg ) 
    classes = None
    with open(class_name, 'r') as f: # Edit CLASS file
        classes = [line.strip() for line in f.readlines()]
    return net, classes

def drawImage(image, indices, boxes, class_ids, confidences, classes, depth_frame = []):
    x = 0
    y = 0
    w = 0
    h = 0
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_prediction(image, class_ids[i],confidences[i], round(x), round(y), round(x + w), round(y + h),classes)

        x_center = x+w/2
        y_center = y+h/2
        if depth_frame == []:
            distance = "NaN"
        else:
            distance = depth_frame[ int(y_center), int(x_center)]
        cv2.putText(image, "x: " + str(x_center), (int(x+20),int(y-20)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
        cv2.putText(image, "y: " + str(y_center), (int(x+20),int(y-40) ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
        cv2.putText(image, "z: " + str(distance)+" mm", (int(x+20),int(y-60) ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1)
    #savePredict(pathSave, name, textPre) # Doi thanh con tro ve dia chi cua anh
        
    scale_percent = 100
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(src=image, dsize=(width,height))

    return image

def detect(image, net):

    scale = 0.00392
    Width = image.shape[1]
    Height = image.shape[0]
    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    #print(outs)
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.2
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            #print(scores)
            class_id = np.argmax(scores)
            #print('b')
            #print(class_id)
            confidence = scores[class_id]
            #print(confidence)
            if confidence > 0.25:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                #print(w,h,x,y)
                class_ids.append(class_id)
                """if confidence < 0.6:
                    class_ids.append(2)""" #change
                confidences.append(float(confidence))
                #print(class_ids)
                boxes.append([x, y, w, h])
                #print(boxes)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)


    return image, indices, boxes, class_ids, confidences

if __name__ == "__main__":
    # cap = cv2.VideoCapture(0)
    rs = RealsenseCamera()
    # net, classes = loadWeight(weights = "./qr_code_yolov4.weights",cfg ="./yolov4.cfg",class_name = "./obj.names")
    net, classes = loadWeight(weights = "./yolov4-tiny-custom_best.weights",cfg ="./yolov4-tiny.cfg",class_name = "./obj.names")
    ret = True
    while ret:
        # ret, color_frame = cap.read()
        ret, bgr_frame, depth_frame = rs.get_frame_stream()
        if ret == True:
            start = time.time()
            image, indices, boxes, class_ids, confidences = detect(bgr_frame, net)
            image = drawImage(image, indices, boxes, class_ids, confidences, classes)

            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            end = time.time()
            print("time: " , end - start)
        else:
            break
    # cap.release()
    cv2.destroyAllWindows()
