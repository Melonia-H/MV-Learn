#!/usr/bin/python3.5
#-*- coding: UTF-8 -*-
import numpy as np
import os
import cv2
from skimage import transform 
import time

from keras.models import load_model
from device.cameradevice import DepthSenseCameraDevice
from utils.face_detect import *

face_anti_state = ['fake','real']

import tensorflow as tf
from keras import backend as K
from keras.models import load_model
import threading as t

class CNN:
    def __init__(self):
        self.cnn_model = load_model("./models/densenet_keras_depth.h5")
        self.cnn_model.predict(np.zeros([1, 224, 224, 1], dtype = np.float32)) # warmup
        self.session = K.get_session()
        self.graph = tf.get_default_graph()
        self.graph.finalize() # finalize

    def find_face_roi(self, depth, ir):
        image_ir = ir * (255 / 3840)
        image_ir = image_ir.astype(np.uint8)
        ir_image = cv2.cvtColor(image_ir, cv2.COLOR_GRAY2RGB)
        bbox  = face_detect(ir_image)
        if bbox is None:
            print('No Face')
            return None
        else:
            x1 = abs(int(bbox[0]))
            y1 = abs(int(bbox[1]))
            x2 = abs(int(bbox[2]))
            y2 = abs(int(bbox[3]))
            frame_depth = depth[y1:y2, x1:x2]
            frame_depth = np.resize(frame_depth, (224, 224))
            frame_depth_roi = frame_depth[np.newaxis,...,np.newaxis]
            return frame_depth_roi

    def query_cnn(self, depth, ir):
        start_time = time.clock()
        X = self.find_face_roi(depth, ir)
        if X is None:
            result = 'no face'
        else:
            with self.session.as_default():
                with self.graph.as_default():
                    prediction = self.cnn_model.predict(X)
            predict_ = prediction.max()
            predict_index = np.argmax(prediction)
            result = face_anti_state[predict_index]
            print(result)
        end_time = time.clock()
        print('---cost time ----', (end_time-start_time))
        return result

# 界面显示， 只有深度图
def display_ui(frame, text='', gesture_rate=''):
    image = np.zeros((540, 1100), np.uint8)

    img_ = frame * (255/np.uint32(1.450))
    img_ = img_.astype(np.uint8)
    image[50:530, 10:650] = img_

    cv2.putText(image, "DEPTH MAP",(130, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (0xFFFF, 0xFFFF, 0xFFFF), 1)
    cv2.putText(image, text, (660, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (0xFFFF, 0xFFFF, 0xFFFF), 1)
    cv2.imshow('DEMO', image)

# 模型预测一次空的数据，防止第一次预测模型非常慢
def main():
    device = DepthSenseCameraDevice()
    device.start()

    detect_status = False
    result = None
    cnn = CNN()

    while True:
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            detect_status = bool(1-detect_status)

        ret, depth, ir = device.get_depth_ir()
        if ret is False:
            print("Error while reading frame ")
            continue
        else:
            if detect_status:
                result = cnn.query_cnn(depth, ir)
            else:
                pass
            display_ui(frame=depth, text='Press s to start...' if result is None else result)
    device.stop()

if __name__ == '__main__':
    main()