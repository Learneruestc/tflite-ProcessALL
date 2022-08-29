# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 10:25:43 2022

@author: liujun
"""

import time
import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


model_path = "./quant_model/tflite_model.tflite"
inter = tf.lite.Interpreter(model_path=model_path)
inter.allocate_tensors()
input_details = inter.get_input_details()
output_details = inter.get_output_details()

#用cv函数自己读的图像格式和X_test的不一致，因此要用X_test的读入
# img = cv2.imread('9-1.jpg')
# img = cv2.resize(img, (28, 28))
# img0=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
res=[]
for i in range(0, 21):
    img0 = X_test[i]
# img0=X_test[9]

    img1 = cv2.resize(img0, (168, 168))
    # cv2.imshow('image',img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    
    img = img0.astype(np.float32)
    # img=np.float32(img0)
    
    img_exp = np.expand_dims(img, axis=0)
    print(img_exp.shape)
    
    inter.set_tensor(input_details[0]['index'], img_exp)
    
    time_start =time.time()
    inter.invoke()
    time_end = time.time()
    print(time_end-time_start)
    
    output_data = inter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
    result = np.squeeze(output_data)
    print('np.argmax(result):',np.argmax(result))
    res.append(np.argmax(result))
    
    str1="mnist_fp32/iter_"+ '%s.txt' % (i)
    np.savetxt(str1, result)









