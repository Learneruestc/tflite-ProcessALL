# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 14:49:38 2022

@author: liujun
"""

import os
import time
import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()


# model_path = "mobilenet_v1_0.25_128.tflite"
# model_path = "mobilenet_v1_1.0_192.tflite"
model_path = "mobilenet_v1_1.0_224.tflite"

# inter = tf.contrib.lite.Interpreter(model_path=model_path)
inter = tf.lite.Interpreter(model_path=model_path)
inter.allocate_tensors()
input_details = inter.get_input_details()
output_details = inter.get_output_details()

# img = cv2.imread('test2.png')
# img = cv2.imread('ILSVRC2012_val_00000001.jpg')
# img = cv2.imread('ILSVRC2012_val_00000004.jpg')
img = cv2.imread('ILSVRC2012_val_0000000653.jpg')

img = cv2.resize(img, (224,224))

cv2.imshow('image',img)
cv2.waitKey(0)

cv2.destroyAllWindows()

img = np.float32(img)
# weight_tensor = tf.cast(weight_tensor, dtype=tf.float32)
img_exp = np.expand_dims(img, axis=0)
# img_exp = np.float32(img_exp)
#img_exp = np.repeat(np.expand_dims(img, axis=0), 128, axis=0)
print(img_exp.shape)

inter.set_tensor(input_details[0]['index'], img_exp)

time_start =time.time()
for i in range(100):
    inter.invoke()
time_end = time.time()
print(time_end-time_start)

output_data = inter.get_tensor(output_details[0]['index'])
print(output_data.shape)
result = np.squeeze(output_data)

print('np.argmax(result):',np.argmax(result))

# 保存
# np.save("v1_0.25_128_fp32/iter_0.tensor", result , allow_pickle=False)
# with open('v1_0.25_128_fp32/iter_0.txt','w')as f:
#     f.write(result)
    
# np.save("v1_0.25_128_fp32/iter_01.txt", result,fmt='%d')
# np.savetxt("v1_0.25_128_fp32/iter_011.txt", result,fmt='%d')

# np.savetxt("v1_1.0_224_fp32/iter_1.txt", result)

