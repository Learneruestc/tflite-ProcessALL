import os
import time
import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

# model_path = "mobilenet_v1_0.25_128_quant.tflite"
# model_path = "mobilenet_v1_1.0_192_quant.tflite"
model_path = "mobilenet_v1_1.0_224_quant.tflite"
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

img_exp = np.expand_dims(img, axis=0)
#img_exp = np.repeat(np.expand_dims(img, axis=0), 128, axis=0)
print(img_exp.shape)

inter.set_tensor(input_details[0]['index'], img_exp)

time_start =time.time()
inter.invoke()
# for i in range(100):
#     inter.invoke()
time_end = time.time()
print('spendTime=',time_end-time_start)

output_data = inter.get_tensor(output_details[0]['index'])
print(output_data.shape)
result = np.squeeze(output_data)
# np.save("v1_0.25_128_quant/iter_0_quant.npy", result , allow_pickle=False)
print('np.argmax(result):',np.argmax(result))

# np.savetxt("v1_1.0_224_quant/iter_2.txt", result)


