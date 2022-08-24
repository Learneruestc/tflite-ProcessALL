# -*- coding: utf-8 -*-
# @Time    : 2020.12.13
# @Author  : wuruidong
# @Email   : wuruidong@hotmail.com
# @FileName: mobilenet_tf.py
# @Software: python
# @Cnblogs : https://www.cnblogs.com/ruidongwu

import cv2
import numpy as np
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
'''
Library Version:
python-opencv==3.4.2.17
tensorflow==1.12.0 or tf_nightly

'''

'''
Location number is obtained by netron (https://github.com/lutzroeder/netron).
Thanks the authors for providing such a wonderful tool.
# stage 1 (install): pip3 install netron
# stage 2 (start with brower): netron -b
# stage 3 (enter local ip): http://localhost:8080/
# stage 4 (open tflite file): mobilenet_v1_0.25_128_quant.tflite
# stage 5 (record location number)
'''
input_location =    np.array([88, 7,  33, 37, 39, 43, 45, 49, 51, 55, 57, 61, 63, 67, 69, 73, 75, 79, 81, 85, 9,  13, 15, 19, 21, 25, 27, 0], dtype=int)
weight_location =   np.array([8,  35, 38, 41, 44, 47, 50, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 11, 14, 17, 20, 23, 26, 29, 32, 3], dtype=int)
bias_location =     np.array([6,  34, 36, 40, 42, 46, 48, 52, 54, 58, 60, 64, 66, 70, 72, 76, 78, 82, 84, 10, 12, 16, 18, 22, 24, 28, 30, 2], dtype=int)
output_location =   np.array([7,  33, 37, 39, 43, 45, 49, 51, 55, 57, 61, 63, 67, 69, 73, 75, 79, 81, 85, 9,  13, 15, 19, 21, 25, 27, 31, 1], dtype=int)

'''
load tflite model from local file.
'''
def load_tflite(model_path=''):
    # inter = tf.contrib.lite.Interpreter(model_path=model_path)
    inter = tf.lite.Interpreter(model_path=model_path) # pip install tf-nightly
    inter.allocate_tensors()
    return inter

'''
load image with img_file name
'''
def load_img(img_file=''):
    img = cv2.imread(img_file)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    return img

'''
This function is network inference with tensorflow library.
But it is a black box for education,
and I want to analysis the principle of quantization with inference.
If the filter/weight/bias/quantization could be exported in cunstom format with tables,
so we can deploy user network or basic network on other platforms,
not only Android/IOS/Raspberry,
but also stm32/FPGA and so on.
'''
def tflite_inference(model, img):
    # get input node information
    input_details = model.get_input_details()
    # get output node information
    output_details = model.get_output_details()
    # set input data
    model.set_tensor(input_details[0]['index'], img)
    # start inference
    model.invoke()
    # get output data
    output_data = model.get_tensor(output_details[0]['index'])
    return output_data

'''
This function refers the paper of "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference".
Thanks for the contribution of tensorflow.

In this function, I will implement the quantitative inference.
According to the network structure of mobilenet_v1,
this function supports depthwise/pointwise/standard convolution,
but the stride is only support 1 and 2.

The main principle is the following equation(1) in the reference paper:
output = Z3+(S1*S2/S3)[(input-Z1)x(weight-Z2)+bias]     (1)

"(input-Z1)x(weight-Z2)" is the operation of depthwise/pointwise/standard convolution,
Zi is the value of zero offset.
Generally, Z3=0, Z1=128(layer0) Z1=0(other layers), Z2>0.
If the activation function is relu6, S1=S3.
so the equation(1) can be written as:
output = (S1*S2/S3)[(input)x(weight-Z2)+bias]           (2)

If scale=(S1*S2/S3) or scale=S2, and the equation(1) can be simplified as:
output = scale*[(input)x(weight-Z2)+bias]               (3)

In equation(3), the data type of scale is float32, input is uin8(other layers) or int8(layer0), weight is uint8,
(weight-Z2) is int16.
'''
def my_conv(model, input, layer_index, layer_type='depthwise', strides=1):
    input_index = input_location[layer_index]
    weight_index = weight_location[layer_index]
    bias_index = bias_location[layer_index]
    output_index = output_location[layer_index]

    # input_quant[0]=>S1, input_quant[1]=>Z1
    input_quant = model._interpreter.TensorQuantization(int(input_index))
    # img_tensor = input-Z1
    img_tensor = input - tf.constant(input_quant[1], dtype=tf.float32)

    # weight_quant[0]=>S2, weight_quant[1]=>Z2
    weight_quant = model._interpreter.TensorQuantization(int(weight_index))
    t_w = model.get_tensor(int(weight_index))
    t_w = np.transpose(t_w, (1, 2, 3, 0))
    weight_tensor = tf.convert_to_tensor(t_w)
    weight_tensor = tf.cast(weight_tensor, dtype=tf.float32)
    # weight_tensor = weight-Z2
    weight_tensor = weight_tensor - tf.constant(weight_quant[1], dtype=tf.float32)
    # bias_tensor = bias
    bias_tensor = tf.convert_to_tensor(model.get_tensor(int(bias_index)), dtype=tf.float32)
    # output_quant[0]=>S3(S3=0), output_quant[1]=>Z3
    output_quant = model._interpreter.TensorQuantization(int(output_index))
    # scale=(S1*S2/S3) Note: If the activation function is relu6, then scale=S2.
    scale = input_quant[0] * weight_quant[0] / output_quant[0]

    if layer_type=='depthwise':
        conv_res = tf.nn.depthwise_conv2d(img_tensor, weight_tensor, strides=[1, strides, strides, 1], padding='SAME')
    elif layer_type=='pointwise':
        conv_res = tf.nn.conv2d(img_tensor, weight_tensor, strides=[1, 1, 1, 1], padding='SAME')
    elif layer_type=='standard':
        conv_res = tf.nn.conv2d(img_tensor, weight_tensor, strides=[1, strides, strides, 1], padding='SAME')
    else:
        print('layer_type = depthwise? pointwise? standard?')
    conv_bias = tf.nn.bias_add(conv_res, bias_tensor)
    conv_scale = conv_bias * tf.constant(scale, dtype=tf.float32)

    return tf.clip_by_value(tf.round(conv_scale), 0, 255)


'''
Classifier of MobileNet
'''
def my_fc(model, input, layer_index):
    input_index = input_location[layer_index]
    weight_index = weight_location[layer_index]
    bias_index = bias_location[layer_index]
    output_index = output_location[layer_index]

    weight_quant = model._interpreter.TensorQuantization(int(weight_index))
    t_w = model.get_tensor(int(weight_index))
    t_w = np.transpose(t_w, (1, 2, 3, 0))
    weight_tensor = tf.convert_to_tensor(t_w)
    weight_tensor = tf.cast(weight_tensor, dtype=tf.float32)
    weight_tensor = weight_tensor - tf.constant(weight_quant[1], dtype=tf.float32)

    return tf.matmul(input, weight_tensor)


model = load_tflite("mobilenet_v1_0.25_128_quant.tflite")
img = load_img('test2.png')

print('***********************TFLite inference**************************')
tf_res = tflite_inference(model, img)
tf_res = np.squeeze(tf_res)
print('TFLite result is', np.argmax(tf_res))

#%%
print('**********Custom inference for principle verification************')

layer_index=0

img_tensor = tf.convert_to_tensor(img)
img_tensor = tf.cast(img_tensor, dtype=tf.float32)
conv0 = my_conv(model, img_tensor, layer_index, layer_type='standard', strides=2)
layer_index = layer_index+1

conv1 = my_conv(model, conv0, layer_index, layer_type='depthwise', strides=1)
layer_index = layer_index+1
conv2 = my_conv(model, conv1, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv3 = my_conv(model, conv2, layer_index, layer_type='depthwise', strides=2)
layer_index = layer_index+1
conv4 = my_conv(model, conv3, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv5 = my_conv(model, conv4, layer_index, layer_type='depthwise', strides=1)
layer_index = layer_index+1
conv6 = my_conv(model, conv5, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv7 = my_conv(model, conv6, layer_index, layer_type='depthwise', strides=2)
layer_index = layer_index+1
conv8 = my_conv(model, conv7, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv9 = my_conv(model, conv8, layer_index, layer_type='depthwise', strides=1)
layer_index = layer_index+1
conv10 = my_conv(model, conv9, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv11 = my_conv(model, conv10, layer_index, layer_type='depthwise', strides=2)
layer_index = layer_index+1
conv12 = my_conv(model, conv11, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv13 = my_conv(model, conv12, layer_index, layer_type='depthwise', strides=1)
layer_index = layer_index+1
conv14 = my_conv(model, conv13, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv15 = my_conv(model, conv14, layer_index, layer_type='depthwise', strides=1)
layer_index = layer_index+1
conv16 = my_conv(model, conv15, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv17 = my_conv(model, conv16, layer_index, layer_type='depthwise', strides=1)
layer_index = layer_index+1
conv18 = my_conv(model, conv17, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv19 = my_conv(model, conv18, layer_index, layer_type='depthwise', strides=1)
layer_index = layer_index+1
conv20 = my_conv(model, conv19, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv21 = my_conv(model, conv20, layer_index, layer_type='depthwise', strides=1)
layer_index = layer_index+1
conv22 = my_conv(model, conv21, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv23 = my_conv(model, conv22, layer_index, layer_type='depthwise', strides=2)
layer_index = layer_index+1
conv24 = my_conv(model, conv23, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

conv25 = my_conv(model, conv24, layer_index, layer_type='depthwise', strides=1)
layer_index = layer_index+1
conv26 = my_conv(model, conv25, layer_index, layer_type='pointwise', strides=1)
layer_index = layer_index+1

pooling_res = tf.nn.avg_pool(conv26, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
pooling_res = tf.round(pooling_res)

fc_res = my_fc(model, pooling_res, layer_index)

with tf.compat.v1.Session() as sess:
    layer_res = sess.run(fc_res)
    print(layer_res.shape)
    print('Custom result is', np.argmax(layer_res))

# with tf.compat.v1.Session() as sess:
#     conv0_tensor = sess.run(conv5)
#     np.save("iter_5.npy", conv0_tensor, allow_pickle=False)
#     print('conv0=',conv0)
#     layer_res = sess.run(fc_res)
#     print(layer_res.shape)
#     print('Custom result is', np.argmax(layer_res))
    
# with tf.compat.v1.Session() as sess:
#     for i in range(27):
#         temp=sess.run(conv0)
#     conv0_tensor = 
#     np.save("iter_0.npy", conv0_tensor, allow_pickle=False)
#     print('conv0=',conv0)
#     layer_res = sess.run(fc_res)
#     print(layer_res.shape)
#     print('Custom result is', np.argmax(layer_res))    

#
#Check it worked
# print(np.load("iter_0.npy")) #[1.  1.5 2. ]


