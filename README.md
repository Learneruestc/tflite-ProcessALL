# tflite-ProcessALL
train a model, and quantization using tensorflowLite, and Inference
本文档介绍了TensorflowLite量化的过程以及量化后推理的过程。
首发于知乎：https://zhuanlan.zhihu.com/p/557438837

....代码文件清单
1） tensorflow训练手写数字体模型以及tensorflowLite量化过程代码：quant-example.ipynb

2）mobilenet_tflite_fp32.py： fp32精度的模型进行推理

3） mobilenet_tflite_quant.py: uint8量化后的模型进行推理

4）mobilenet_tf.py: uint8量化后的模型进行推理以及自己一层一层写出来的网络推理

5）cosine_similarity2.py 计算两个tensor余弦相似度的，一张图片number就是1，只有iter_0存在，2张测试图片number就是2，有iter_0，iter_1存在；（已删）
