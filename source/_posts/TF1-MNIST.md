---
title: TF1-MNIST
date: 2018-04-01 18:24:02
tags: tensorflow
mathjax: true
---
MNIST是一个计算机视觉数据集，包含各种手写数字图片，也包含每一张图片的标签（即，该数字是几）。
<!-- more -->
像这组图，对应标签是5,0,4,1.
![](/images/TF1/TF1_1.png)
### 目的
通过训练一个模型用于预测图片里的数字，从而了解tensorflow的工作流程和ML的概念。

### 使用模型
softmax regression

### MNIST数据集

 - mnist.train:60000行训练数据集
 - mnist.test:10000行测试数据集（用于评估模型性能，易泛化到其他数据集）
 - mnist.train.images：训练用的图片，设为xs
 每张图片有28*28个像素点，用长度为28*28=784的向量表示，如下图（实际上，这样的表示会丢失图片的二维结构信息，但这里的softmax回归不会用到结构信息）
![](/images/TF1/TF1_2.png)
    xs是一个形状为[60000,784]（图片索引，像素点索引）的张量，存的值表示图片中像素的强度，取值介于0和1之间

![](/images/TF1/TF1_3.png)

 - mnist.train.labels:训练用标签，设为ys。标签是介于0到9的数字。
 表示为one-hot vectors（one-hot向量除了某一维数字是1，其余都是0）.数字n将表示为第n维为1，其余维度为0的10维向量。labels是一个[60000,10]的数字矩阵.

### 构建模型
softmax回归模型：该模型分两步，可以给不同的对象分配概率。

#### step 1
 对图片像素值进行加权求和，以求得该图片属于某个数字类的证据（evidence）。如果证据充分，则权值为正数，反之为负数。


So，对于给定的输入图片x，代表的是数字i的证据可以表示为：
$$
evidence_i=\sum_j W_{i,j}x_j+b_i
$$
$W_i$是权重，$b_i$是数字$i$类的偏置量，$j$是给定图片$x$的像素索引用于像素求和。
#### step 2
再用softmax函数把证据转换为概率y：
$$
y=softmax(evidence)
$$
softmax函数可以将图片对应每个数字的匹配度转换为概率值
$$
softmax（x）= normalize(exp(x))
$$
$$
softmax(x)_i=\frac{exp(x_i)}{\sum_j exp(x_j)}
$$
####模型 过程表示图
![](/images/TF1/TF1_4.png)
用等式表示为：
![](/images/TF1/TF1_5.png)
进一步表示为：
$$
y=softmax(Wx+b)
$$
### 实现回归模型

```python
import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 通过操作符合标量来描述可交互的操作单元
# x是一个占位符，当运行计算时，才输入
# 输入任意数量的mnist图像，每一张图展开为784维向量，用2维浮点数张量来表示，shape是[None,784]
x=tf.placeholder(tf.float32,[None,784]) 

W=tf.Variable(tf.zeros([784,10]))#Variable表示可修改的张量
b=tf.Variable(tf.zeros([10]))

#定义模型
y=tf.nn.softmax(tf.matmul(x,W)+b) 

#y_用于输入正确值
y_=tf.placeholder("float",[None,10])

# 用于评估模型好坏的cost函数 cross_entropy，
# 是100张图片的交叉熵总和
# 100个数据点的预测表现比单一数据点的表现能更好描述模型性能
cross_entropy=-tf.reduce_sum(y_*tf.log(y)) 

#用梯度下降法以0.01的学习速率最小化交叉熵
#给描述的计算的那张图里面增加一系列新的计算操作单元用于实现反向传播算法和梯度下降算法
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初始化创建的变量
init=tf.initialize_all_variables()

#在一个Session里面启动模型，并初始化变量
sess=tf.Session()
sess.run(init)

#开始训练模型，让模型循环训练1000次
#该循环的每个步骤中，随机抓取训练数据中的100个批处理数据点
# 然后用这些数据点作为参数替换之前的占位符来运行train_step
#使用一小部分的随机数据来进行训练被称为随机训练（stochastic training）
#这里更确切的说是随机梯度下降训练
for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys})

#评估模型性能
#用 tf.equal 来检测预测是否真实标签匹配(索引位置一样表示匹配)
#correct_prediction 是一组布尔值
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))#argmax 会返回最大值所在的索引

#把布尔值转换成浮点数，然后取平均值
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

#计算所学习到的模型在测试数据集上面的正确率
print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

#最终结果值应该大约是91%
```
#### 参考资料
tensorflow官方文档中文版

