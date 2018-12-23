---
title: Self-supervised Learning
mathjax: true
date: 2018-12-16 20:39:45
tags: 
---

### Motivation

a significant limitation of supervised machine learning, viz. requiring lots of external training samples or supervisory data consisting of inputs and corresponding outputs.

<!-- more -->

### 提出者：Yann LeCun

### 思想

和word2vec的算法思想很相似，通过周围的词来预测一个词的语义context（这是word2vec？记得有两种），自监督也采用这种思想，automatically identifying, extracting and using supervisory signals.

### 什么是self-supervised learning？

- autonomous supervised learning

- 是一种表示学习方法，不需要预先已经标注好的数据，只需要上下文和embedded metadata 作为监督信号。

  就图像识别来说，可以通过图像上的相对位置信息作为监督信号，训练得到丰富的视觉表示，识别出该图像的内容是什么。不仅使用于计算机视觉，还有其他领域。

#### Self-supervised vs. supervised learning

​	自监督是监督学习，因为它们的目标都是从数据对（输入和有标签的输出）学得一个函数。但自监督并不是像监督学习那样需要明显的带标签的输入输出数据对，而是把correlations, embedded metadata, or domain knowledge available （输入中隐含的或者从数据中自动抽取）作为监督信号。自监督学习也已经能用于回归和分类啦。

#### Self-supervised vs. unsupervised learning

​	自监督类似无监督学习，因为都是从没有明确标签的数据中进行学习。无监督学习是学习数据的内在关系、结构，着重于clustering、grouping、dimensionality reduction, recommendation engines, density estimation, or anomaly detection，这些都与自监督不一样。

#### Self-Supervised vs. semi-supervised learning

​	半监督学习是使用小部分有标签数据、大部分无标签数据进行学习；但自监督学习使用的数据都是没有明确提供标签的数据。

​	针对监督学习的缺点，学习方法和scalability，即需要大量的有标签数据、数据清洗、为某些特定问题专门训练一个模型，这与人类的学习方式不一样。人们花了多年时间在数据收集和专业标注上，比如：tens of millions of labeled bounding boxes or polygons and image level annotations, but these datasets [Open Images](https://storage.googleapis.com/openimages/web/index.html), [PASCAL Visual Object Classes](http://host.robots.ox.ac.uk/pascal/VOC/index.html), [Image Net](http://www.image-net.org/), and Microsoft [COCO](http://cocodataset.org/) collectively pale in comparison to billions of images generated on a daily basis on social media, or millions of videos requiring object detection or depth perception in autonomous driving. 而人类在学习的时候，需要少量数据、多源的，可以针对多个任务，能很好的泛化。

### 参考文献

https://hackernoon.com/self-supervised-learning-gets-us-closer-to-autonomous-learning-be77e6c86b5a



