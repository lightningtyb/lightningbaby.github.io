---
title: SSAH_reading notes
date: 2018-12-16 20:16:34
tags: cross-modal
mathjax: true
---
有待进一步了解：semantic network，self-supervised learning，公共语义空间,multi-label annotations,跨媒体哈希

# 背景

​	跨媒体检索主要有两类办法：1.同时学习common和modality-specific特征，将不同模态用一个共享层，对他们的关系进行建模。2.two-stage framework：先对各模态提取特征，然后构建低纬度的公共表示空间。（Cross-modal Retrieval with Correspondence Autoencoder）
<!-- more -->
​	高纬的modality-specific 特征有助于bridge 模态gap，因此如何encourage 更多丰富的语义关系，构建更准确的模态关系，这对于在实际的应用中，达到令人满意的性能变得十分重要。（？？？之前在上一篇论文中提到，对于使用方法1来说，有利于检索的是公共语义信息，不太有利的是modality-specific特征，因为模型会把后者也进行学习，这对于模型来说是harmful！！！哦，因为本篇论文的思想是单独提取出各模态特征，采用的是方法二）

问题：如何提升跨模态检索的准确率，以及规避/减小模态鸿沟

该论文提出：利用自监督对抗哈希SSAH解决跨模态检索问题

贡献：用双对抗网络，最大化语义关系、不同模态的表示一致性

​	   用自监督语义网络，用**多标签annotations**的方式，挖掘高纬语义信息

问题：跨模态检索

解决方法：根据各种模态生成内容标签，然后检索共同或者相近的内容标签，从而进行匹配。通常做法是，先提取每个模态的特征，根据相似度建立索引（？？？），再检索。

问题：如何加快检索速度，节省存储空间

解决方法：特征尽量短、二进制表示，用哈希编码实现。根据多模态内容生成哈希码，希望不同模态的同个对象的哈希码尽可能相近，不同对象的哈希码尽量不同。同时，又由于跨模态的相近对象存在语义相关性，通常的做法是，将不同模态的内容映射到公共语义空间。大多数用shallow 跨模态哈希方法，包括无监督方法、有监督方法。当然有监督的方法更能提纯跨模态关系，效果会更好。但这些方法都是基于hand-crafted features，会限制**instance的判别表示**，从而降低学得的二进制哈希码的准确率。

缺陷：

+ 对于模态间的语义相关性采用单一的label进行标注，但对于一些数据集来说，一个数据对象可以是有多个类别标签的。这样有多个类别标签的数据对象能更准确的描述出语义关系。

+ 用预先定义的loss函数，构建对应的哈希码，会narrow 模态鸿沟（？？难道不是更好吗）。而去哈希码的长度一般小于128位，就会丢失很多有用的信息，无法捕捉到模态的内在统一性（说的这么绝对的咩）

# Self-Supervised Adversarial Hashing Networks for Cross-Modal Retrieval

<img src="https://ws1.sinaimg.cn/large/006tNbRwly1fy8usmfcwpj31ig0mu101.jpg" style="zoom:35%">

**SSAH**

+ 端到端的自监督语义网络LabNet，用**多标签annotations**的方式，挖掘高纬语义信息

+ 用双对抗网络ImgNet,TxtNet，最大化语义关系、不同模态的表示一致性

## 实现思路

+ Phase1:ImgNet和TxtNet各自生成**modality-specific特征**（这些语义信息包含在它们各自的输出层里）到公共语义空间，用于找寻模态间的语义相关性。LabNet从multi-label notation里学习**语义特征（semantic features）**，可以看作是公共语义空间，用来监督模态特征学习。

+ Phase2:把上述得到的modality-specific特征和semantic特征不断的输入到两个判别器中，在相同的语义特征的监督下，两个模态的特征分布逐渐趋于一致。

## 实现细节

### 总体目标

+ ImgNet,TxtNet各自学习单独的哈希函数，$H^{v,t}=f^{v,t}(v,t;\theta ^{v,t})$;

+ 针对两个模态学习到统一的哈希码，$B^{v,t}\in \{-1,1\}^K,K$是二进制码的长度.

### 符号定义

+ $O=\{o_i\}^n_{i=1}$是数据集，包含$n$个instance

  $o_i=(v_i,t_i,l_i),v_i \in R^{1\times d_{v}}$是图像特征,$t_i \in R^{1\times d_{t}}$是文本特征

  $l_i=[l_{i1,...,l_{ic}}]$是multi-label annotations，c是类别数，一个instance可能会有很多个label;
$$
l_{i,j}=\begin{cases}
  1 & \text{如果} o_i \text{属于第}j\text{类} \\
  0\ & \text{其他} \\
  \end{cases}
$$

+ $dis_H(b_i,b_j)=\frac {1}{2}(K-<b_i,b_j>)$用来衡量两个二进制码的相似性；

+ $S$是pairwise multi-label 的相似性矩阵，用来描述两个instances的语义相似性，

$$
S_{i,j}=\begin{cases}
  1 & \text{如果}o_i \text{和}o_j \text{语义相似,}\text{即至少共享一个label} \\
  0\ & \text{其他} \\
  \end{cases}
$$

+ $$P(S_{i,j}|B)=\begin{cases}
  \delta(\Psi{_{ij}}) \quad S_{ij}=1 \\
  1-\delta(\Psi{_{ij}})\quad S_{ij}=0 \\
  \end{cases}$$

  $\delta(\Psi{_{ij}}) =\frac{1}{1+e^{-\Psi{_{ij}}}},\Psi{_{ij}}=\frac{1}{2}<b_i,b_j>$ ，可以看出两个instance内积越大，$P(S_{i,j}|B)$也越大，它们就越相似。


### **LabNet (自监督语义生成)**

<img src="https://ws3.sinaimg.cn/large/006tNbRwly1fy8usrkjfzj31ps08edic.jpg" style="zoom:35%">

+ 作用：在公共语义空间，对模态间的语义关系进行建模，将语义特征映射到哈希码。目标是学习这个映射函数。

+ 输入：multi-label annotation

+ 输出：$H^l=f^l (l;\theta^l)$，其中$f^{v,t, l}$是哈希函数，$\theta ^{v,t,l }$是要学的参数

+ 网络结构

  四层前馈神经网络，$L \rightarrow 4096\rightarrow 512 \rightarrow N ,N=K+c$

+ 目标函数
  $$
  \begin{aligned}
  min_{B^l,\theta ^l,\hat L_l}=& \alpha J_1+\gamma J_2+\eta J_3 +\beta J4  \\ 
  =& -\alpha \sum^n_{i,j=1}(S_{i,j}\Delta^l_{i,j}-log(1+e^{\Delta^l_{i,j}}))\\
  =& -\gamma \sum^n_{i,j=1}(S_{i,j}\Gamma ^l_{i,j}-log(1+e^{\Gamma^l_{i,j}}))\\
  =& -\eta ||H^l-B^l||^2_F+\beta||\hat L^l-L||^2_F\\
  & s.t. B^l \in \{-1,1\}^K\\
  \end{aligned}
  $$

   其中$\alpha,\gamma,\beta,\eta$是要学习的参数，$\Delta^l_{i,j}=\frac {1}{2}(F^l_{*i})^T(F^l_{*j})$，$\Gamma ^l_{i,j}=\frac {1}{2}(H^l_{*i})^T(H^l_{*j})$,$F^{v,t,l} \in R^{s \times n}$表示公共语义空间里图像、文本、标签的语义特征，$S$是语义空间维度，是深度神经网络的输出层。

+ $B^{v,t,l}=sign(H^{v,t,l})\in \{-1,1\}^K$,三个网络学到了各自的哈希函数$H^{v,t,l}$，得到哈希码，通过$sign$函数得到二进制码，得到更加轻量级的特征。

+   $J_1$保证语义特征相似性,$J_2$保证有相似label的instance要有相近的哈希码，  $J_3$是学习得到的哈希码二进制化的近似loss，  $J_4$是原label和预测的label的分类loss。

### **特征学习**

​	Loss函数和$(1)$中等式一样，但$F^l_{*i},F^l_{*j}$是用的LabNet生成的公共语义空间生成的，这可以确定相关性，从而避开模态鸿沟？这也是自监督学习起到的作用。

#### **TxtNet**

<img src="https://ws1.sinaimg.cn/large/006tNbRwly1fy8usvfukqj31du07476f.jpg" style="zoom:50%">

+ 目标：学习能将文本特征映射到哈希码的函数

+ 生成网络：

  + 输入：词向量，BoW模型，但是这样的表示太稀疏，不适合生成哈希码，因此采用了多尺度混合模型来解决这个问题。

  + 多尺度混合模型$\rightarrow 4096\rightarrow 512 \rightarrow N$

  + 多尺度混合模型：5层平均池化（池化层大小：$1*1,2*2,3*3,5*5,10*10$）和$1*1$的卷积层，前者用来提取多尺度特征，后者用来融合提取到的多尺度特征。

+ 对抗网络结构：3层前馈网络（$F^{v,t,l} \rightarrow 4096 \rightarrow 4096 \rightarrow 1$）

#### **ImgNet**

<img src="https://ws4.sinaimg.cn/large/006tNbRwly1fy8utcicvtj31e007076g.jpg" style="zoom:50%">



+ 目标：学习能将文本特征映射到哈希码的函数

+ 生成网络有两种，一种是采用CNN-F的前7层提取图像特征+fc8+输出层（N个节点），第二种是CNN-F用vgg19代替，其他部分不变。

### **对抗学习**

<img src="https://ws4.sinaimg.cn/large/006tNbRwly1fy8utfi4s9j30ma0iw0v8.jpg" style="zoom:50%">

+ 输入：modality features和LabNet产生的语义特征

+ 输出：0/1（判断标签来自ImgNet或TxtNet时输出0，来自LabNet输出为1）

+ 在公共空间分配给语义特征的模态标签表示为$Y=	\{y_i\}^{3\times n}_{i=1},y_i\in\{0,1\}$
  + 模态标签：$Y^l=\{y_i^l\}^{n}_{i=1},y_i^l=1$
  + 图像和文本的模态标签：$Y^{v,t}=\{y_i^{v,t}\}^{n}_{i=1},y_i^{v,t}=0$

+ 目标函数：$min_{\theta ^{*,l}_{adv}}L^{*,l}_{adv}=\sum ^{2\times n}_{i=1}||D^{*,l}(x^{*,l}_{i})-y^{*,l}_{i}||^2_2,*=v,t$，$x^{*,l}_{i}$是公共空间的语义特征，$y^{*,l}_{i}$是判别值，相当于一个二元分类器

### **训练过程的伪代码** 

<img src="https://ws1.sinaimg.cn/large/006tNbRwly1fy8utjtnfyj30zk0n6wii.jpg" style="zoom:50%">

# 实验

### 数据集

<img src="https://ws4.sinaimg.cn/large/006tNbRwly1fy8utn23duj30lo050gmn.jpg" style="zoom:50%">

### 评估

MAP、PR、Precision@top k

### Baseline
5个基于浅层结构方法（CVH , STMH , CMSSH , SCM ,SePH ），一个基于深层结构的方法（DCMH）

### 超参数的确定
从数据库随机选择2000个数据点作为验证集，最终设置$\alpha=\gamma=1,\beta=\eta=10^{-4}$

<img src="https://ws4.sinaimg.cn/large/006tNbRwly1fy8utqt4loj317207kmyt.jpg" style="zoom:50%">

### Hamming Ranking

SSAH分别采用CNN-F做实验，与6个baseline进行对比，SSAH的效果都更好：

+ 与5个基于浅层结构方法的baseline相比，在MIRFLICKR-25K数据集上，SSAH的MAP值能高10%；

+ 与1个基于深度结构方法的baseline相比，在MIRFLICKR-25K数据集上，SSAH的MAP值能高5%。

<img src="https://ws1.sinaimg.cn/large/006tNbRwly1fy8uttl8bhj314e0hktdt.jpg" style="zoom:50%">

SSAH分别采用VGG19做实验，与6个baseline进行对比，SSAH的效果都更好，且几乎所有的效果比用CNN-F更好；与5个基于浅层结构方法的baseline相比，在MIRFLICKR-25K数据集上，SSAH的MAP值能高5%；

<img src="https://ws1.sinaimg.cn/large/006tNbRwly1fy8utwg4nmj31480hkgqn.jpg" style="zoom:50%">

### Hash Lookup

SSAH使用CNN-F作为ImgNet的生成网络；

Hamming距离小于某个值认为是正样本，这个值称为Hamming Radius，改变Radius可以改变Precision-Recall的值，于是可以得到P-R曲线，P-R曲线与坐标轴围成的面积越大，说明效果越好。

PR curve结果图中，SSAH比其他几个baseline的效果都好。

<img src="https://ws1.sinaimg.cn/large/006tNbRwly1fy8utz553zj31760q6wxg.jpg" style="zoom:50%">

### Ablation Study

SSAH-1：不使用自监督学习；SSAH-2：TxtNet替换为全连接网络；SSAH-3：去掉对抗学习

<img src="https://ws3.sinaimg.cn/large/006tNbRwly1fy8uu2e5zej30ky09qabs.jpg" style="zoom:50%">

数据集是 MIRFLICKR-25K，哈希码长度为16bits;

可以看出SSAH-1的曲线的值都是最低，即自监督语义网络能够显著提升检索性能；

同样SSAH-2，SSAH-3曲线的值都低于SSAH，足以证明TxtNet和对抗学习在这其中起到的作用。

### Training Efficiency

<img src="https://ws2.sinaimg.cn/large/006tNbRwly1fy8uu74rr5j30ks08c3zo.jpg" style="zoom:50%">

​	SSAH比基于深度结构方法的DCMH能更快收敛，且MAP值也更高，说明SSAH能从高纬语义特征和哈希码中学到更多监督信息，训练的ImgNet,TxtNet对于找到模态间的关系更有效。

### 和ACMR对比

+ ACMR是目前跨模态检索效果最好的方法，也第一个使用对抗学习，但它不是基于哈希的方法；

+ 在NUS-WIDE-10k 数据集的10个最大的类里，随机选取10,000 image/text pairs；

<img src="https://ws1.sinaimg.cn/large/006tNbRwly1fy8uuaoafcj30hi048mxi.jpg" style="zoom:50%">

原因分析：可能是因为SSAH采用了两个对抗网络，能学好模态间的特征分布，捕捉模态关系。



# 总结

- 为了提升跨模态检索的效率，该论文提出的SSAH，融合了多标签的自监督语义网络和对抗学习，使得不同模态间语义相关性最大化和特征分布一致性。

