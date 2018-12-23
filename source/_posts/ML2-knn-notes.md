---
title: ML2-knn notes
date: 2018-04-01 17:17:44
tags: machine learning
mathjax: true
---
### 概述
&emsp;&emsp;k近邻法（k-nearest neighbor,k-NN）是一种基本分类与回归方法。
&emsp;&emsp;给定一个训练数据集，对新的输入实例，在训练数据集中找到与该实例最邻近的k个实例，这k个实例的多数属于某个类，就把该输入实例分为这个类。
<!-- more -->

### 算法
 输入：训练数据集
$$
T=\{(x_1,y_1),(x_2,y_2),..,(x_N,y_N)\}
$$
 其中，$x_i \in X \subseteq R^n$为实例的特征向量，$y_i \in Y=\{c_1,c_2,...,c_k\}$为实例的类别，$i=1,2,...,N$；实例特征向量x;
 输出：实例$x$所属的类$y$
 （1）根据给定的距离度量，在训练集T中寻找出距离x最近的k个点，涵盖这k的点的领域记作$N_k(x)$；
 （2）在$N_k(x)$中根据分类决策规则（eg:多数表决），确定x所属的类别y：
$$
y=arg \max_{c_j} \sum_{x_i \in N_k(x)} I(y_i=c_i), i=1,2,...,K
$$
 其中，$I$为指示函数，即当$y_i=c_i$时$I$为1，否则$I$为0$。
 特殊情况是$k=1$的情形，称为最近邻算法。


#### k近邻模型
 &emsp;&emsp;k近邻法的模型对应于特征空间的划分。模型由三个基本要素--距离度量、k值的选择和分类决策规则决定。
#### 模型
&emsp;&emsp;当上述三个要素确定后，对于任何一个新的输入实例，所属的类唯一地确定。
&emsp;&emsp;特征空间中，对每个训练实例点$x_i$，距离该点比其他店更近的所有点组成的一个区域，叫做单元（cell）。每个训练实例点拥有一个单元，所有训练实例点的单元构成对特征空间的一个划分。最近邻法将实例$x_i$的类$y_i$作为其单元中所有点的类标记（class label）。则每个单元的实例点的类别是确定的。
#### 距离度量
&emsp;&emsp;特征空间中两个实例点的距离是这两个实例点相似度的反映。k近邻模型的特征空间一般是n维实数向量空间$R^n$。使用的距离是欧式距离，也可以是其他距离，比如更一般的$L_P$距离（$L_P$ distance）或Minkowski距离。
&emsp;&emsp;设特征空间X是n维实数向量空间$R^n$，$x_i,x_j \in X,
x_i=(x_i^{(1)},x_i^{(2)},...,x_i^{(n)})^T$，
$x_j=(x_j^{(1)},x_j^{(2)},...,x_j^{(n)})^T$，$x_i,x_j的L_P距离定义为：$
$$
L_P(x_i,x_j)=\lgroup \sum_{l=1}^n |x_i^{(l)}-x_j^{(l)}|^p\rgroup ^{\frac{1}{p}}
$$
 当$p \ge 1。当p=2$时，称为Euclidean distance，即
$$
L_2(x_i,x_j)=\lgroup \sum_{l=1}^n |x_i^{(l)}-x_j^{(l)}|^2\rgroup ^{\frac{1}{2}}
$$
 当$p =1$时，称为Manhattan distance，即
$$
L_1(x_i,x_j)= \sum_{l=1}^n |x_i^{(l)}-x_j^{(l)}|
$$
 当$p=\infty$时，是各个坐标距离的最大值，即
$$
L_\infty(x_i,x_j)= \max_{l}|x_i^{(l)}-x_j^{(l)}|
$$


#### k值的选择
&emsp;&emsp;k值的选择会对k近邻法的结果产生重大影响。
&emsp;&emsp;如果选择较小的k值，相当于用较小的邻域中的训练实例进行预测，学习的近似误差（approximation error）会减小，只有与输入实例较近的（相似的）训练实例才会对预测结果起作用。但缺点是学习的估计误差（estimation error）会增大，预测结果会对近邻的实例点非常敏感。k值的减小意味着整体模型变得负责，容易发生过拟合。
&emsp;&emsp;如果k值较大，相当于用较大邻域里的训练实例进行预测。优点是可以减少学习的估计误差，但缺点就是会增大近似误差。这是与输入实例较远的（不太相似）的训练实例也会对预测起作用，是预测发生错误。k值的增大意味着模型变得更简单。
&emsp;&emsp;如果k=N，则将输入实例预测为训练实例中最多的类。即模型过于简单，完全忽略了训练实例中的大量有用信息，不可取。
&emsp;&emsp;实际应用中，k值一般去一个比较小的数值。通常常采用交叉验证法（将原始数据(dataset)进行分组,一部分做为训练集(train set),另一部分做为验证集(validation set or test set),首先用训练集对分类器进行训练,再利用验证集来测试训练得到的模型(model),以此来做为评价分类器的性能指标）来选取最优值。
#### 分类决策规则
&emsp;&emsp;多用是**多数表决**，即由输入实例的k个近邻的训练实例中的多数类决定输入实例的类。多数表决的规则等价于经验风险最小化。
#### k近邻法的实现：kd树
&emsp;&emsp;实现的过程中，主要的问题是如何对训练数据进行快速k近邻搜索。这在特征空间的维数大，及训练数据容量大时尤其必要。
&emsp;&emsp;最简单的方法是线性扫描（linear scan）。需要计算输入实例与每一个训练实例的距离。当训练集很大时，计算非常耗时，不可取。
&emsp;&emsp;为了改善，可以使用特殊的结构存储训练数据，比如kd树（kd tree）。
##### 构造kd树
**例：**给定一个二维空间的数据集：
$$
T=\{(2,3)^T,(5,4)^T,(9,6)^T,(4,7)^T,(8,1)^T,(7,2)^T\}
$$
构造一个平衡kd树。
**解：**

 - 根节点对应包含数据集T的矩形，选择$x^{(1)}$轴，6个数据点的$x^{(1)}$坐标中位数是7，以平面$x^{(1)}=7$将空分为左右两个子矩形（子节点）；
 - 左矩形以$x^{(2)}=4$分为两个子矩形，右矩形以$x^{(2)}=6$分为两个子矩形；
   如此递归，最后得到如下图所示的特征空间划分和kd树。

![](/images/features_zone_classfy.png)
![](/images/kd_tree.png)

##### 搜索kd树
输入：已构造的kd树，目标点x；
输出：x的最近邻

 - （1）在kd树中找到包含目标的x的叶节点：从根节点出发，递归地向下访问kd树。若目标点x当前维的坐标小于切分点的坐标，则移动到左子节点，否则移动到右子节点。直到子节点为叶节点为止。
 - （2）以此叶节点为“当前最近点”
 - （3）递归地向上回退，在每个节点进行下列操作：
 - （a）如果该节点保存的实例点比当前最近点距离目标更近，则以该实例点为“当前最近点”；
 - （b）当前最近点一定存在于该节点一个子节点对应的区域。检查该子节点的父节点的另一个子节点对应的区域是否有更近的点。具体来说，是检查另一子节点对应的区域是否以目标点为球心，以目标点与“当前最近点”间的距离为半径的球体相交。
	 &emsp;如果相交，可能在另一个子节点对应的区域内存在距目标点更近的点，移动到另一个子节点。接着，递归的进行最近邻搜索；
	  &emsp;如果不相交，向上回退。
 - （4）当回退到根节点是，搜索结束。最后的“当前最近点”即x的最近邻点。

&emsp;&emsp;kd树更适用于训练实例远大于空间维数时的k近邻搜索。当空间维数接近训练实例数时，效率会迅速下降，几乎接近线性扫描。

### python实现
```python
import numpy as np
import operator

def Dataset():
    np.random.seed(13)
    dataList=np.random.randint(1,10,8)
    print('dataList',dataList)
    data=np.array(dataList).reshape(4,2)
    print('data',data)
    lables=['A','B','A','B']
    return data,lables
def classfy(target,dataset,labels,k):
    dataSize=dataset.shape[0]

    #compute Euclidean distance=sqrt(sum of all the difference between tartget and dataSet)
    minus=np.tile(target,(dataSize,1))-dataset
    temp=minus**2
    temp1=temp.sum(axis=1) # sum of each row
    distance=temp1**0.5

    sortedDistIdx=distance.argsort()# return the indcies of sorted ele,emts
    count={}
    #count labels
    for i in range(k):
        theLabel=labels[sortedDistIdx[i]]
        print('label={},i={}'.format(theLabel,i))
        count[theLabel]=count.get(theLabel,0)+1

    sortedCount=sorted(count.items(),key=operator.itemgetter(1),reverse=True)
    return sortedCount[0][0]

data,label=Dataset()
target=[3,2]

className=classfy(target,data,label,3)
print('target is class:',className)

```

