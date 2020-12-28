## **一.cnn的基本结构**

**卷积神经网络有输入层、卷积层、池化层、全连接层、输出层。（**计算机理解这些层均为矩阵。）

**①卷积层****(Convolution Layer)**的激活函数使用的是ReLU，即ReLU(x)=max(0,x);

②在卷积层后面是**池化层****(Pooling layer)**，需要注意的是，池化层没有激活函数

③在若干卷积层+池化层后面是**全连接层（****Fully Connected Layer,** **简称****FC****）**，全连接层其实就是我们前面讲的DNN结构，只是输出层使用了Softmax激活函数来做图像识别的分类。

## **二.卷积的知识**

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyibZ6wzK04Esaa69GQVHKd1NaiaWlANBgrvwedpULRiaS0jUica6zia6MKqw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image001.png) |


微积分中卷积的表达式为：

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyK6m6hg2o0XoiawM2kJCve9u5cWeX52EKmd84aiaCPQj1T2vDiboUQS7iag/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image002.png) |


离散形式为：

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyYIJcowkzHQu6tn3nC0myMzeFr5SnntbxsUyIbmicu5BDV0P0E7y30NQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image003.png) |


矩阵形式为：

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyqGH6RngIo1yrjyMOEfiaIzGb8JXu4icCbRRYyOQarbTjaqxHN7yXSZBQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image004.png) |


如果是二维的卷积，则表示式为：

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyCCv5rBOJFYPUg7q5ZIFODHibVAI9RWU70Ym3sf7sNWEcngsKIqibtS0w/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg) |


在CNN中，虽然我们也是说卷积，但是我们的卷积公式和严格意义数学中的定义稍有不同,比如对于二维的卷积，定义为：

​      后面讲的CNN的卷积都是指的上面的最后一个式子。其中，我们叫W为我们的**卷积核**，而X则为我们的**输入**。**如果X是一个二维输入的矩阵，而W也是一个二维的矩阵。但是如果X是多维张量，那么W也是一个多维的张量。**

## **三.cnn之前向传播**

定义的CNN模型参数

1）一般我们的卷积核不止一个，比如有K个，那么我们输入层的输出，或者说第二层卷积层的对应的输入就K个。

2）卷积核中每个子矩阵的的大小，一般都用子矩阵为方阵的卷积核，比如FxF的子矩阵。

3）填充padding（以下简称P），我们卷积的时候，为了可以更好的识别边缘，一般都会在输入矩阵在周围加上若干圈的0再进行卷积，加多少圈则P为多少。（预处理）

4）步幅stride（以下简称S），即在卷积过程中每次移动的像素距离大小。

（1）CNN隐层到卷积层

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHClCkSR0cbqFK0xrglXwP4GvUFoHWOKVaCLIibGQMR4bP7ckuxCZiaIplqXsMc5ravxweC7F7Mf7oEA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image007.png) |


 卷积是有效提取图像特征的方法。这时表达式和输入层的类似，即 

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHClCkSR0cbqFK0xrglXwP4GDYMgcv3J2wJj4ia95hqRNLKDQiaf0lvMCmSEib46icBFuIB9tKrfdHb8XQ/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image009.jpg) |


其中，上标代表层数，星号代表卷积，而b代表我们的偏倚,σ为激活函数，一般是ReLU。也可以写成M个子矩阵子矩阵卷积后对应位置相加的形式，即：

其中输出矩阵边长=(输入边长-卷积核长+1)/步长。

动态展示图如下图所示。

（2）CNN隐层到池化层

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8ly6ZIrMgAFS5xNspgAmibfGUQR8mkDbpghKYCialKdmOZAiaZ7JxHibMG4cw/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image010.png) |



池化能够减少特征的数量。最大池化可提取图片的纹理，均值池化可保留背景特征。比如输入的若干矩阵是N×N维的，而我们的池化大小是k × k的区域，则输出的矩阵都是N/k × N/k维的。

这里需要需要我们**定义的CNN****模型参数是：

1）池化区域的大小k

2）池化的标准，一般是MAX或者Average。

如下图所示，为最大池化过程。

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyeFlCzv6SsQgBwK6VvqysIfcHa6AiaWnM5EFpicVKtdUlYib97KPEeokUA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg) |


**(3)CNN****隐层到全连接层

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHClCkSR0cbqFK0xrglXwP4G6u9HNeP6J6EnZWXpf2cygjl0gib8l4mupwtxdPpoDibKoMZr1oe0VQ0Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image013.png) |


由于全连接层就是普通的DNN模型结构，因此我们可以直接使用DNN的前向传播算法逻辑，即：

这里的激活函数一般是sigmoid或者tanh。经过了若干全连接层之后，最后的一层为**Softmax输出层**。此时输出层和普通的全连接层唯一的区别是，激活函数是softmax函数。

这里需要定义的CNN模型参数是：

1）全连接层的激活函数

2）全连接层各层神经元的个数

**前向传播算法流程**

**输入：**1个图片样本，CNN模型的层数L和所有隐藏层的类型，对于卷积层，要定义卷积核的大小K，卷积核子矩阵的维度F，填充大小P，步幅S。对于池化层，要定义池化区域大小k和池化标准（MAX或Average），对于全连接层，要定义全连接层的激活函数（输出层除外）和各层的神经元个数。

**输出：**CNN模型的输出a^L

**1)**  根据输入层的填充大小P，填充原始图片的边缘，得到输入张量a^1。

**2****）**初始化所有隐藏层的参数W,b　　

**3****）**for l=2 to L−1:

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHClCkSR0cbqFK0xrglXwP4GRP1ovAQmwiaIgJeoMG0p3ZOEUjBZ4qsWsIxhETODEOIVGgoDc0VQInA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image014.png) |

**a)**  如果第l层是卷积层,则输出为

**b)**  如果第l层是池化层,则输出为al=pool(al−1), 这里的pool指按照池化区域大小k和池化标准将输入张量缩小的过程。

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHClCkSR0cbqFK0xrglXwP4Gpl1duR5rcFTqDnfN5q0PrqdUP1DejRPsPwicrHK7mic3M2SfiaHpy64qg/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image015.png) |


**c)**  如果第l层是全连接层,则输出为

|      |                                                              |
| ---- | ------------------------------------------------------------ |
|      | ![http://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHClCkSR0cbqFK0xrglXwP4GMk9aibLrG98nTFaVJt3NCNLam1bWjwvrRTSIeuSTI2VjftNKx61qwuA/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image016.png) |

**4)** 对于输出层第L层:

三.cnn反向传播算法和更新参数

![img](file:///C:/Users/邓琬媚/AppData/Local/Temp/msohtmlclip1/01/clip_image017.png)

 