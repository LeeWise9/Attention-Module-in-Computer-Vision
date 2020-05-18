# Attention-Module-in-Computer-Vision
This project summarizes the attention module in computer vision, including the principle and implementation.


本文将总结一些 CV 算法中使用的的注意力机制。

## SENET<br>
原理：https://blog.csdn.net/xys430381_1/article/details/89158063

SENET 的注意力机制是通道域的，使网络给一些 channel 分配更高的权重值。

Sequeeze-and-Excitation(SE) block 并不是一个完整的网络结构，而是一个子结构，可以嵌到其他分类或检测模型中，作者采用SENet block和ResNeXt结合在ILSVRC 2017 的分类项目中拿到第一，在ImageNet数据集上将 top-5 error 降低到 2.251%，原先的最好成绩是 2.991%。

<p align="center">
	<img src="https://github.com/taki0112/SENet-Tensorflow/blob/master/assests/senet_block.JPG" alt="Sample"  width="800">
</p>

<p align="center">
	<img src="https://github.com/hujie-frank/SENet/raw/master/figures/SE-Inception-module.jpg" alt="Sample"  width="400">
</p>

<p align="center">
	<img src="https://github.com/hujie-frank/SENet/raw/master/figures/SE-ResNet-module.jpg" alt="Sample"  width="400">
</p>
<br>
<br>

## CBAM<br>

CBAM 相比于 SENET 不仅有 Channel Attention Module，还有 Spatial Attention Module，即通道域 + 空间域。

通道域让网络学会 “look what”，空间则是“look where”。这两个模块的实现都是基于全局池化的。

<p align="center">
	<img src="https://github.com/kobiso/CBAM-keras/raw/master/figures/overview.png" alt="Sample"  width="600">
</p>

<p align="center">
	<img src="https://github.com/kobiso/CBAM-keras/raw/master/figures/submodule.png" alt="Sample"  width="600">
</p>
<br>
<br>

## Residual Attention Network<br>
原理：https://www.jianshu.com/p/eb95c369fede

图中左图显示了在残差注意力网络中主干网络和注意力模块之间的关系，注意力模块为主干网络以某一个特征图为节点的分叉子网络；右图中的结果显示网络模型中，不同层特征图响应的注意力不同，在浅层结构中，网络的注意力集中于背景等区域，而在深层结构中，网络的注意力特征图（Attention Feature Map）聚焦于待分类的物体。这印证了深层次的特征图具有更高的抽象性和语义表达能力，对于物体分类较浅层特征有较大的作用。

<p align="center">
	<img src="https://upload-images.jianshu.io/upload_images/13655458-ef57547ec9dad4fc.png" alt="Sample"  width="600">
</p>

残差注意力网络由多层注意力模块堆叠而成，每个注意力模块包含了两个分支：mask branch 和 trunk branch。其中 trunk branch 可以是当前的任何一种SOTA 卷积神经网络模型。mask branch 的主要部分就是残差注意力学习机制。通过下采样(down sampling)和上采样(up sampling)，以及残差模块(residual unit)，组成了注意力的机制。然后使用点乘操作将两个分支的特征图组合在一起，得到最终的输出特征图。

<p align="center">
	<img src="https://upload-images.jianshu.io/upload_images/13655458-463350acd5f5c740.png" alt="Sample"  width="600">
</p>

<p align="center">
	<img src="https://upload-images.jianshu.io/upload_images/13655458-051308ea5a396fe6.png" alt="Sample"  width="400">
</p>

注意力模块可选三种类型：<br>
* Mix Attention：对每个通道和每个空间位置使用 Sigmoid，是混合域的注意力；<br>
* Channel Attention：对图片特征张量直接做全局平均池化（global average pooling），得到的是通道域的注意力（类比SENet）；<br>
* Spatial Attention：求图片特征张量在通道域上的平均值的激活函数，忽略了通道域的信息，从而得到空间域的注意力。<br>

<p align="center">
	<img src="https://upload-images.jianshu.io/upload_images/13655458-b6dc6501e38f85dd.png" alt="Sample"  width="300">
</p>
<br>
<br>

## Non-local Neural Networks<br>
原理：https://blog.csdn.net/elaine_bao/article/details/80821306

用 non-local similarity 来做图像 denoise。

主要思想：CNN 中的 convolution 单元每次只关注邻域 kernel size 的区域，就算后期感受野越来越大，终究还是局部区域的运算，这样就忽略了全局其他片区（比如很远的像素）对当前区域的贡献。

所以 non-local blocks 捕获这种 long-range 关系：对于 2D 图像，就是图像中任何像素对当前像素的关系权值；对于3D视频，就是所有帧中的所有像素，对当前帧的像素的关系权值。

<p align="center">
	<img src="https://img-blog.csdn.net/20180626215626172?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2VsYWluZV9iYW8=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70" alt="Sample"  width="400">
</p>
<br>
<br>

## DANet<br>
原理：https://blog.csdn.net/wumenglu1018/article/details/95949039

把 Self-attention 的思想用在图像分割，可通过 long-range 上下文关系更好地做到精准分割。

主要思想：把 deep feature map 进行 spatial-wise self-attention，同时也进行 channel-wise self-attetnion，最后将两个结果进行 element-wise sum 融合。

在 CBAM 分别进行空间和通道 self-attention的思想上，直接使用了 non-local 的自相关矩阵 Matmul 的形式进行运算，避免了 CBAM 手工设计 pooling，多层感知器等复杂操作。

Attention map 计算的是所有像素与所有像素之间的相似性，空间复杂度为 (HxW)x(HxW) 。

<p align="center">
	<img src="https://pic2.zhimg.com/v2-f6e56a4d34e1dfc38520b93c33b9525c_1200x500.jpg" alt="Sample"  width="600">
</p>
<br>
<br>

## CCNet<br>

在 DANet 的基础上减少了计算量（牺牲了准确率），前者 attention map 计算的空间复杂度为 (HxW)x(HxW)，本文采用 criss-cross，只计算每个像素与其同行同列即上的像素的相似性，间接计算到每个像素与每个像素的相似性，将空间复杂度降为 (HxW)x(H+W-1)。

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20190509150804954.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h5czQzMDM4MV8x,size_16,color_FFFFFF,t_70" alt="Sample"  width="500">
</p>

CCNet 的网络的架构与 DANet 相同，但是 attention 不同：在计算矩阵相乘时每个像素只抽取特征图中对应十字位置的像素进行点乘，计算相似度。

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20190509151055122.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h5czQzMDM4MV8x,size_16,color_FFFFFF,t_70" alt="Sample"  width="500">
</p>
<br>
<br>

## PAN<br>
Pyramid Attention Network for Semantic Segmentation

将 Attention 机制与金字塔结构结合，可以在高层语义指导的基础上提取相对于较低层的精确的密集特征，取代了其他方法里面的复杂的空洞卷积 dilated 和多个编码解码器的操作；使用全局 pooling 给底层特征加权，用于选取特征 map。

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20190417111047993.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h5czQzMDM4MV8x,size_16,color_FFFFFF,t_70" alt="Sample"  width="700">
</p>

如图所示有两个注意力模块：FPA 和 GAU。

#### FPA（Feature Pyramid Attention）<br>
针对 SPP 结构的一些缺点（pooling 丢失局部信息，对 small parts 预测不好，棋盘伪影效应，缺乏上下文信息等）提出优化方案。

* 1.提取到高层特征后不再做 pooling 的操作，而通过卷积提取更高层的语义；<br>
* 2.基于更高层语义的 Attention 机制，加强具有物体信息的部位带有的权值，得到带有 Attention 的输出；<br>
* 3.金字塔卷积结构采用不同大小的卷积核，代表着不同的感受野，应对不同 scale 的问题。<br>

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20190417111743947.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h5czQzMDM4MV8x,size_16,color_FFFFFF,t_70" alt="Sample"  width="700">
</p>

#### GAU（Global Attention Upsample）<br>
针对上采样重构图像计算量大的问题，设计一种新的高低层融合方法。

抛弃了 decoder 的结构，用 Global Pooling 得到高层的权重，低层特征经过卷积实现与高层特征相同数量的 map，map 和低层相乘后再与高层相加。

<p align="center">
	<img src="https://img-blog.csdnimg.cn/20190417111623705.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3h5czQzMDM4MV8x,size_16,color_FFFFFF,t_70" alt="Sample"  width="500">
</p>
<br>
<br>






















