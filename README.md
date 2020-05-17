# Attention-Module-in-Computer-Vision
This project summarizes the attention module in computer vision, including the principle and implementation.


本文将总结一些 CV 算法中使用的的注意力机制。

## SENET<br>

SENET 的注意力机制是通道域的，使网络给一些 channel 分配更高的权重值。

Sequeeze-and-Excitation(SE) block 并不是一个完整的网络结构，而是一个子结构，可以嵌到其他分类或检测模型中，作者采用SENet block和ResNeXt结合在ILSVRC 2017 的分类项目中拿到第一，在ImageNet数据集上将 top-5 error 降低到 2.251%，原先的最好成绩是 2.991%。

<p align="center">
	<img src="https://github.com/taki0112/SENet-Tensorflow/blob/master/assests/senet_block.JPG" alt="Sample"  width="800">
</p>

<p align="center">
	<img src="https://github.com/hujie-frank/SENet/raw/master/figures/SE-Inception-module.jpg" alt="Sample"  width="400">
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



