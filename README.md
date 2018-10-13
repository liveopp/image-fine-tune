# 基于TensorFlow的图片Transfer Learning

对已有的CNN网络（如Inception v3, VGG16), 用新的数据和label信息重新训练。
可以支持只训练最后一层，可以调整所有layer的参数。

##
相比于[TF-slim](https://github.com/tensorflow/models/tree/master/research/slim#Tuning)的优点

1. 支持新的dataset api, 无需先将图片转成tfrecord, 可以直接读入训练
2. 支持更多的数据增强模式
3. 可以先优化最后一层，之后调整所有layer参数, 以达到最优效果
4. 支持只在一部分类别上训练，并且可以随时中断迭代，查看测试集效果，方便测试
