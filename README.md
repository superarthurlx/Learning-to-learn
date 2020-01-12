# Learning to learn by gradient descent by gradient descent

项目名称：Learning to learn by gradient descent by gradient descent 复现
参考论文：Learning to learn by gradient descent by gradient descent, 2016, NIPS
项目成员：唐雯豪(@thwfhk), 巫子辰(@SuzumeWu), 杜毕安(@scncdba), 王昕兆(@wxzsan)
项目地址：https://github.com/thwfhk/Learning-to-learn
指导成员：黄佳磊

## 要求

```
python 3.5.2
tensorflow 1.10
numpy 1.13
```

## 说明

只实现了论文中的第一个部分，训练一个寻找二次函数最小值的优化器。

运行`python train_optimizer.py` 来训练一个optimizer

运行`python main.py` 来测试训练好的optimizer的效果

参数的信息在代码中有详细说明，期中`main.py`可以使用SGD来进行效果对比

## 效果展示

当前复现的效果比SGD差一点。

使用lstm训练的optimizer的优化效果:![](https://github.com/thwfhk/Learning-to-learn/blob/master/figure_lstm.png)

使用SGD的优化效果：![](https://github.com/thwfhk/Learning-to-learn/blob/master/figure_SGD.png)