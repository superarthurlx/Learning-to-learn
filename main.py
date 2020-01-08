import tensorflow as tf
import numpy as np

num_samples = 10 # 随机取样的函数个数
n_unroll = 20 # BPnum_samplesnum_samples中unroll的数量
n_dimension = 3 # 原问题中f的参数的数量
hidden_size = 20 # LSTM中隐藏层的大小
num_layers = 2 # LSTM的层数

max_epoch = 20 # 使用optimizer训练目标函数的epoch数

optim_method = "SGD"

def get_n_samples(n_dimension, n): # 一次取得n个样本
    theta = np.random.randn(n_dimension, 1)
    W = np.random.randn(n, n_dimension, n_dimension)
    y = np.zeros([n, n_dimension, 1])
    for i in range(n):
        y[i] = np.dot(W[i], theta)
    return W, y

if __name__ == "__main__":
    train()