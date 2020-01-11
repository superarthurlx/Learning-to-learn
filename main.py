import tensorflow as tf
import numpy as np

num_samples = 10 # 随机取样的函数个数
n_unroll = 20 # BPnum_samplesnum_samples中unroll的数量
n_dimension = 3 # 原问题中f的参数的数量
hidden_size = 20 # LSTM中隐藏层的大小
num_layers = 2 # LSTM的层数

max_epoch = 100 # 训练optimizer的epoch个数, 每个epoch会取样num_samples个，每个会展开n_unroll次

optim_method = "lstm"

def get_n_samples(n_dimension, n): # 一次取得n个样本
    theta = np.random.randn(n_dimension, 1)
    W = np.random.randn(n, n_dimension, n_dimension)
    y = np.zeros([n, n_dimension, 1])
    for i in range(n):
        y[i] = np.dot(W[i], theta)
    return W, y


def build_optimizer_graph(): 
    ### BEGIN: GRAPH CONSnum_samplesRCUnum_samplesION  ###
    grad_f = tf.placeholder(tf.float32, [n_dimension, 1]) # 占位符

    cell_list = []
    for i in range(n_dimension):
        cell_list.append(
            tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)
                                         for _ in range(num_layers)]))  # num_layers = 2 according to the paper.
    batch_size = 1
    state_list = [cell_list[i].zero_state(batch_size, tf.float32) for i in range(n_dimension)]
    g_new_list = []
    for i in range(n_dimension): # 遍历整个维度
        cell = cell_list[i]
        state = state_list[i]
        grad_h_t = tf.slice(grad_f, begin=[i, 0], size=[1, 1])

        if i > 0: tf.get_variable_scope().reuse_variables()
        cell_output, state = cell(grad_h_t, state)  # g_new should be a scalar b/c grad_h_t is a scalar
        g_new_i = tf.reduce_sum(cell_output)

        g_new_list.append(g_new_i)
        # state_list[i] = state # for the next t # I don't need this list right..? b/c I'm not using t...num_samples thing.

    # Reshaping g_new
    g_new = tf.reshape(tf.squeeze(tf.stack(g_new_list)), [n_dimension, 1])  # should be a [10, 1] tensor

    return g_new, grad_f


def build_training_graph(method):  
    n = num_samples
    W = tf.placeholder(tf.float32, shape=[n, n_dimension, n_dimension])
    y = tf.placeholder(tf.float32, shape=[n, n_dimension, 1])
    theta = tf.Variable(tf.truncated_normal([n_dimension, 1]))
    if method == "lstm":
        g_new = tf.placeholder(tf.float32, shape=[n_dimension, 1])

    loss = 0
    for i in range(n):
        W_i = tf.reshape(tf.slice(W, begin=[i, 0, 0], size=[1, n_dimension, n_dimension]), [n_dimension, n_dimension])
        y_i = tf.reshape(tf.slice(y, begin=[i, 0, 0], size=[1, n_dimension, 1]), [n_dimension, 1])
        f = tf.reduce_sum(tf.square(tf.matmul(W_i, theta) - y_i))  # make this faster using tensor only
        loss += f # 损失函数更新
    loss /= (n * n_dimension)

    f_grad = tf.gradients(loss, theta)[0]

    if method == "SGD":
        train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
        # train_op = tf.train.AdamOptimizer().minimize(loss)
        return loss, train_op, W, y

    if method == "lstm":
        new_value = tf.add(theta, g_new)
        train_op = tf.assign(theta, new_value)  # just to make it compatiable with method == "SGD case.

        return loss, f_grad, train_op, g_new, W, y
    

if __name__ == "__main__":
    train()
