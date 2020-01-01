import numpy as np
import tensorflow as tf

T = 10 # 随机取样的函数个数
n_unroll = 20 # BPTT中unroll的数量
n_dimension = 3 # 原问题中f的参数的数量
hidden_size = 20 # LSTM中隐藏层的大小
num_layers = 2 # LSTM的层数

def construct_graph():
    g = tf.Graph()
    print("Graph Construction Begin")
    with g.as_default():
        cell_list = []
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in range(n_dimension): 
                # 对每个f的参数有一个lstm(2层)，不同f参数之间的lstm之间参数共享
                cell_list.append(tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse) 
                    for _ in range(num_layers)])) 

            loss = 0
            for t in range(T): # 最后的loss对T个样本取平均
                print("t:", t);
                # 随机取样一个n元二次函数
                W = tf.random_normal([n_dimension, n_dimension]);
                y = tf.random_normal([n_dimension, 1])
                theta = tf.random_normal([n_dimension, 1])
                f = tf.reduce_sum(tf.square(tf.matmul(W, theta) - y))
                grad_f = tf.gradients(f, theta)[0] # 计算f对theta的梯度（注意返回对象是个list）

                batch_size = 1 # 一次送一个f进去
                state_list = [cell_list[i].zero_state(batch_size, tf.float32) for i in range(n_dimension)] # 保存每个lstm的隐藏层状态
                sum_f = 0
                g_new_list = []
                for i in range(n_dimension):
                    cell = cell_list[i]
                    state = state_list[i]
                    grad_h_t = tf.slice(grad_f, begin=[i, 0], size=[1, 1])
                    # 当前f的第i个参数对theta的梯度

                    # BPTT
                    for k in range(n_unroll):
                        if k > 0: 
                            tf.get_variable_scope().reuse_variables()
                        cell_output, state = cell(grad_h_t, state) 
                        g_new_i = tf.reduce_sum(cell_output) # 使用lstm输出状态的权值和作为f的参数i的变化量

                    g_new_list.append(g_new_i)

                # 转换成tensor
                g_new = tf.reshape(tf.squeeze(tf.stack(g_new_list)), [n_dimension, 1])  # [n_dimension, 1] tensor
                theta = tf.add(theta, g_new) # 更新f的参数theta
                f_at_theta_t = tf.reduce_sum(tf.square(tf.matmul(W, theta) - y)) # 计算现在的loss

                loss += f_at_theta_t

        loss = loss / T
        tvars = tf.trainable_variables()  
        grads = tf.gradients(loss, tvars)
        lr = 0.001  
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        print("Graph Construction End")