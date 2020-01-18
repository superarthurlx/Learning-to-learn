import tensorflow as tf
import numpy as np

num_samples = 10 # 随机取样的函数个数
n_unroll = 20 # BPTT中unroll的数量
n_dimension = 5 # 原问题中f的参数的数量
hidden_size = 20 # LSTM中隐藏层的大小
num_layers = 2 # LSTM的层数

max_epoch = 20 # 训练optimizer的epoch个数, 每个epoch会取样num_samples个，每个会展开n_unroll次

optim_method = "lstm"

def get_n_samples(n_dimension, n): # 一次取得n个样本
    theta = np.random.randn(n_dimension, 1)
    W = np.random.randn(n, n_dimension, n_dimension)
    y = np.zeros([n, n_dimension, 1])
    for i in range(n):
        y[i] = np.dot(W[i], theta)
    return W, y


def build_optimizer_graph(): 
    grad_f = tf.compat.v1.placeholder(tf.float32, [n_dimension, 1]) # 占位符

    cell_list = []
    for i in range(n_dimension):
        cell_list.append(
            tf.compat.v1.nn.rnn_cell.MultiRNNCell([tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hidden_size, reuse=tf.compat.v1.get_variable_scope().reuse)
                                         for _ in range(num_layers)])) 
    batch_size = 1
    state_list = [cell_list[i].zero_state(batch_size, tf.float32) for i in range(n_dimension)]
    g_new_list = []
    for i in range(n_dimension): # 遍历整个维度
        cell = cell_list[i]
        state = state_list[i]
        grad_h_t = tf.slice(grad_f, begin=[i, 0], size=[1, 1])

        if i > 0: tf.compat.v1.get_variable_scope().reuse_variables()
        cell_output, state = cell(grad_h_t, state) 
        g_new_i = tf.reduce_sum(input_tensor=cell_output)

        g_new_list.append(g_new_i)

    g_new = tf.reshape(tf.squeeze(tf.stack(g_new_list)), [n_dimension, 1]) 

    return g_new, grad_f


def build_training_graph(method):  
    n = num_samples
    W = tf.compat.v1.placeholder(tf.float32, shape=[n, n_dimension, n_dimension])
    y = tf.compat.v1.placeholder(tf.float32, shape=[n, n_dimension, 1])
    theta = tf.Variable(tf.random.truncated_normal([n_dimension, 1]))
    if method == "lstm":
        g_new = tf.compat.v1.placeholder(tf.float32, shape=[n_dimension, 1])

    loss = 0
    for i in range(n):
        W_i = tf.reshape(tf.slice(W, begin=[i, 0, 0], size=[1, n_dimension, n_dimension]), [n_dimension, n_dimension])
        y_i = tf.reshape(tf.slice(y, begin=[i, 0, 0], size=[1, n_dimension, 1]), [n_dimension, 1])
        f = tf.reduce_sum(input_tensor=tf.square(tf.matmul(W_i, theta) - y_i))  
        loss += f # 损失函数更新
    loss /= n

    f_grad = tf.gradients(ys=loss, xs=theta)[0]

    if method == "SGD":
        train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)
        return loss, train_op, W, y

    if method == "lstm":
        new_value = tf.add(theta, g_new)
        train_op = tf.compat.v1.assign(theta, new_value) 
        return loss, f_grad, train_op, g_new, W, y
  
  
def main():
    g = tf.Graph()
    with g.as_default():
        with tf.compat.v1.Session() as sess:
            if optim_method == "lstm":
                loss_op, f_grad_op, train_op, g_new_ph, W_ph, y_ph = build_training_graph(method=optim_method)
                g_op, f_grad_ph = build_optimizer_graph()
            elif optim_method == "SGD":
                loss_op, train_op, W_ph, y_ph = build_training_graph(method=optim_method)
            else:
                print("Error: Optimization Method Not Defined.")
                exit()

            sess.run(tf.compat.v1.global_variables_initializer())

            ## 将已训练的optimizier复原
            if optim_method == "lstm":
                import pickle
                with open("variable_dict.pickle", "rb") as f:
                    variable_dict = pickle.load(f)

                for var in tf.compat.v1.trainable_variables():
                    ## 给当前计算（图）进行赋值
                    if var.name in variable_dict:
                        assign_op = var.assign(variable_dict[var.name]) 
                        sess.run(assign_op)

            W_val, y_val = get_n_samples(n_dimension, num_samples)

            ## 更新参数
            cost_list = []
            if optim_method == "lstm":
                g_new_val = np.zeros([n_dimension, 1])
                for epoch in range(max_epoch):
                    loss_val, f_grad_val, _ = sess.run([loss_op, f_grad_op, train_op],
                                                       feed_dict={g_new_ph: g_new_val, W_ph: W_val, y_ph: y_val})
                    g_new_val = sess.run(g_op, feed_dict={f_grad_ph: f_grad_val})
                    print(loss_val)
                    cost_list.append(loss_val)
            elif optim_method == "SGD":
                for epoch in range(max_epoch):
                    loss_val, _ = sess.run([loss_op, train_op], feed_dict={W_ph: W_val, y_ph: y_val})
                    print(loss_val)
                    cost_list.append(loss_val)

            ## 绘制图
            import matplotlib.pyplot as plt
            cost_list = cost_list[3:]
            plt.plot(range(len(cost_list)), cost_list)
            imagename = "figure_" + optim_method + ".png"
            plt.savefig(imagename)	
	

if __name__ == "__main__":
    main()
