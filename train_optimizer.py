import numpy as np
import tensorflow as tf

# 说明：
# 数据个数 num_samples 个，每个数据都是以同一个theta作为最小值点的 n_dimension 元二次函数
# 每个 epoch 只有一个batch，就是把所有 num_samples 个函数的loss加起来作为总loss然后进行优化
# 一共有 max_epoch 个epoch

# 如果这 num_samples 个用的是同一个theta最小值的话，效果更好，但是但一个值后就无法继续下降了

num_samples = 10 # 随机取样的函数个数
n_unroll = 20 # BPTT中unroll的数量
n_dimension = 3 # 原问题中f的参数的数量
hidden_size = 20 # LSTM中隐藏层的大小
num_layers = 2 # LSTM的层数

max_epoch = 50 # 训练optimizer的epoch个数, 每个epoch会取样num_samples个，每个会展开n_unroll次

# W 和 y 是训练数据, theta是模型参数
def get_n_samples(n_dimension, n): # 一次取得n个样本
    theta = np.random.randn(n_dimension, 1)
    W = np.random.randn(n, n_dimension, n_dimension)
    y = np.zeros([n, n_dimension, 1])
    for i in range(n):
        y[i] = np.dot(W[i], theta)
    W = tf.convert_to_tensor(W, dtype = 'float')
    y = tf.convert_to_tensor(y, dtype = 'float')
    return W, y

def construct_graph_and_train_optimizer():
    g = tf.Graph()
    print("Graph Construction Begin")
    with g.as_default():
        Wn, yn = get_n_samples(n_dimension, num_samples)
        cell_list = []
        with tf.compat.v1.variable_scope(tf.compat.v1.get_variable_scope()) as scope:
            for i in range(n_dimension): 
                # 对每个f的参数有一个lstm(2层)，不同f参数之间的lstm之间参数共享
                cell_list.append(tf.compat.v1.nn.rnn_cell.MultiRNNCell([
                    tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hidden_size, reuse=tf.compat.v1.get_variable_scope().reuse) 
                    for _ in range(num_layers)])) 

            loss = 0
            for cur in range(num_samples):
                print("sample:", cur)
                # 随机取样一个n元二次函数
                W = tf.random.normal([n_dimension, n_dimension]);
                y = tf.random.normal([n_dimension, 1])
                #W = Wn[i]
                #y = yn[i]
                theta = tf.random.normal([n_dimension, 1], dtype='float')


                batch_size = 1 # 一次送一个f进去
                state_list = [cell_list[i].zero_state(batch_size, tf.float32) for i in range(n_dimension)] # 保存每个lstm的隐藏层状态

                # BPTT 
                sum_f = 0
                for t in range(n_unroll): # 这里和论文不太一样，只进行了n_unroll=20次（因为不太会写truncated BPTT）
                    f = tf.reduce_sum(input_tensor=tf.square(tf.matmul(W, theta) - y)) # 每次的theta都不一样
                    grad_f = tf.gradients(ys=f, xs=theta)[0] # 计算f对theta的梯度（注意返回对象是个list）

                    g_new_list = []
                    for i in range(n_dimension):
                        cell = cell_list[i]
                        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", i);
                        #print(cell)
                        #print("--------------------------------------")
                        state = state_list[i]
                        grad_h_t = tf.slice(grad_f, begin=[i, 0], size=[1, 1])
                        # 当前f的第i个参数对theta的梯度

                        if(i>0): 
                            tf.compat.v1.get_variable_scope().reuse_variables()
                        cell_output, state = cell(grad_h_t, state) 
                        g_new_i = tf.reduce_sum(input_tensor=cell_output) # 使用lstm输出状态的权值和作为f的参数i的变化量
                        g_new_list.append(g_new_i)

                    # 转换成tensor
                    g_new = tf.reshape(tf.squeeze(tf.stack(g_new_list)), [n_dimension, 1])  # [n_dimension, 1] tensor
                    theta = tf.add(theta, g_new) # 更新f的参数theta
                    f_at_theta_t = tf.reduce_sum(input_tensor=tf.square(tf.matmul(W, theta) - y)) # 计算现在的loss
                    sum_f += f_at_theta_t

                loss += sum_f

        loss = loss / num_samples
        tvars = tf.compat.v1.trainable_variables()  
        grads = tf.gradients(ys=loss, xs=tvars)
        lr = 0.001  
        optimizer = tf.compat.v1.train.AdamOptimizer(lr) # 使用学习率0.001的adam优化器
        train_op = optimizer.apply_gradients(zip(grads, tvars)) 

    print("Graph Construction End")

    with tf.compat.v1.Session(graph=g) as sess:
        sess.run(tf.compat.v1.global_variables_initializer()) # 初始化变量

        #print(tf.trainable_variables())
        for epoch in range(max_epoch):
            cost, _ = sess.run([loss, train_op]) # 计算loss并更新lstm的参数
            print("Epoch {:d} , loss {:f}".format(epoch, cost))

        import pickle
        print("Saving variables of optimizer...")
        variable_dict = {}
        for var in tf.compat.v1.trainable_variables():
            #print(var.name)
            #print(var.eval())
            variable_dict[var.name] = var.eval()
        with open("variable_dict.pickle", "wb") as f:
            pickle.dump(variable_dict, f)
            print("Saved successfully!")
        # 保存优化器参数

if __name__ == "__main__":
    construct_graph_and_train_optimizer()
