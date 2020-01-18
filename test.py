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
        cell_list.append(tf.compat.v1.nn.rnn_cell.MultiRNNCell([tf.compat.v1.nn.rnn_cell.BasicLSTMCell(
            hidden_size, reuse=tf.compat.v1.get_variable_scope().reuse) for _ in range(num_layers)])) 
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



g = tf.Graph()
sess = tf.compat.v1.Session(graph = g)

## 将已训练的optimizier复原
import pickle
with open("variable_dict.pickle", "rb") as f:
    variable_dict = pickle.load(f)

def lstm_optimizer(grads):
    with sess.as_default():
        with g.as_default():
            g_op, grad_f = build_optimizer_graph()
            sess.run(tf.compat.v1.global_variables_initializer())

            for var in tf.compat.v1.trainable_variables():
                if var.name in variable_dict:
                    #print("[hi]", var.name)
                    assign_op = var.assign(variable_dict[var.name]) 
                    sess.run(assign_op)

            g_new_val = sess.run(g_op, feed_dict={grad_f: grads})
            return g_new_val


# -------------------------------------------------
normal_init = tf.random_normal_initializer()

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.theta = tf.Variable(initial_value = normal_init(shape=(n_dimension, 1), dtype='float'))
    def call(self, W):
        return tf.matmul(W, self.theta)

model = MyModel()

def calc_loss(model, W, y): # 一个sample的loss
    error = model(W) - y
    return tf.reduce_mean(tf.square(error))

loss_history = []
lr = 0.1
#optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

def train_step(W, y):
    with tf.GradientTape() as tape:
        loss_value = 0
        for i in range(num_samples):
            Wi = W[i]
            yi = y[i]
            loss_value += calc_loss(model, Wi, yi)
        loss_value /= num_samples
    
    loss_history.append(loss_value.numpy())
    grads = tape.gradient(target=loss_value, sources=[model.theta])

    #print(grads, "\n", model.theta, "\n---------------------------------------")

    if optim_method == 'SGD':
        #optimizer.apply_gradients(zip(grads, [model.theta]))
        model.theta.assign_sub(lr * grads[0]) # 手动进行SGD
    elif optim_method == 'lstm':
        g_new_val = lstm_optimizer(grads[0].numpy())
        #print("!!!!!!!!!!!!!!!", type(g_new_val), g_new_val.shape)
        model.theta.assign_sub(-g_new_val)


def train(epochs):
    W, y = get_n_samples(n_dimension, num_samples)
    for epoch in range(epochs):
        train_step(W, y)
        print ('Epoch {} finished'.format(epoch))


def main():
    train(epochs = max_epoch)
    print(loss_history, sep='\n')

if __name__ == "__main__":
    main()