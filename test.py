import tensorflow as tf
import numpy as np

# 说明：
# 数据个数 num_samples 个，每个数据都是以同一个theta作为最小值点的 n_dimension 元二次函数
# 每个 epoch 只有一个batch，就是把所有 num_samples 个函数的loss加起来作为总loss然后进行优化
# 一共有 max_epoch 个epoch

num_samples = 10 # 随机取样的函数个数(一个epoch的大小) 
#batch_size = 10

n_unroll = 20 # BPTT中unroll的数量
n_dimension = 5 # 原问题中f的参数的数量
hidden_size = 20 # LSTM中隐藏层的大小
num_layers = 2 # LSTM的层数

max_epoch = 20 

optim_method = "lstm"

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

def LSTMCell():
    return tf.compat.v1.nn.rnn_cell.BasicLSTMCell(hidden_size, reuse=tf.compat.v1.get_variable_scope().reuse)

class LSTMOptimizer():
    def __init__(self):
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph = self.graph)

        with self.graph.as_default():
            self.grad_f = tf.compat.v1.placeholder('float', [n_dimension, 1])

            cell_list = []
            for i in range(n_dimension):
                cell_list.append(tf.compat.v1.nn.rnn_cell.MultiRNNCell([LSTMCell() for _ in range(num_layers)])) 
            batch_size = 1
            state_list = [cell_list[i].zero_state(batch_size, tf.float32) for i in range(n_dimension)]
            g_new_list = []
            for i in range(n_dimension): # 遍历每个维度
                cell = cell_list[i]
                state = state_list[i]
                grad_h_t = tf.slice(self.grad_f, begin=[i, 0], size=[1, 1])

                if i > 0: tf.compat.v1.get_variable_scope().reuse_variables()

                cell_output, state = cell(grad_h_t, state) 
                g_new_i = tf.reduce_sum(input_tensor=cell_output)

                g_new_list.append(g_new_i)

            self.g_new = tf.reshape(tf.squeeze(tf.stack(g_new_list)), [n_dimension, 1]) 

            self.sess.run(tf.compat.v1.global_variables_initializer())

            ## 将已训练的optimizier复原
            import pickle
            with open("variable_dict.pickle", "rb") as f:
                variable_dict = pickle.load(f)
            
            for var in tf.compat.v1.trainable_variables():
                if var.name in variable_dict:
                    #print("[hi]", var.name)
                    assign_op = var.assign(variable_dict[var.name]) 
                    self.sess.run(assign_op)

    
    def optimize(self, grads):
        g_new_val = self.sess.run(self.g_new, feed_dict={self.grad_f: grads})
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
SGD_optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
lstm_optimizer = LSTMOptimizer()

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
        g_new_val = lstm_optimizer.optimize(grads[0].numpy())
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