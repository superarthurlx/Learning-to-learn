import tensorflow as tf
import numpy as np

# 说明：
# 数据个数 num_samples 个，每个数据都是以同一个theta作为最小值点的 n_dimension 元二次函数
# 每个 epoch 只有一个batch，就是把所有 num_samples 个函数的loss加起来作为总loss然后进行优化
# 一共有 max_epoch 个epoch

num_samples = 1 # 随机取样的函数个数(一个epoch的大小) 
#batch_size = 1

n_unroll = 20 # BPTT中unroll的数量
n_dimension = 5 # 原问题中f的参数的数量
hidden_size = 20 # LSTM中隐藏层的大小
num_layers = 2 # LSTM的层数

max_epoch = 20

optim_method = "SGD"

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

# -------------------------------------------------
normal_init = tf.random_normal_initializer()

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.theta = tf.Variable(initial_value = normal_init(shape=(n_dimension, 1), dtype='float'))
    def __call__(self, W):
        return tf.matmul(W, self.theta)
    def calc_loss(self, W, y):
        error = self(W) - y
        return tf.reduce_mean(tf.square(error))

model = MyModel()

loss_history = []
lr = 0.1
SGD_optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

class LSTMOptimizer(tf.keras.Model):
    def __init__(self, hidden_size, num_layers, n_dimension, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.cell_list = []
        for i in range(n_dimension):
            self.cell_list.append(tf.keras.layers.StackedRNNCells([tf.keras.layers.LSTMCell(hidden_size) for _ in range(num_layers)]))

        self.state_list = [self.cell_list[i].get_initial_state(batch_size = self.batch_size, dtype = 'float') for i in range(n_dimension)]
        self.time = 0

    def refresh_state(self):
        self.state_list = [self.cell_list[i].get_initial_state(batch_size = self.batch_size, dtype = 'float') for i in range(n_dimension)]
        self.time = 0

    def __call__(self, grad_f):
        #print("----------------------grad_f\n", grad_f)
        self.time += 1
        g_new_list = []
        for i in range(n_dimension):
            cell = self.cell_list[i]
            state = self.state_list[i]
            grad_h_t = tf.slice(grad_f, begin=[i, 0], size=[1, 1]) # 当前f的第i个参数对theta的梯度

            cell_output, state = cell(grad_h_t, state) 
            self.state_list[i] = state # ！！！应该有这个吧
            g_new_i = tf.reduce_sum(input_tensor=cell_output) # 使用lstm输出状态的权值和作为f的参数i的变化量
            g_new_list.append(g_new_i)

        # 转换成tensor
        g_new = tf.reshape(tf.squeeze(tf.stack(g_new_list)), [n_dimension, 1])  # [n_dimension, 1] tensor
        return g_new

LSTM_optimizer = LSTMOptimizer(hidden_size, num_layers, n_dimension, batch_size=1)
checkpoint = tf.train.Checkpoint(model=LSTM_optimizer)   # 键名保持为“myAwesomeModel”
checkpoint.restore('./models/LSTM_optimizer.tf-1')

# 这里每个 epoch 都是同一个函数，相当于对一个函数做了 max_epoch 次
def train_step(W, y): 
    LSTM_optimizer.refresh_state()

    with tf.GradientTape() as tape:
        loss_value = 0
        for i in range(num_samples):
            Wi = W[i]
            yi = y[i]
            loss_value += model.calc_loss(Wi, yi)
        loss_value /= num_samples
    
    loss_history.append(loss_value.numpy())
    grads = tape.gradient(target=loss_value, sources=[model.theta])

    #print(grads, "\n", model.theta, "\n---------------------------------------")

    if optim_method == 'SGD':
        #optimizer.apply_gradients(zip(grads, [model.theta]))
        model.theta.assign_sub(lr * grads[0]) # 手动进行SGD
    elif optim_method == 'lstm':
        g_new_val = LSTM_optimizer(grads[0].numpy())
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

    import matplotlib.pyplot as plt
    #loss_history = loss_history[3:]
    plt.plot(range(len(loss_history)), loss_history)
    imagename = "figure_" + optim_method + ".png"
    plt.savefig(imagename)	
    plt.show()

if __name__ == "__main__":
    main()