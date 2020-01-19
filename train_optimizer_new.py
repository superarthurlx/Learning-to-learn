import numpy as np
import tensorflow as tf

# 说明：
# 数据个数 num_samples 个，每个数据都是以同一个theta作为最小值点的 n_dimension 元二次函数
# 每个 epoch 只有一个batch，就是把所有 num_samples 个函数的loss加起来作为总loss然后进行优化
# 一共有 max_epoch 个epoch

# 如果这 num_samples 个用的是同一个theta最小值的话，效果更好，但是但一个值后就无法继续下降了

num_samples = 1 # 随机取样的函数个数
n_unroll = 20 # BPTT中unroll的数量
n_dimension = 3 # 原问题中f的参数的数量
hidden_size = 20 # LSTM中隐藏层的大小
num_layers = 2 # LSTM的层数
batch_size = 1

max_epoch = 40 # 训练optimizer的epoch个数, 每个epoch会取样num_samples个，每个会展开n_unroll次

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


#--------------------------------------------------------
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


LSTM_optimizer = LSTMOptimizer(hidden_size, num_layers, n_dimension, batch_size=1)

def train_step():
    Wn, yn = get_n_samples(n_dimension, num_samples)

    with tf.GradientTape(persistent=True) as tape:
        #loss = tf.Tensor(0)
        loss = 0
        for cur in range(num_samples):
            print("sample:", cur)
            W = Wn[cur]
            y = yn[cur]
            model = MyModel()
            tape.watch(model.theta)

            LSTM_optimizer.refresh_state()

            # BPTT 
            sum_f = 0
            for t in range(n_unroll): # 这里和论文不太一样，只进行了n_unroll=20次（因为不太会写truncated BPTT）
                f = model.calc_loss(W, y)
                #print(f, type(f))
                grad_f = tape.gradient(target = f, sources = model.theta)

                g_new = LSTM_optimizer(grad_f)
                model.theta = model.theta + g_new
                f_at_theta_t = model.calc_loss(W, y)
                sum_f += f_at_theta_t

            loss += sum_f

        loss = loss / num_samples
        variables = LSTM_optimizer.variables
        grads = tape.gradient(loss, variables)
        optimizer = tf.keras.optimizers.Adam()
        optimizer.apply_gradients(zip(grads, variables)) 
    
    return loss

def train():
    #print(tf.trainable_variables())
    for epoch in range(max_epoch):
        loss = train_step()
        print("Epoch: {:d} , loss: {:f}".format(epoch, loss))

    # 保存optimizer参数
    #LSTM_optimizer.save('LSTM_optimizer.tf', save_format='tf')
    #LSTM_optimizer.save_weights('LSTM_optimizer.h5')
    checkpoint = tf.train.Checkpoint(model=LSTM_optimizer)
    checkpoint.save('LSTM_optimizer.tf')

if __name__ == "__main__":
    train()