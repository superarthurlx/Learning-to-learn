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
    def call(self, W):
        return tf.matmul(W, self.theta)

model = MyModel()

def calc_loss(model, W, y): # 一个sample的loss
    error = model(W) - y
    return tf.reduce_mean(tf.square(error))

loss_history = []
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

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
    #print("loss:", loss_value, "grads:", grads, model_theta.shape)
    #print(grads, "\n", model.theta, "\n---------------------------------------")
    optimizer.apply_gradients(zip(grads, [model.theta]))

def train(epochs):
    W, y = get_n_samples(n_dimension, num_samples)
    for epoch in range(epochs):
        train_step(W, y)
        print ('Epoch {} finished'.format(epoch))


def main():
    if optim_method == 'SGD':
        train(epochs = max_epoch)
        print(loss_history, sep='\n')

if __name__ == "__main__":
    main()
