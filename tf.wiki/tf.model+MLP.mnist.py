import numpy as np
import tensorflow as tf


# 数据获取及预处理： tf.keras.datasets
# 预备工作，实现一个MNISTLoader类，来读取 MNIST 数据集。
class MNISTLoader:
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        # 在 TensorFlow 中，图像数据集的一种典型表示是 [图像数目，长，宽，色彩通道数] 的四维张量
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)  # [60000]
        self.test_label = self.test_label.astype(np.int32)  # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, self.num_train_data, batch_size)
        return self.train_data[index, :], self.train_label[index]


# 模型的构建： tf.keras.Model 和 tf.keras.layers
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()  # Flatten层将除第一维（batch_size）以外的维度展平/拉直
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)  # 全连接层1/hidden layer1
        self.dense2 = tf.keras.layers.Dense(units=10)  # output

    def call(self, inputs):  # [batch_size, 28, 28, 1]
        x = self.flatten(inputs)  # [batch_size, 784]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)  # 令输出归一化，且和为1，凸显最大值，抑制小量
        return output


# 模型的训练： tf.keras.losses 和 tf.keras.optimizer
# 定义一些模型超参数：
num_epochs = 5
batch_size = 50
learning_rate = 0.001
# 实例化模型和数据读取类，并实例化一个 tf.keras.optimizer 的优化器（这里使用常用的 Adam 优化器）：
model = MLP()  # 实例化模型类
data_loader = MNISTLoader()  # 实例化数据读取类
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  # 实例化优化器


# 然后迭代进行以下步骤：
# 从 data_Loader : 中随机取一批训练数据； batch_size = 50
# 将这批数据送入模型，计算出模型的预测值；
# 将模型预测值与真实值进行比较，计算损失函数（loss）。这里使用 tf.keras.losses 中的交叉熵函数作为损失函数；
# 计算损失函数关于模型变量的导数；
# 将求出的导数值传入优化器，使用优化器的 apply_gradients 方法更新模型参数以最小化损失函数（优化器的详细使用方法见 tf.model+layers.py ）。
num_batches = int(data_loader.num_train_data // batch_size * num_epochs)  # 除法，后向下取整
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(X)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)  # 求平均
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))


# 模型的评估： tf.keras.metrics
# 使用 tf.keras.metrics 中的 SparseCategoricalAccuracy 评估器来评估模型在测试集上的性能
sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(data_loader.num_test_data // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(data_loader.test_data[start_index: end_index])
    sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())
