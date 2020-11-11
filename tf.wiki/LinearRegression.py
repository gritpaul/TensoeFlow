import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
tic0 = timer()
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

tic1 = timer()
# numpy
a, b = 0, 0
num_epoch = 1000
learning_rate = 5e-2
for e in range(num_epoch):
    # 手动计算损失函数关于自变量（模型参数）的梯度
    y_pred = a * X + b
    grad_a, grad_b = 2 * (y_pred - y).dot(X), 2 * (y_pred - y).sum()

    # 更新参数
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b
print(a, b)
toc1 = timer()
print(toc1 - tic1) # 输出的时间，秒为单位

tic2 = timer()
# tf

X = tf.constant(X)
y = tf.constant(y)

a = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 1000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-2)
for e in range(num_epoch):
    # 使用tf.GradientTape()记录损失函数的梯度信息
    with tf.GradientTape() as tape:
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y))

    # TensorFlow自动计算损失函数关于自变量（模型参数）的梯度
    grads = tape.gradient(loss, variables)

    # TensorFlow自动根据梯度更新参数a, b
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)
toc2 = timer()
print(toc2 - tic2) # 输出的时间，秒为单位

toc0 = timer()
print(toc0 - tic0) # 输出的时间，秒为单位

#此处自动求导的梯度下降慢于手工求导的方式，因为后者节约了时间。但很多函数无法手工求导。