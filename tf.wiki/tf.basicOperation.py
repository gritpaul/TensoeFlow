import tensorflow as tf

# tf.compat.v1.disable_eager_execution()
# 关闭tf2的即时执行
# 定义一个随机数（标量）
random_float = tf.random.uniform(shape=())
# 缺省shape值为空 纯量张量
print(random_float)
# 定义一个有3行2列的零向量,定义整形值
zero_vector = tf.zeros(shape=(3, 2), dtype=tf.int32)
print(zero_vector.numpy())
# 定义两个2×2的常量矩阵
A= tf.constant([[1., 2.], [3., 4.]])
B = tf.constant([[5., 6.], [7., 8.]])
# 查看矩阵a/b的形状、类型和值
print(A.shape)  # ?乘?
print(A.dtype)  # float / int
print(B.numpy())  # 纯值输出

# basic operation
c = tf.add(A, B)  # 求和
d = tf.matmul(A, B)  # 乘积
print(c)
print(d)

# auto derivation
x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:
    # 进入了tape的上下文环境，则在该环境中计算步骤都会被自动记录
    y = tf.square(x)  # 对输入张量的每一个元素求平方，不改变张量形状。
y_grad = tape.gradient(y, x)
print(y,y_grad)
# 离开上下文环境后，记录将停止，但记录器 tape 依然可用，
# 因此可以通过 tape.gradient(y, x) 求张量 y 对变量 x 的导数


# 对多元函数求导
X = tf.constant([[1., 2.], [3., 4.]])
Y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=[[1.], [1.]])
with tf.GradientTape() as tape:
    L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - Y))
    # 对输入张量的所有元素求和，输出一个形状为空的纯量张量
w_grad, b_grad = tape.gradient(L, [w, b])  # 计算L(w, b)关于w, b的偏导数
print(L, w_grad, b_grad)
