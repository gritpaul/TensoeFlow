import tensorflow as tf


# tf2.0默认打开即时执行模式，如果需要关闭， 那么执行
# tf.compat.v1.disable_eager_execution()
# 关闭tf2的即时执行模式

# tensorflow Tensor concept
def intro_tensor():
    # 定义一个随机数（标量）
    random_float = tf.random.uniform(shape=())  # 缺省shape值为空 纯量张量
    print(random_float)

    # 定义一个有3行2列的零向量,定义整型值
    zero_vector = tf.zeros(shape=(3, 2), dtype=tf.int32)
    # 如果不用'dtype'定义元素类型，那么tf将默认定义元素为tf.float32
    print(zero_vector.numpy())

    # 定义两个2×2的常量矩阵
    a = tf.constant([[1., 2.], [3., 4.]])
    b = tf.constant([[5., 6.], [7., 8.]])
    # 查看矩阵a/b的形状、类型和值
    print(a.shape)  # 输出矩阵的长和宽
    print(a.dtype)  # 输出float / int
    print(b.numpy())  # 输出值
    # Tensor同样具有形状、类型和值三种属性。

# basic operation
def intro_operation():
    # 定义两个2×2的常量矩阵
    a = tf.constant([[1., 2.], [3., 4.]])
    b = tf.constant([[5., 6.], [7., 8.]])

    c = tf.add(a, b)  # 求和
    d = tf.matmul(a, b)  # 乘积
    print(c.numpy())
    print(d.numpy())

# auto derivation
def intro_autode():
    x = tf.Variable(initial_value=3.)
    # 与普通tensor一样， 变量同样具有形状、类型和值三种属性。
    with tf.GradientTape() as tape:
        # tf.GradientTape 是 自动求导记录器
        # 进入了tape的上下文环境，则在该环境中计算步骤都会被自动记录
        y = tf.square(x)  # 对输入张量的每一个元素求平方，不改变张量形状。
    # 离开上下文环境后，记录将停止，但记录器 tape 依然可用，
    # 因此可以通过 tape.gradient(y, x) 求张量 y 对变量 x 的导数
    y_grad = tape.gradient(y, x)
    print('y =', y.numpy(), '\ny_grad =', y_grad.numpy())

# 对多元函数求导
def intro_mul_autode():
    x = tf.constant([[1., 2.], [3., 4.]])
    y = tf.constant([[1.], [2.]])
    w = tf.Variable(initial_value=[[1.], [2.]])
    b = tf.Variable(initial_value=[[1.], [1.]])
    with tf.GradientTape() as tape:
        L = tf.reduce_sum(tf.square(tf.matmul(x, w) + b - y))
        # tf.reduce_sum: 对输入张量的所有元素求和，输出一个形状为空的纯量张量
        # tf.square: 对输入张量的每个元素求平方，不改变张量形状
    w_grad, b_grad = tape.gradient(L, [w, b])
    # 计算L(w, b)关于w, b的偏导数
    print( L, '\n', w_grad, '\n', b_grad )


# -------------------------------------------------------------
# 运行某一个函数的代码
# 1. 请修改下面的函数名
# 2. 然后点击PyCharm的运行按钮
# -------------------------------------------------------------

intro_mul_autode()
