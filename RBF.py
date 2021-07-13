"""
    Module: RBF
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2021/7/13
    Introduce: Regularization gaussian radial basis neural network
    介绍: 正则化高斯径向基神经网络(输入层-径向基层(隐藏层)-输出层)
    TODO:
        step1: RBF的数学原理
            1) 径向基函数的意义
            2) 权重初始化方法
            3) RBF中的初始化方法
            4) 反向传播(链式法则, 自动微分)
        step2: 编写程序
            1) 初始化 W(wx + b): fan-in, fan-out, caffe
            2) 初始化 C:
                1. min X + (max X - min X)/2p + (j - 1)(max X - min X)/p
                2. Kmeans聚类
                3. 正态随机分布
            3) 初始化 D:
                1. df*√[(1/n)*的∑(xi - cji)^2]
                2. 正态随机分布
                3. max
            for:
                4) 径向基核 RBF(Z): e^(-||(X - Cj)/Dj||^2)
                5) 输出层 Y: W * Z
                6) 损失函数 E: 1/2*||Y - O||^2
                7) 反向传播
                    W(t) = W(t-1) - α*əE(t-1)/əw(t-1) + β*w(t-1)
                    C(t) = C(t-1) - α*əE(t-1)/əC(t-1) + β*C(t-1)
                    D(t) = D(t-1) - α*əE(t-1)/əD(t-1) + β*D(t-1)
        step3: 测试
"""

import numpy as np
from functools import partial


class RBF(object):
    """
        args:
            alpha: 学习率
            delta: 正则率
            means: C的个数[隐藏层的神经元个数]
            dim: 维度
    """
    def __init__(self, alpha=0.01, delta=0.001, means=3):
        self.alpha = alpha
        self.delta = delta
        self.means = means
        self.w = None
        self.c = np.random.randn(means, 1)
        self.d = np.random.randn(means, 1)
        self.dim = None

    def rbf(self, x, c, d):
        """
            径向基函数
        """
        return np.exp(-np.power((np.sum(x - c, axis=1) / d), 2))

    def initialize(self):
        """
            初始化w
        """
        self.w = np.random.randn(self.means + 1, self.dim[0])

    def __forward(self, train_x):
        """
            向前传播
        """
        p = partial(self.rbf, train_x)
        z = np.array([p(c=self.c[i], d=self.d[i]) for i in range(self.c.shape[0])])
        z = np.c_[z.T, np.ones((train_x.shape[0], 1))]
        y = np.dot(z, self.w)
        return z, y

    def forward(self, train_x, train_y):
        """
            向前传播
        """
        z, y = self.__forward(train_x)
        e = np.sum(np.power(y - train_y, 2)) / (2 * train_y.shape[0])
        return z, y, e

    def backward(self, z, y, train_x, train_y):
        """
            向后传播
        """
        self.c = np.array([self.c[i]
                           - self.alpha * (1 / (train_y.shape[0] * np.power(self.d[i], 2))
                                           * np.dot(np.dot(z[i], np.dot(y - train_y, self.w.T).T), train_x - self.c[i]))
                           + self.delta * self.c[i]
                           for i in range(self.c.shape[0])])
        self.d = np.array([self.d[i]
                           - self.alpha * (1 / (train_y.shape[0] * self.d[i])
                                           * np.dot(np.dot(z[i], np.dot(y - train_y, self.w.T).T), z[:, i]))
                           + self.delta * self.d[i]
                           for i in range(self.d.shape[0])])

        self.w -= self.alpha * (1 / train_y.shape[0] * np.dot(z.T, y - train_y)) + self.delta * self.w

    def fit(self, train_x, train_y, repeat=1000):
        """
            训练函数(train)
        """
        self.dim = (train_x.shape[1], train_y.shape[1])
        self.initialize()

        for _ in range(1, repeat+1):
            z, y, e = self.forward(train_x, train_y)
            self.backward(z, y, train_x, train_y)
            print("iteration[{i}]: loss({loss})".format(i=_, loss=e))

    def predict(self, test_x):
        """
            预测函数
        """
        _, test_y = self.__forward(test_x)
        return test_y

    def score(self, test_x, test_y):
        """
            准确率函数
        """
        y = self.predict(test_x)
        return sum(y == test_y) / y.shape[0]


if __name__ == '__main__':
    # 拟合Hermit多项式
    X = np.linspace(-5, 5, 500)
    X = X.reshape(X.shape[0], -1)
    y = np.multiply(1.1 * (1 - X + 2 * X ** 2), np.exp(-0.5 * X ** 2))
    rbf = RBF(means=50)
    rbf.fit(X, y, 1000)
    rbf.predict(X)
    import matplotlib.pyplot as plt

    plt.plot(X, y, 'red')
    plt.show()
    plt.plot(X, y, "blue")
    plt.show()
