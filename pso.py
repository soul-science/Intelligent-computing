import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Pso(object):
    def __init__(self, f, i, extent, lr=0.01, c1=0.5, c2=0.5, vt=15, vm=-15, al=50):
        self.f = f
        self.extent = extent
        self.w = None
        self.lr = lr
        self.al_x = extent[0] + np.random.rand(al, i) * (extent[1] - extent[0])
        self.al_v = np.random.randn(al, i)
        self.al_best_e = np.zeros(al)
        self.al_best_x = np.zeros((al, i))
        self.best_e = 0
        self.best_x = np.zeros((1, i))
        self.v = [vm, vt]
        self.c = [c1, c2]

        self.for_ = None
        self.fit_ = None

    def evaluate(self):
        return np.apply_along_axis(self.f, 1, self.al_x)  # axis=0 对每列进行函数运算; axis=1 对每行进行函数运算

    def iterate(self, evaluations, batch):
        if batch != 0:
            for i in range(evaluations.shape[0]):
                if evaluations[i] < self.al_best_e[i]:
                    self.al_best_e[i] = evaluations[i]
                    self.al_best_x[i] = self.al_x[i]

            dm = np.argmin(self.al_best_e)
            # print(self.al_best_x, self.al_best_e)
            if self.best_e > self.al_best_e[dm]:
                self.best_e = self.al_best_e[dm]
                self.best_x = self.al_best_x[dm]

        else:
            self.al_best_e = evaluations
            self.al_best_x = np.copy(self.al_x)
            dm = np.argmin(self.al_best_e)
            self.best_e = self.al_best_e[dm]
            self.best_x = self.al_best_x[dm]

        for i in range(evaluations.shape[0]):
            self.al_v[i] = self.w(batch)*self.al_v[i] + np.random.ranf() * self.c[0] * (self.al_best_x[i] - self.al_x[i]) \
                           + self.c[1] * np.random.ranf() * (self.best_x - self.al_x[i])

            self.al_v[i][self.al_v[i] < self.v[0]] = self.v[0]
            self.al_v[i][self.al_v[i] > self.v[1]] = self.v[1]

            self.al_x[i] = self.al_x[i] + self.al_v[i]

            self.al_x[i][self.al_x[i] < self.extent[0]] = self.extent[0]
            self.al_x[i][self.al_x[i] > self.extent[1]] = self.extent[1]

    def fit(self, for_=1000):
        self.fit_ = np.array([])
        self.for_ = for_
        self.w = lambda w: (1 + 1 / for_ * 2) ** (-w)

        for i in range(for_):
            evaluations = self.evaluate()
            self.iterate(evaluations, i)
            self.fit_ = np.append(self.fit_, self.best_e)

    def result(self):
        return [self.best_x, self.best_e]

    def draw_w(self):
        if self.w is None:
            raise ValueError("w is None, you should fit firstly!")
        x = np.linspace(0, self.for_, self.for_*10)
        y = self.w(x)
        plt.plot(x, y)
        plt.show()

    def draw_fit(self):
        if self.fit_ is None:
            raise ValueError("you should fit firstly!")
        x = np.linspace(0, self.for_, self.for_)
        y = self.fit_
        plt.plot(x, y)
        plt.show()


def draw(f, extent, point, interval):
    x = np.linspace(extent[0], extent[1], interval)
    y = f(x)
    plt.plot(x, y)
    plt.plot(point[0], point[1], 'ro')
    plt.show()


def draw_3d(f, extent, point, interval):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = np.linspace(extent[0], extent[1], interval)
    y = np.linspace(extent[0], extent[1], interval)
    # x.shape = (1, x.shape[0])
    # y.shape = (1, y.shape[0])
    x, y = np.meshgrid(x, y)
    print(x.shape, y.shape, f(x, y).shape)
    ax.plot_surface(x, y, f(x, y), rstride=1, cstride=1)
    ax.scatter(point[0][0], point[0][1], point[1], c='r', marker='^')
    ax.view_init(elev=30, azim=125)
    plt.show()


"""
    改进：
        1. 自适应权重系数 w - f(w)
        2. 全局控制因子(收缩因子)
        3. 自适应学习因子
        4. 权重w策略(递减权重策略 自适应权重策略 随即权重策略)
        ……
"""


if __name__ == '__main__':
    import time
    t = time.time()
    f1 = lambda x: x * np.sin(x) + x*np.cos(2*x)
    # f1 = lambda x: np.sum(x**2/4000) - np.cos(x/np.sqrt(np.arange(1, x.shape[0]+1))).prod() + 1
    # f2 = lambda x, y: (x**2 + y**2)/4000 - np.cos(x/np.sqrt(1)) * np.cos(y/np.sqrt(2)) + 1
    # f1 = lambda x: x**(1/x)*(np.log2(1/x) + 1)
    # f1 = lambda x: np.exp(-2*x)
    pso = Pso(f1, 1, [-10, 10])
    pso.fit(100)
    print(time.time() - t)
    x, y = pso.result()
    print(x, y)
    draw(f1, [-10, 10], [x, y], 1000)
    # draw_3d(f2, [-8, 8], [x, y], 160)

    pso.draw_fit()

