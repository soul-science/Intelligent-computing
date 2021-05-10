"""
    Module: SA
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2021/5/10
    Introduce:
    介绍:
        [1] 模拟退火算法
            args: f, n, field, t, extent
                f: 函数
                n: 解的维数
                field: 领域<为数字>
                t: 控制函数
                extent: 搜索范围

        [2] 优化: 群体模拟退火算法
            args: f, k, n, field, t, extent
                f, n, field, t, extent: 同上
                k: 种群个数
            step1: 初始化k<将范围化成k+1个区域>个x(i, 0), 并且算出f(i, 0)
            for:
                step2: 在x(i, j)的邻域范围内生成x(i, j+1), 并算出Δf = f(i, j+1) - f(i, j),
                    并且根据 min{1, exp(Δf/t)} > rand(0, 1) 去接受 x(i, j+1) <循环>,然后在
                    这k个x(i, j+1)选出最优的f并且与f_best比较,优于则保存。
                step3: 然后进行退火操作 -> t(i+1) = c*t(i)
                        [c: 线性 1 -> 0 or 指数函数 1 -> 0] <本程序采用第二种>
                step4: 进行一个种群的淘汰优化, 删除f(max)所在的个体 <让速度进行一个优化>


            TODO(可考虑):
                1. 加入遗传算法
                2. 混合粒子群算法
                3. 采取自适应控制因子
                    ...

"""
import numpy as np
import matplotlib.pyplot as plt
# import pso


class SA(object):

    def __init__(self, f: object, n: int, field: float or int, t: float or int, extent: list):
        self.f = f
        self.n = n
        self.field = field
        self.t = t
        self.extent = extent
        self.best_y = None
        self.best_x = None
        self.c = None

        self.for_ = None
        self.fit_ = None

    def __repr__(self) -> str:
        return "SA(f:{f}, field:{field}, t:{t}, extent:{extent})".format(
            f=self.f, field=self.field, t=self.t, extent=self.extent)

    def initialize(self, low, high, k):
        """
        :param low:
            左界限
        :param high:
            右界限
        :param k:
            种群数
        :return:
            x(i, 0)
        """
        return low + np.random.rand(k, self.n) * (high - low)

    def iterate(self, x, c):
        """
        :param x:
            x(i, j)
        :param c:
            控制参数c
        :return:
            x(i, j+1), y(i, j+1)
        """
        low = x - self.field
        high = x + self.field

        low[low < self.extent[0]] = self.extent[0]
        high[high > self.extent[1]] = self.extent[1]

        while True:
            xx = self.initialize(low, high, 1)

            fxx = np.apply_along_axis(self.f, 1, xx)
            fx = np.apply_along_axis(self.f, 1, x)

            if min(np.exp(-(fxx - fx) / c), 1) > np.random.rand():
                return xx, fxx

    def fit(self, for_):
        """
        :param for_:
            循环次数
        :return:
            None
        """
        self.fit_ = np.array([])
        self.for_ = for_
        self.c = lambda c: (self.t + 1 / for_ * 2) ** (-c)

        x = self.initialize(np.array([self.extent[0]]*self.n), np.array([self.extent[1]]*self.n), 1)
        self.best_x, self.best_y = x, np.apply_along_axis(self.f, 1, x)

        for i in range(for_):
            x, y = self.iterate(x, self.c(i))

            self.best_x, self.best_y = (x, y) if self.best_y > y else (self.best_x, self.best_y)
            self.fit_ = np.append(self.fit_, self.best_y)

    def result(self):
        """
        :return:
            x, y <best>
        """
        return self.best_x, self.best_y

    def draw_fit(self):
        if self.fit_ is None:
            raise ValueError("you should fit firstly!")
        x = np.linspace(0, self.for_, self.for_)
        y = self.fit_
        plt.plot(x, y)
        plt.show()


class KSA(SA):

    def __init__(self, f: object, k: int, n: int, field: float or int, t: float or int, extent: list):
        super().__init__(f, n, field, t, extent)
        self.k = k
        self.extents = [extent[0] + (extent[1] - extent[0]) / k * i for i in range(k+1)]

    def __repr__(self) -> str:
        return "KSA(f:{f}, k:{k}, field:{field}, t:{t}, extent:{extent})".format(
            f=self.f, k=self.k, field=self.field, t=self.t, extent=self.extent)

    def iterate(self, x, c):

        low = x - self.field
        high = x + self.field

        low[low < self.extent[0]] = self.extent[0]
        high[high > self.extent[1]] = self.extent[1]
        xx = []
        fxx = []
        fx = np.apply_along_axis(self.f, 1, x)
        # print(x)

        for i in range(x.shape[0]):
            while True:

                xx_ = self.initialize(low[i], high[i], 1)

                fxx_ = np.apply_along_axis(self.f, 1, xx_)
                # print(fxx_, fx[i])
                if min(np.exp(-(fxx_ - fx[i]) / c), 1) > np.random.rand():
                    xx.extend(xx_)
                    fxx.extend(fxx_)
                    break

        return np.array(xx), np.array(fxx)

    def initialize(self, low, high, k):
        return low + np.random.rand(k, self.n) * (high - low)

    def fit(self, for_):
        self.fit_ = np.array([])
        self.for_ = for_
        self.c = lambda c: (self.t + 1 / for_ * 2) ** (-c)

        x = self.initialize(
            np.array([self.extents[i] for i in range(self.k)]).reshape(self.k, 1),
            np.array([self.extents[i] for i in range(1, self.k+1)]).reshape(self.k, 1), self.k)
        y = np.apply_along_axis(self.f, 1, x)
        index = np.argmin(y)

        self.best_x, self.best_y = x[index], y[index]

        for i in range(for_):
            x, y = self.iterate(x, self.c(i))
            index = np.argmin(y)
            self.best_x, self.best_y = (x[index], y[index]) if self.best_y > y[index] else (self.best_x, self.best_y)

            if x.shape[0] > 1:
                index = np.argmax(y)
                x = np.delete(x, index, axis=0)

            self.fit_ = np.append(self.fit_, self.best_y)


def draw(f, extent, point, interval):
    x = np.linspace(extent[0], extent[1], interval)
    y = f(x)
    plt.plot(x, y)
    plt.plot(point[0], point[1], 'ro')
    plt.show()


if __name__ == '__main__':
    import time

    t = time.time()
    # f1 = lambda x: np.cos(x)
    f1 = lambda x: x * np.sin(x) + x * np.cos(2 * x)
    # f1 = lambda x: np.sum(x ** 2 / 4000) - np.cos(x / np.sqrt(np.arange(1, x.shape[0] + 1))).prod() + 1
    # f2 = lambda x, y: (x ** 2 + y ** 2) / 4000 - np.cos(x / np.sqrt(1)) * np.cos(y / np.sqrt(2)) + 1
    sa = KSA(f1, 50, 1, 0.1, 1, [-10, 10])
    sa.fit(100)
    print(time.time() - t)
    x, y = sa.result()
    print(x, y)

    draw(f1, [-10, 10], [x, y], 100)
    sa.draw_fit()
    # pso.draw_3d(f2, [-8, 8], [x, y], 160)
