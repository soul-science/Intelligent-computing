import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pso import Pso, draw


'''
    人工免疫粒子群算法(均方误差——聚合适应度)
'''


class AiPso(Pso):
    def __init__(self, f, i, extent, k=0.8, lr=0.01, c1=0.5, c2=0.5, vt=15, vm=-15, al=50):
        super().__init__(f, i, extent, lr, c1, c2, vt, vm, al)
        self.k = k

    def fitness(self, c):
        return super().evaluate() * np.exp(-c)

    def single_concentration(self, single_x):
        s = 0
        for x in self.al_x:
            if np.any(x != single_x):
                if 1/(1+np.square(x - single_x).mean()) > self.k:
                    s += 1

        return s / (self.al_x.shape[0] - 1)

    def concentration(self):
        return np.apply_along_axis(self.single_concentration, 1, self.al_x)

    def fit(self, for_=1000):
        self.for_ = for_
        self.fit_ = np.array([])
        self.w = lambda w: (1 + 1 / for_ * 2) ** (-w)

        for i in range(for_):
            fitness = self.fitness(self.concentration())
            self.iterate(fitness, i)
            self.fit_ = np.append(self.fit_, self.best_e)


if __name__ == '__main__':
    import time
    t = time.time()
    f1 = lambda x: x * np.sin(x) + x*np.cos(2*x)
    # f1 = lambda x: np.sum(x**2/4000) - np.cos(x/np.sqrt(np.arange(1, x.shape[0]+1))).prod() + 1
    # f2 = lambda x, y: (x**2 + y**2)/4000 - np.cos(x/np.sqrt(1)) * np.cos(y/np.sqrt(2)) + 1
    # f1 = lambda x: -(x**(1/x))
    pso = Pso(f1, 1, [-10, 10])
    pso.fit(100)
    print(time.time() - t)
    x, y = pso.result()
    print(x, y)
    draw(f1, [-10, 10], [x, y], 1000)
    # draw_3d(f2, [-8, 8], [x, y], 160)

    pso.draw_fit()
