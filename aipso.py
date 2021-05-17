"""
    Module: AiPso
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2021/4/20
    Introduce: Artificial immune particle swarm optimization (mean square error (mse) - polymerization fitness
    介绍: 人工免疫粒子群算法(均方误差——聚合适应度)
"""
import numpy as np
from pso import Pso, draw, draw_3d


class AiPso(Pso):
    def __init__(self, f, i, extent, k=0.8, lr=0.01, c1=0.5, c2=0.5, vt=15, vm=-15, al=50):
        super().__init__(f, i, extent, lr, c1, c2, vt, vm, al)
        self.k = k
        self.al_best_e = np.zeros((al, 2))
        self.best_e = np.zeros(2)
        self.iterate = self.to_eval  # 猴子补丁

    def fitness(self, evaluation, c):
        """
            聚合适应度
        """
        return evaluation * np.exp(-c)

    def single_concentration(self, single_x):
        """
            单个粒子与其余粒子的相似度
        """
        s = 0
        for x in self.al_x:
            if np.any(x != single_x):
                if np.all(1/(1+np.square(x - single_x.mean())) > self.k):  # 均方误差
                    s += 1

        return s / (self.al_x.shape[0] - 1)

    def concentration(self):
        """
            整个粒子的适应度
        """
        return np.apply_along_axis(self.single_concentration, 1, self.al_x)

    def to_eval(self, evaluations, fitness, batch):
        """
            用来覆盖pso算法的evaluate函数(猴子补丁)
        """
        if batch != 0:
            for i in range(evaluations.shape[0]):
                if evaluations[i] < self.al_best_e[i, 0]:
                    self.al_best_e[i] = [evaluations[i], fitness[i]]
                    self.al_best_x[i] = self.al_x[i]

            dm = np.argmin(self.al_best_e[:, 0])
            if self.best_e[0] > self.al_best_e[dm, 0]:
                self.best_e = self.al_best_e[dm]
                self.best_x = self.al_best_x[dm]

        else:
            self.al_best_e = np.concatenate((evaluations, fitness), axis=1)
            self.al_best_x = np.copy(self.al_x)
            dm = np.argmin(self.al_best_e[:, 0])
            self.best_e = self.al_best_e[dm]
            self.best_x = self.al_best_x[dm]

        total = sum(self.al_best_e[:, 1]) + self.best_e[1]
        fit_x = np.vstack((self.al_best_x, self.best_x))
        fit_e = np.vstack(((self.al_best_e[:, 1] / total).reshape(self.al_x.shape[0], 1), [self.best_e[1]/total]))

        s = 0
        for i in range(fit_e.shape[0] - 1, -1, -1):
            a = fit_e[i]
            fit_e[i] = 1 - s
            s += a

        rands = np.random.rand(self.al_best_e.shape[0])
        out = []

        for rand in rands:
            for i in range(fit_e.shape[0]):
                if rand < fit_e[i]:
                    out.append(i)

        for i in range(evaluations.shape[0]):
            self.al_v[i] = self.w(batch)*self.al_v[i] + np.random.ranf() * self.c[0] * (self.al_best_x[i] - self.al_x[i]) \
                           + self.c[1] * np.random.ranf() * (fit_x[out[i]] - self.al_x[i])

            self.al_v[i][self.al_v[i] < self.v[0]] = self.v[0]
            self.al_v[i][self.al_v[i] > self.v[1]] = self.v[1]

            self.al_x[i] = self.al_x[i] + self.al_v[i]

            self.al_x[i][self.al_x[i] < self.extent[0]] = self.extent[0]
            self.al_x[i][self.al_x[i] > self.extent[1]] = self.extent[1]

    def fit(self, for_=1000):
        """
            启动函数
        """
        self.for_ = for_
        self.fit_ = np.array([])
        self.w = lambda w: (1 + 1 / for_ * 2) ** (-w)

        for i in range(for_):
            evaluation = super().evaluate()
            c = self.concentration()
            evaluation.resize(evaluation.shape[0], 1)
            c.resize(c.shape[0], 1)
            fitness = self.fitness(evaluation, c)
            self.iterate(evaluation, fitness, i)
            self.fit_ = np.append(self.fit_, self.best_e[0])


if __name__ == '__main__':
    import time
    t = time.time()
    f1 = lambda x: x * np.sin(x) + x*np.cos(2*x)
    pso = AiPso(f1, 1, [-10, 10], k=0.9)
    pso.fit(100)
    print(time.time() - t)
    x, y = pso.result()
    print(x, y[0])
    pso.draw_fit()
    draw(f1, [-10, 10], [x, y[0]], 1000)



