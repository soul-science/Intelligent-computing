"""
    Module: GA
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2021/4/20
    Introduce: This is the Genetic Algorithm which contains discrete and continuous two types [DGA and CGA]
    介绍: 这是一种包含离散和连续两种类型的遗传算法。 GA = {DGA, CGA}
"""

import numpy as np
import matplotlib.pyplot as plt


class GABase(object):
    """
        Introduce:
            This is the base class of genetic algorithms
        介绍：
            这是遗传算法的基类(GABase)

        arg:
            f: the objective function
            sv: coefficient of Selection
            cv: coefficient of variation
        参数:
            f: 目标函数
            sv: 选择系数
            cv: 变异系数
    """
    def __init__(self, f, sv, cv):
        self.f = f
        self.sv = sv
        self.cv = cv
        self.pop = None  # 种群
        self.best = [None, np.inf]  # 最优个体及其适应度(目标函数值)
        self.fit_ = None   # 每轮最优适应度的记录(作图用)
        self.for_ = None   # 轮数

    def initialize(self):
        """
            This is the initialization function.
            初始化函数
        """
        pass

    def fitness(self):
        """
            # 用户可通过此函数来自定义目标函数...
                func(a_singe_pop) => return the fitness
        """
        return np.apply_along_axis(self.f, 1, self.pop)

    def select(self, fitness):
        """
            This is the selection function.Through the roulette algorithm to eliminate
             the number of times for the current population is 1/10.
            选择函数，通过轮盘赌算法进行淘汰，淘汰次数为当前种群的1/10。
        """
        min_pos = np.argmin(fitness)
        if fitness[min_pos] < self.best[1]:
            self.best = [self.pop[min_pos], fitness[min_pos]]

        min_fitness = fitness[min_pos]

        fitness = fitness - min_fitness
        u = np.sum(fitness)

        fitness = fitness / u

        s = 0
        for i in range(fitness.shape[0] - 1, -1, -1):
            a = fitness[i]
            fitness[i] = 1 - s
            s += a

        rands = np.random.rand(self.pop.shape[0] // 10)
        out = set()
        for rand in rands:
            for i in range(fitness.shape[0]):
                if rand < fitness[i]:
                    out.add(i)
                    break
        self.pop = np.delete(self.pop, list(out), axis=0)

    def cross(self):
        """
            This is cross function.
            交叉函数
        """
        pass

    def mutate(self):
        """
            This is mutation function.
            变异函数
        """
        pass

    def fit(self, for_=1000):
        """
            This is opening function to start the train of GA.
            启动函数
        """
        self.fit_ = np.array([])
        self.for_ = for_
        self.best = [None, np.inf]
        self.initialize()
        for _ in range(for_):
            self.select(fitness=self.fitness())
            self.cross()
            self.mutate()
            self.fit_ = np.append(self.fit_, self.best[1])

    def draw_fit(self):
        """
            This is drawing function to draw the fitness line chart.
            画图函数 => 画出适应度折线图
        """
        if self.fit_ is None:
            raise ValueError("you should fit firstly!")
        x = np.linspace(0, self.for_, self.for_)
        y = self.fit_
        plt.plot(x, y)
        plt.show()


class DGA(GABase):
    """
        Introduce:
            This is a discrete type of genetic algorithm
        介绍：
            这是离散类型的遗传算法(DGA)

        arg:
            f: the objective function
            sv: coefficient of Selection
            cv: coefficient of variation
            kinds: the type of individual genes
            max_pop: the max population
            length: the number of genes contained in an individual
            can_replace: whether duplicate genes can be present in an individual
        参数:
            f: 目标函数
            sv: 选择系数
            cv: 变异系数
            kinds: 个体基因的种类
            max_pop: 最大种群数量
            length: 一个个体所包含的基因
            can_replace: 一个个体中是否可以出现重复的基因
    """
    def __init__(self, f, cv, sv, kinds, max_pop, length, can_replace=False):
        super().__init__(f, cv, sv)
        self.kinds = list(kinds)
        self.max_pop = max_pop
        self.length = length
        self.can_replace = can_replace

    def initialize(self):
        """
            There are two types of initialization: repeatable and non-repeatable. The default
             setting for initializing the population is 1/10 of the maximum population.
            分成了可重复和不可重复两种初始化方式，初始化种群这里默认设置为最大种群数的1/10。
        """
        if self.can_replace is True:
            self.pop = np.random.choice(self.kinds, [self.max_pop // 10, self.length])   # 使用整除获得整数
        else:
            self.pop = np.ones([self.max_pop // 10, self.length]) * np.inf
            for i in range(self.max_pop // 10):
                for j in range(self.length):
                    while True:
                        choice = np.random.choice(self.kinds)
                        if choice not in self.pop[i]:
                            self.pop[i][j] = choice
                            break

    def fitness(self):
        return super().fitness()

    def select(self, fitness):
        super().select(fitness)

    def cross(self):
        """
            In the process of crossover inheritance, individuals before and
             after each other are randomly selected at different locations to generate two individuals
            交叉遗传过程采取随机选取不同位置对前后的个体进行交叉遗传，生成两个个体。
        """
        if self.can_replace is True:
            for i in range(self.pop.shape[0]):
                if self.max_pop <= self.pop.shape[0]:
                    break
                if np.random.rand() < self.sv:
                    selected = np.random.randint(self.length, size=[1, np.random.randint(self.length)])
                    child = [self.pop[i].copy(), self.pop[(i+1) % self.pop.shape[0]]]
                    child[0][selected] = self.pop[(i+1) % self.pop.shape[0]][selected]
                    child[1][selected] = self.pop[i][selected]
                    self.pop = np.vstack((self.pop, child))
        else:
            for i in range(self.pop.shape[0]):
                if self.max_pop <= self.pop.shape[0]:
                    break
                if np.random.rand() < self.sv:  # 使用随机数进行判断
                    selected = np.random.randint(self.length, size=[1, np.random.randint(self.length)])
                    child = [self.pop[i].copy(), self.pop[(i+1) % self.pop.shape[0]]]
                    for j in selected[0]:
                        if self.pop[(i+1) % self.pop.shape[0]][j] not in child[0]:
                            child[0][j] = self.pop[(i+1) % self.pop.shape[0]][j]
                        if self.pop[i][j] not in child[1]:
                            child[1][j] = self.pop[i][j]

                    self.pop = np.vstack((self.pop, child))  # 对种群进行拼接

    def mutate(self):
        """
            The mutation process also uses random numbers to make changes to individual genes.
            变异过程也是采用随机数对单个的基因进行更改。
        """
        if self.can_replace:
            for i in range(self.pop.shape[0]):
                if np.random.rand() < self.cv:
                    pos = np.random.randint(self.length)
                    self.pop[i][pos] = np.random.choice(self.kinds)
        else:
            for i in range(self.pop.shape[0]):
                if np.random.rand() < self.cv:
                    pos = np.random.randint(self.length)
                    while True:
                        cross = np.random.choice(self.kinds)
                        if cross not in self.pop[i]:
                            self.pop[i][pos] = cross
                            break

    def fit(self, for_=1000):
        super().fit(for_)

    def draw_fit(self):
        super().draw_fit()


class CGA(GABase):
    """
        Introduce:
            This is a continuous type of genetic algorithm
        介绍：
            这是连续类型的遗传算法(CGA)

        arg:
            f: the objective function
            sv: coefficient of Selection
            cv: coefficient of variation
            max_pop: the max population
            length: the number of genes contained in an individual
            extent: limit the scope of a single gene
        参数:
            f: 目标函数
            sv: 选择系数
            cv: 变异系数
            max_pop: 最大种群数量
            length: 一个个体所包含的基因数量
            extent: 单个基因的范围限定
    """
    def __init__(self, f, sv, cv, max_pop, length, extent):
        super().__init__(f, sv, cv)
        self.max_pop = max_pop
        self.length = length
        self.extent = extent

    def initialize(self):
        """
            Generate a population within a range.
            生成在范围之内的种群。
        """
        self.pop = self.extent[0] + np.random.rand(self.max_pop // 10, self.length) * (self.extent[1] - self.extent[0])

    def fitness(self):
        return super().fitness()

    def select(self, fitness):
        super().select(fitness)

    def cross(self):
        for i in range(self.pop.shape[0]):
            if self.max_pop <= self.pop.shape[0]:
                break
            if np.random.rand() < self.sv:
                selected = np.random.randint(self.length, size=[1, np.random.randint(self.length)])
                child = [self.pop[i].copy(), self.pop[(i + 1) % self.pop.shape[0]]]
                child[0][selected] = self.pop[(i + 1) % self.pop.shape[0]][selected]
                child[1][selected] = self.pop[i][selected]
                self.pop = np.vstack((self.pop, child))

    def mutate(self):
        for i in range(self.pop.shape[0]):
            if np.random.rand() < self.cv:
                pos = np.random.randint(self.length)
                self.pop[i][pos] = self.extent[0] + np.random.rand() * (self.extent[1] - self.extent[0])

    def fit(self, for_=1000):
        super().fit(for_)

    def draw_fit(self):
        super().draw_fit()


class GA(object):
    mode = ["DGA", "CGA"]

    def __init__(self, f, sv, cv, max_pop, length, extent_or_kinds, can_replace=False):
        if len(list(extent_or_kinds)) != 2:
            self.mode = GA.mode[0]
            self.ga = DGA(f, cv, sv, extent_or_kinds, max_pop, length, can_replace)
            self.msg = "{mode}(f:{f}, sv:{sv}, cv:{cv}, max_pop:{max_pop}," \
                       " length:{length}, kinds:{kinds}, can_replace{can_replace}"\
                .format(mode=self.mode, f=f, sv=sv, cv=cv, max_pop=max_pop, length=length, kinds=extent_or_kinds,can_replace=can_replace)
        else:
            self.mode = GA.mode[1]
            self.ga = CGA(f, cv, sv, max_pop, length, extent_or_kinds)
            self.msg = "{mode}(f:{f}, sv:{sv}, cv:{cv}, max_pop:{max_pop}," \
                       " length:{length}, extent:{extent})"\
                .format(mode=self.mode, f=f, sv=sv, cv=cv, max_pop=max_pop, length=length, extent=extent_or_kinds)

    def __repr__(self):
        return self.msg

    def __str__(self):
        return self.__repr__()

    def fit(self, for_=1000):
        self.ga.fit(for_)

    def result(self):
        return self.ga.best

    def draw_fit(self):
        self.ga.draw_fit()


if __name__ == "__main__":
    f = lambda x: sum(x * np.sin(x) + x * np.cos(2 * x))

    # 测试 DGA
    dga = DGA(f, cv=0.1, sv=0.01, kinds=np.linspace(0, 10, 100), max_pop=100, length=5)
    dga.fit(1000)
    dga.draw_fit()
    print(dga.best)

    # 测试 CGA
    cga = CGA(f, cv=0.1, sv=0.1, max_pop=100, length=5, extent=[0, 10])
    cga.fit(1000)
    cga.draw_fit()
    print(cga.best)

    # 测试 GA
    ga = GA(f, cv=0.1, sv=0.1, max_pop=100, length=5, extent_or_kinds=[0, 10])
    ga.fit(1000)
    ga.draw_fit()
    print(ga.result())
