"""
    Module: ACO
    Author: ShaoHaozhou
    motto: Self-discipline, self-improvement, self-love
    Date: 2021/5/17
    Introduce:
    介绍:
        [1] 蚁群算法
            args: start, end, others, rect, c, v, n
                start: 起点
                end: 终点
                others: 途径点
                c: 信息素挥发度 (0, 1)
                alpha: 信息素启发度 (0, 5]
                beta: 距离启发度 [0, 5]
                q: 信息增量 [10, 9999]
                n: 蚂蚁数量 n > 0
            step1: 首先初始化距离矩阵 rect 和 信息素矩阵 messages
            for:
                step2: 对第i个蚂蚁 ant(i) 进行路径选择(采用轮盘赌方式); 每选择一个后信息素进行一个实时的更新
                step3: 保存当前最优路径, 并进行信息素的更新
                    {
                    选择时的信息素矩阵:
                        p(i, j) =  τ(i, j)^α * η(i, j)^λ
                        η(i, j) = 1 / L(i, j)
                        # L(i, j) 为 i -> j 的距离
                    信息素实时更新:
                        τ(i, j) = (1 - c)*τ(i, j) + c*q
                    信息素整体路径更新:
                        Δτ = Δτ + q/d(i,j)]
                        τ(i, j) = (1 - c)*τ(i, j) + c*Δτ
                    }

"""
import numpy as np
import matplotlib.pyplot as plt


class ACO(object):
    def __init__(self, start, end, points, c=0.1, alpha=1, beta=3, q=100, n=50):
        self.start = start
        self.end = end
        self.points = points
        self.c = c
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.q = q

        self.length = len(points)
        self.fp = lambda x, y: (1 - self.c) * x + self.c * y
        self.distance = lambda x, y: np.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

        self.ant = None
        self.messages = np.ones((self.length, self.length))
        self.rect = self.initialize(points)
        self.v = 1 / self.rect

        self.best = None
        self.fit_ = None
        self.for_ = None

    def initialize(self, points):
        """
            初始化距离邻接矩阵 L(i, j)
        """
        rect = np.zeros((self.length, self.length))
        for i in range(self.length):
            for j in range(self.length):
                rect[i][j] = self.distance(points[i], points[j])
        return rect

    def calculate_distance(self, s):
        """
            算出距离向量 d
        """
        d = []

        for i in range(self.length - 1):
            d.append(self.rect[s[i], s[i+1]] if d == [] else d[i-1] + self.rect[s[i], s[i+1]])

        return d

    def select(self, i):
        """
            蚂蚁路径选择函数, 用来选择蚂蚁下一次走的位置
        """
        pos = self.ant[i][-1]
        taboo = self.ant[i]
        p = []  # 信息素矩阵
        q = []  # 距离倒数矩阵
        for j in range(self.length):
            if j not in taboo and j != self.end:
                p.append([j, self.messages[pos][j]])
                q.append(self.v[pos][j])

        p, q = [np.array(x) for x in [p, q]]

        p[:, 1] = (p[:, 1] ** self.alpha) * (q ** self.beta)

        p[:, 1] = p[:, 1] / sum(p[:, 1])

        # 轮盘赌
        s = 0
        for j in range(len(p) - 1, -1, -1):
            a = p[j, 1]
            p[j, 1] = 1 - s
            s += a

        rand = np.random.rand()

        for j in range(len(p)):
            if rand < p[j, 1]:
                self.ant[i].append(int(p[j, 0]))
                self.messages[pos, int(p[j, 0])] = self.fp(self.messages[pos, int(p[j, 0])], self.q)
                break

    def iterate(self):
        """
            一次循环函数(包含选择、取优，更新)
        """
        s = []
        for i in range(self.n):
            for j in range(self.length - 2):
                self.select(i)
        for i in range(self.n):
            self.ant[i].append(self.end)
            s.append(self.calculate_distance(self.ant[i]))

        index = np.argmin(np.array(s)[:, -1])

        if self.best is None or self.best[0] > s[index][-1]:
            self.best = [s[index][-1], self.ant[index]]

        middle = np.zeros((self.length, self.length))

        for i in range(self.n):
            for j in range(self.length - 1):
                middle[self.ant[i][j], self.ant[i][j+1]] = \
                    middle[self.ant[i][j], self.ant[i][j+1]] + self.q / s[i][j]

        self.messages = self.fp(self.messages, middle)

    def fit(self, for_):
        """
            启动函数
        """
        self.fit_ = np.array([])
        self.for_ = for_
        for i in range(for_):
            print(i)
            self.ant = [[self.start] for _ in range(self.n)]
            self.iterate()
            self.fit_ = np.append(self.fit_, self.best[0])

    def draw_fit(self):
        """
            适应度折线绘图函数
        """
        if self.fit_ is None:
            raise ValueError("you should fit firstly!")
        x = np.linspace(0, self.for_, self.for_)
        y = self.fit_
        plt.plot(x, y)
        plt.show()


if __name__ == '__main__':
    points = open("./points.txt", 'r', encoding="utf-8", ).read().splitlines()
    points = [list(map(int, each.strip().split(" "))) for each in points]

    aco = ACO(start=28, end=30, points=points, n=50)
    aco.fit(200)
    print(aco.best)
    aco.draw_fit()
    a = [28, 0, 14, 13, 11, 12, 10, 22, 15, 4, 5, 6, 1, 3, 7, 8, 9, 2, 17, 16, 18, 23, 24, 19, 20, 21, 25, 27, 26, 29,
         30]
    print(aco.calculate_distance(a)[-1], a)
