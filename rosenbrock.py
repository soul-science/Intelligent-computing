"""
    TODO: test  f(x, y) = 100*(x**2 - y)**2 + (1 - y)**2
"""
from aipso import AiPso
from pso import Pso, draw_3d
from GA import GA
from SA import KSA

import time


if __name__ == '__main__':
    fx = lambda x: -(100*(x[0]**2 - x[1])**2 + (1 - x[1])**2)
    fxy = lambda x, y: 100*(x**2 - y)**2 + (1 - y)**2
    msg = "{method}: time: {time}, x: {x}, y: {y}"

    # Pso算法
    t = time.time()
    pso = Pso(fx, 2, [-2.048, 2.048])
    pso.fit(100)
    x, y = pso.result()
    print(msg.format(method="Pso", time=time.time() - t, x=x, y=-y))
    draw_3d(fxy, [-2.048, 2.048], [x, -y], 200)
    pso.draw_fit()

    # AiPso算法
    t = time.time()
    aipso = AiPso(fx, 2, [-2.048, 2.048])
    aipso.fit(100)
    x, y = aipso.result()
    print(msg.format(method="AiPso", time=time.time() - t, x=x, y=-y[0]))
    draw_3d(fxy, [-2.048, 2.048], [x, -y[0]], 200)
    aipso.draw_fit()

    # GA遗传算法
    t = time.time()
    ga = GA(fx, cv=0.1, sv=0.1, max_pop=100, length=2, extent_or_kinds=[-2.048, 2.048])
    ga.fit(1000)
    x, y = ga.result()
    print(msg.format(method="GA", time=time.time() - t, x=x, y=-y))
    draw_3d(fxy, [-2.048, 2.048], [x, -y], 200)
    ga.draw_fit()

    # KSA群体模拟退火
    t = time.time()
    ksa = KSA(fx, 204, 2, 0.1, 1, [-2.048, 2.048])
    ksa.fit(100)
    x, y = ksa.result()
    print(msg.format(method="KSA", time=time.time() - t, x=x, y=-y))
    draw_3d(fxy, [-2.048, 2.048], [x, -y], 200)
    ksa.draw_fit()
