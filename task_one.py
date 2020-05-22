import numpy as np
import matplotlib.pyplot as plt

'''
 16 вариант - Цимбалов - 29 метод
'''
c2, A, B, C = [0.85, 2, -2, 3]


def f(x, y):
    return np.array([2 * x * y[1] ** (1 / B) * y[3],
                     2 * B * x * np.exp(B / C * (y[2] - A)) * y[3],
                     2 * C * x * y[3],
                     -2 * x * np.log(y[0])
                     ])


def exact_ans(x):
    return np.array([np.exp(np.sin(x ** 2)),
                     np.exp(B * np.sin(x ** 2)),
                     C * np.sin(x ** 2) + A,
                     np.cos(x ** 2)])


# Второй метод Рунге - Кутта 3 порядка
def main_method(f, x0, y0, h):
    k1 = f(x0, y0)
    k2 = f(x0 + 1 / 2 * h, y0 + 1 / 2 * h * k1)
    k3 = f(x0 + h, y0 - h * k1 + 2 * h * k2)
    return y0 + h * (1 / 6 * k1 + 4 / 6 * k2 + 1 / 6 * k3)


def solve(h, f, x0, y0, x_end, n):
    if h == -1:
        h = (x_end - x0) / n
    else:
        n = (x_end - x0) / h
        n = int(n)

    x_arr, y_arr = [x0], [y0]

    for i in range(n):
        x_new = x_arr[i] + h
        x_arr.append(x_new)
        y_new = main_method(f, x_arr[i], y_arr[i], h)
        y_arr.append(y_new)

    return np.array(x_arr), np.array(y_arr)


par = {'x0': 0, 'y0': np.array([1, 1, A, 1]), 'x_end': 5, 'n': 200}

if __name__ == "__main__":

    RK_X, RK_Y = solve(-1, f, **par)

    opponent_X, opponent_Y = solve(-1, f, **par)

    for i in range(4):
        plt.plot(RK_X, exact_ans(RK_X)[i], label='exact $y_{%s}(x)$' % (i + 1))
        plt.plot(RK_X, RK_Y[:, i], label='approx $y_{%s}(x)$' % (i + 1))
        plt.plot(opponent_X, opponent_Y[:, i], label='opponent $y_{%s}(x)$' % (i + 1))
        plt.legend()
        plt.show()
