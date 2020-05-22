from task_one import *


def Hoin_method(f, x0, y0, h):
    return y0 + h / 2 * (f(x0, y0) + f(x0 + h, y0 + h * f(x0, y0)))


def get_first_step(p, tol, f, x0, y0, x_end):
    delta1 = np.power(1 / max(np.abs(x0), np.abs(x_end)), p + 1) + \
             np.linalg.norm(f(x0, y0)) ** (p + 1)

    h1 = np.power(tol / delta1, 1 / (p + 1))
    u1 = Hoin_method(f, x0, y0, h1)

    delta2 = np.power(1 / max(np.abs(x0 + h1), np.abs(x_end)), p + 1) + \
             np.linalg.norm(f(x0 + h1, y0 + u1)) ** (p + 1)

    h2 = np.power(tol / delta2, 1 / (p + 1))

    return min(h1, h2)


def solve_optimal(p, rtol, atol, f, x0, y0, x_end, n):
    tol = rtol * np.linalg.norm(y0) + atol
    h = get_first_step(p, tol, f, x0, y0, x_end)

    x_arr, y_arr = [x0], [y0]

    y_old = main_method(f, x0, y0, h)
    y_mid = main_method(f, x0, y0, h / 2)
    y_new = main_method(f, x0 + h / 2, y_mid, h / 2)

    err_old = (y_new - y_old) / (1 - 2 ** (-p))
    err_new = (y_new - y_old) / ((2 ** p) - 1)

    h_cur = h
    h_next = h
    h_max = h_cur

    while x_arr[-1] < x_end:
        if np.linalg.norm(err_old) > tol * 2 ** p:
            h_cur /= 2

            y_old = y_mid
            y_mid = main_method(f, x_arr[-1], y_arr[-1], h_cur / 2)
            y_new = main_method(f, x_arr[-1] + h_cur / 2, y_mid, h_cur / 2)

            err_old = (y_new - y_old) / (1 - 2 ** (-p))
            err_new = (y_new - y_old) / ((2 ** p) - 1)

            continue

        elif tol < np.linalg.norm(err_old) <= tol * 2 ** p:
            h_next = h_cur / 2

            x_arr.append(x_arr[-1] + h_cur)
            y_arr.append(y_new + err_new)

            h_max = max(h_max, h_cur)

        elif tol * (2 ** (-p - 1)) <= np.linalg.norm(err_old) <= tol:
            h_next = h_cur

            x_arr.append(x_arr[-1] + h_cur)
            y_arr.append(y_old + err_old)

            h_max = max(h_max, h_cur)

        elif np.linalg.norm(err_old) < tol * (2 ** (-p - 1)):
            h_next = min(2 * h_cur, h_max)

            x_arr.append(x_arr[-1] + h_cur)
            y_arr.append(y_old + err_old)

            h_max = max(h_max, h_cur)

        h_cur = h_next

        y_old = main_method(f, x_arr[-1], y_arr[-1], h_cur)
        y_mid = main_method(f, x_arr[-1], y_arr[-1], h_cur / 2)
        y_new = main_method(f, x_arr[-1] + h_cur / 2, y_mid, h_cur / 2)

        err_old = (y_new - y_old) / (1 - 2 ** (-p))
        err_new = (y_new - y_old) / ((2 ** p) - 1)
        tol = rtol * np.linalg.norm(y_arr[-1]) + atol

    return np.array(x_arr), np.array(y_arr)


def plot_optimal(rtol, atol, p):
    x_opt, y_opt = solve_optimal(p, rtol, atol, f, **par)

    for i in range(4):
        plt.plot(x_opt, exact_ans(x_opt)[i], label='exact $y_{%s}(x)$' % (i + 1))
        plt.plot(x_opt, y_opt[:, i], label='approx $y_{%s}(x)$' % (i + 1))
        plt.legend()
        plt.show()

    err = exact_ans(x_opt).T - y_opt
    err = -np.log2(np.sum(err ** 2, axis=1) ** 0.5)

    plt.plot(x_opt, err)
    plt.xlabel('x')
    plt.ylabel('error(x)')
    plt.show()


if __name__ == "__main__":
    plot_optimal(0.000001, 1e-12, 2)
