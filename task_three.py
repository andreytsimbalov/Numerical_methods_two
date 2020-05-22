from task_one import *


def get_h_opt(h, tol, y_n, y_2n, p):
    r_n = (y_2n - y_n) / (1 - 2 ** (-p))


    return h * np.power(tol / np.linalg.norm(r_n), 1 / p)


def solve_h_opt(p, tol, h, y_n, y_2n, f, x0, y0, x_end, n):
    h_opt = get_h_opt(h, tol, y_n, y_2n, p)

    n = (x_end - x0) / h_opt
    n = int(n)

    x_arr, y_arr = [x0], [y0]

    for i in range(n):
        x_new = x_arr[i] + h_opt
        x_arr.append(x_new)

        y_new = main_method(f, x_arr[i], y_arr[i], h_opt)
        y_arr.append(y_new)

    return np.array(x_arr), np.array(y_arr), h_opt


if __name__ == "__main__":
    k = 7
    tol = 1e-5
    p = 2
    h_old = 1 / 2 ** k

    x0, y0 = solve(h_old, f, **par)
    x1, y1 = solve(h_old / 2, f, **par)

    x_opt, y_opt, h_opt = solve_h_opt(p, tol, h_old, y0[-1], y1[-1], f, **par)

    print("h_old: ", h_old)
    print("h_opt: ", h_opt)

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
