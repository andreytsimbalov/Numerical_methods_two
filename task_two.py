from task_one import *

if __name__ == "__main__":
    kk = np.arange(4, 15)
    norms = []

    for k in kk:
        x0, y0, x_end = par['x0'], par['y0'], par['x_end']

        method_x, method_y = solve(-1, f, x0, y0, x_end, (x_end - x0) * (2 ** k))
        err = np.linalg.norm(exact_ans(method_x[-1]) - method_y[-1])
        norms.append(-np.log2(err))
        print(k, -np.log2(err))

    plt.plot(kk, norms)
    plt.show()
