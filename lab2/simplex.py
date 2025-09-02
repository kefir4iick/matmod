import numpy as np


def simplex(v_coef, matrix, v_cons):
    m, n = matrix.shape

    table = np.hstack([matrix, np.eye(m), v_cons.reshape(-1, 1)])
    table = np.vstack([table, np.hstack([v_coef, np.zeros(m + 1)])])

    iteration = 0
    while True:
        iteration += 1

        if all(table[-1, :-1] >= -1e-10):
            break

        enter_column = np.argmin(table[-1, :-1])

        ratios = np.full(m, np.inf)
        for i in range(m):
            if table[i, enter_column] > 1e-10:
                ratios[i] = table[i, -1] / table[i, enter_column]

        if all(ratios == np.inf):
            raise ValueError("Problem is unbounded")

        leave_row = np.argmin(ratios)

        pivot = table[leave_row, enter_column]

        table[leave_row] = table[leave_row] / pivot

        for i in range(m + 1):
            if i != leave_row:
                factor = table[i, enter_column]
                table[i] = table[i] - factor * table[leave_row]

    solution = np.zeros(n)
    for j in range(n):
        col = table[:-1, j]
        if np.sum(np.abs(col)) == 1 and np.sum(col > 0) == 1:
            row_idx = np.where(col > 0)[0][0]
            solution[j] = table[row_idx, -1]

    return table[-1, -1], solution


c = np.array([-1, -1, 0, 0, 0])

A = np.array([
    [2, 11, 1, 0, 0],
    [1, 1, 0, 1, 0],
    [4, -5, 0, 0, 1]
])

b = np.array([38, 7, 5])

optimal_value, solution = simplex(c, A, b)

print("=" * 50)
print("FINAL RESULTS:")
print("Optimal x1:", solution[0])
print("Optimal x2:", solution[1])
print("Slack variables:", solution[2:])
print("Max value of function z = x1 + x2:", optimal_value)
print("=" * 50)

print("\nCONSTRAINT CHECK:")
print(f"2*{solution[0]} + 11*{solution[1]} = {2 * solution[0] + 11 * solution[1]} ≤ 38")
print(f"{solution[0]} + {solution[1]} = {solution[0] + solution[1]} ≤ 7")
print(f"4*{solution[0]} - 5*{solution[1]} = {4 * solution[0] - 5 * solution[1]} ≤ 5")
