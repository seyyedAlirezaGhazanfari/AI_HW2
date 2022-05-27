import matplotlib.pyplot as plt
import numpy as np

tolerance = 0.1
h = 0.000001
tolerance_2 = 0.1


def plot_design(func, start, end):
    x = np.linspace(start, end, 100)
    y = func(x)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.plot(x, y, 'r')
    plt.show()


def func_1(x):
    return ((x ** 4) * (np.exp(x)) - np.sin(x)) / 2


def func_2(x):
    a = 5 * (np.log10(np.sin(5 * x) + np.sqrt(x)))
    return a


def func_3(x):
    return np.cos(5 * np.log10(x)) - (x ** 3) / 10


start_range_func_1 = -2
end_range_func_1 = 1
start_range_func_2 = 2
end_range_func_2 = 6
start_range_func_3 = 0.5
end_range_func_3 = 2

plot_design(func_1, -2, 1)
plot_design(func_2, 2, 6)
plot_design(func_3, 0.5, 2)


def derivative(func, x):
    return (func(x + h) - func(x)) / h


def gradient_descent(func, learning_rate, max_repeat, start_point, end_point):
    x = float(np.random.uniform(low=start_point, high=end_point, size=1))
    for _ in range(max_repeat):
        diff = - learning_rate * derivative(func, x)
        if abs(diff) < 0.001:
            break
        x += diff
        if x < start_point:
            x = start_point
        elif x > end_point:
            x = end_point
    return x


print("part 2")
learning_rates = [0.1, 0.4, 0.6, 0.9]
for learning_rate in learning_rates:
    x = gradient_descent(func_1, learning_rate, 1000000, start_range_func_1, end_range_func_1)
    print("learning rate = " + str(learning_rate) + " -------> " + "x = " + str(x) + " f(x) = " + str(func_1(x)))


def find_min(func, learning_rate):
    return gradient_descent(func, learning_rate, 100000, 2, 2.5)


print("part 3")
minimum = find_min(func_2, 0.001)
print("original minimum = " + "x = " + str(minimum) + " f(x) = " + str(func_2(minimum)))
for learning_rate in learning_rates:
    success = 0
    for _ in range(1000):
        result = gradient_descent(func_2, learning_rate, 1000, start_range_func_2, end_range_func_2)
        if func_2(minimum) - tolerance_2 <= func_2(result) <= func_2(minimum) + tolerance_2:
            success += 1
    print("learning rate = " + str(learning_rate) + " ---> " + "success rate = " + str(success / 10))


# part 4
def second_derivative(func, x):
    return (func(x + h) - 2 * func(x) + func(x - h)) / (h ** 2)


def newton_raphson(func, max_iteration, start, end):
    x = float(np.random.uniform(low=start, high=end, size=1))
    for _ in range(max_iteration):
        diff = - derivative(func, x) / second_derivative(func, x)
        if diff == 0 and second_derivative(func, x) > 0:
            break
        x += diff
        if x < start:
            x = start
        elif x > end:
            x = end
    return x


print("part 4")
x = newton_raphson(func_1, 100000, start_range_func_1, end_range_func_1)
print("convex function minimum = " + "x = " + str(x) + " f(x) = " + str(func_1(x)))

success = 0
for _ in range(1000):
    result = newton_raphson(func_2, 1000, start_range_func_2, end_range_func_2)
    if func_2(minimum) - tolerance_2 <= func_2(result) <= func_2(minimum) + tolerance_2:
        success += 1
print("success rate = " + str(success / 10))


# part 5
def partial_derivative(func, x, y, is_x_derivation=True):
    h = 0.001
    if is_x_derivation:
        return (func(x + h, y) - func(x, y)) / h
    else:
        return (func(x, y + h) - func(x, y)) / h


def gradient(func, x, y):
    vector = [partial_derivative(func, x, y), partial_derivative(func, x, y, False)]
    return vector


def gradient_descent2(func, learning_rate, max_repeat, start_position):
    position = list(start_position)
    x1_seq = list()
    x2_seq = list()
    for _ in range(max_repeat):
        x1_seq.append(position[0])
        x2_seq.append(position[1])
        vector = gradient(func, position[0], position[1])
        position[0] -= learning_rate * vector[0]
        position[1] -= learning_rate * vector[1]
        if position[0] > 15:
            position[0] = 15
        elif position[0] < -15:
            position[0] = -15
        if position[1] > 15:
            position[1] = 15
        elif position[1] < -15:
            position[1] = -15
    return position, x1_seq, x2_seq


def draw_points(func, x_1_sequence, x_2_sequence):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    X1, X2 = np.meshgrid(np.linspace(-15.0, 15.0, 1000), np.linspace(-15.0, 15.0, 1000))
    Y = func(X1, X2)
    f_sequence = [func(x_1_sequence[i], x_2_sequence[i]) for i in range(len(x_1_sequence))]

    # First subplot
    ax = fig.add_subplot(1, 2, 1)

    cp = ax.contour(X1, X2, Y, colors='black', linestyles='dashed', linewidths=1)
    ax.clabel(cp, inline=1, fontsize=10)
    cp = ax.contourf(X1, X2, Y, )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.scatter(x_1_sequence, x_2_sequence, s=10, c="y")

    # Second subplot
    ax = fig.add_subplot(1, 2, 2, projection='3d')

    ax.contour3D(X1, X2, Y, 50, cmap="Blues")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter3D(x_1_sequence, x_2_sequence, f_sequence, s=10, c="r")

    plt.show()


def optimization3(func, start, end, learning_rate, max_repeat):
    start_point_x = float(np.random.uniform(low=start, high=end, size=1))
    start_point_y = float(np.random.uniform(low=start, high=end, size=1))
    result_position, x1_seq, x2_seq = gradient_descent2(func, learning_rate, max_repeat, (start_point_x, start_point_y))

    draw_points(func, x1_seq, x2_seq)
    return result_position


def func_4(x, y):
    return ((2 ** x) / 10000) + (np.exp(y) / 20000) + (x ** 2) + (4 * (y ** 2)) - (2 * x) - (3 * y)


learning_rates = [0.01, 0.1, 0.18, 0.25]
print("part 5")
for learning_rate in learning_rates:
    result = optimization3(func_4, -15, 15, learning_rate, 1000)
    print(
        "learning rate = " + str(learning_rate) + " ---> " + "x position = " + str(result[0]) + " y position = " + str(
            result[1]) + " f(x, y) = " + str(func_4(result[0], result[1])))


# last part
def simulated_annealing_optimization(f, stopping_temp, stopping_iter, alpha, initial_T, gamma):
    current = float(np.random.uniform(2, 6, 1))
    T = initial_T
    for t in range(1, stopping_iter + 1):
        T = gamma * T
        if T == stopping_temp:
            break
        next = float(np.random.uniform(
            low=current - alpha,
            high=current + alpha,
            size=1
        )
        )
        if next < 2:
            next = 2
        elif next > 6:
            next = 6
        E = f(next) - f(current)
        if E < 0:
            current = next
        elif np.random.rand() < np.exp((-1 * E) / T):
            current = next
    return current


def create_alphas():
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


alphas = create_alphas()
print("part 6")
for alpha in alphas:
    success = 0
    for _ in range(1000):
        result = simulated_annealing_optimization(func_2, 0, 1000, alpha, 200, 0.9)
        if minimum - tolerance <= result <= minimum + tolerance:
            success += 1
    print("alpha = " + str(alpha) + " ----> " + str(success / 10))
