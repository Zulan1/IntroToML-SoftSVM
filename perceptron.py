# appendix for 6, not needed.
import numpy as np

def create_data_set(d):
    x_set, y_set = [], []
    for i in range(d):
        x = []
        for j in range(d):
            if j < i:
                x.append((-1) ** i)
            elif j == i:
                x.append((-1) ** (i + 1))
            else:
                x.append(0)
        x_set.append(np.array(x))
        y_set.append((-1) ** (i + 1))

    return np.vstack(x_set), np.vstack(y_set)

def perceptron(train_x, train_y):
    t = 1
    w = np.zeros(train_x.shape[1])
    while any(train_y[i] * (w @ train_x[i]) <= 0 for i in range(train_x.shape[0])):
        for i in range(train_x.shape[0]):
            if train_y[i] * (w @ train_x[i]) <= 0:
                w += train_y[i] * train_x[i]
                t += 1

    print(f'd={train_x.shape[1]}, t={t}\nw={w}')
    return w

for d in range(1, 11):
    train_x, train_y = create_data_set(d)
    perceptron(train_x, train_y)
    print('\n\n')
