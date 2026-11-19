import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], float)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], float)


def scale_features(array_in):
    arr = np.copy(array_in)
    minn = np.min(arr)
    rangee = np.max(arr) - minn
    for i, v in enumerate(arr):
        arr[i] = (v - minn) / rangee
    return arr

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost(y, y_hat):
    t1 = np.dot(y, np.log(y_hat))
    t2 = np.dot((1 - y), np.log(1-y_hat))
    return - (t1 + t2) / y.size

def predict(x, w, b):
    z = np.dot(w, x) + b
    y_hat = sigmoid(z)
    if y_hat > .5:
        p = 1
    else:
        p = 0
    print(f"Prediction for x = {x}: {p}")

def logistic_regression(sett, y_vec, epochs, alpha):
    w = 0.5
    b = 0.0
    x = scale_features(np.copy(sett))
    # x = np.copy(sett)
    y = np.copy(y_vec)

    cost_array = np.array([])
    epoch_array = np.array([])

    for i in range(epochs):
        z = np.dot(w, x) + b
        y_hat = sigmoid(z)
        costt = cost(y, y_hat)

        cost_array = np.append(cost_array, costt)
        epoch_array = np.append(epoch_array, i)


        if i % 100 == 0:
            print(f"Epoch {i}:   Cost = {costt},   W = {w},   b = {b}")


        m = y.size
        dW = np.dot(x.T, (y_hat - y)) / m  # (X.T is shape (1, m); result is a scalar)
        dB = np.sum(y_hat - y) / m

        w -= alpha * dW
        b -= alpha * dB

    return w, b, cost_array, epoch_array

new_params = logistic_regression(x, y, 1000, .4)
w = new_params[0]
b = new_params[1]
print("\nTesting Model:")
predict(4.5, w, b)
predict(6.5, w, b)

costs = new_params[2]
epochs = new_params[3]
plt.plot(epochs, costs)
plt.xlabel("Number of Epochs")
plt.ylabel("Cost of W and b")
plt.show()