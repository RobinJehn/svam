from matplotlib import pyplot as plt
x = [10.9]
y = [9.7]


for i in range(1, 1000):
    x_new = x[-1] + 0.1 * (x[-1] - x[-1] * y[-1])
    y_new = y[-1] + 0.1 * (y[-1] - x[-1] * y[-1])

    if x_new <= 0 or y_new <= 0:
        break
    x.append(x_new)
    y.append(y_new)

plt.plot(x, label='x')
plt.plot(y, label='y')
plt.legend()
plt.show()