# This script creates a plot with 7 curves based on the provided data.
import matplotlib.pyplot as plt

data = {
    (1,   4): [ 2.63,  6.37, 21.07, 21.07, 21.90, 22.53, 22.33],
    (2,   6): [ 4.43,  9.83, 25.37, 25.93, 27.10, 27.07, 27.33],
    (2,   8): [ 6.47, 11.43, 27.60, 27.90, 28.87, 28.67, 28.87],
    (2,  10): [ 7.20, 12.37, 27.87, 28.00, 29.03, 28.73, 29.07],
    (3,   9): [ 6.82, 13.10, 28.97, 30.63, 31.33, 31.00, 32.57],
    (3,  12): [ 9.75, 15.70, 30.30, 31.43, 32.10, 31.73, 33.13],
    (3,  15): [11.25, 17.37, 30.53, 31.60, 32.27, 31.90, 33.13],
    (5,  10): [ 7.23, 11.77, 32.23, 32.33, 34.00, 34.27, 34.97],
    (5,  15): [11.00, 17.67, 34.70, 34.63, 36.33, 36.23, 37.33],
    (5,  20): [13.97, 20.77, 35.57, 34.97, 36.53, 36.53, 37.70],
    (8,  16): [10.43, 14.57, 36.10, 37.27, 38.17, 38.87, 38.87],
    (8,  24): [14.53, 19.63, 37.93, 38.40, 39.50, 39.83, 39.83],
    (8,  32): [16.80, 22.00, 38.13, 38.70, 39.40, 39.93, 39.90],
    (13, 26): [ 9.23, 17.43, 39.43, 40.10, 41.93, 40.97, 41.90],
    (13, 39): [15.37, 21.57, 39.83, 40.67, 42.17, 41.37, 42.43],
    (13, 52): [18.23, 23.00, 40.07, 40.63, 42.13, 41.33, 42.37],
    (21, 42): [10.77, 17.43, 40.67, 41.90, 42.93, 43.07, 43.10],
    (21, 63): [15.83, 21.10, 41.40, 42.03, 42.90, 42.97, 43.23],
    (21, 84): [18.63, 23.17, 41.33, 41.87, 42.90, 42.90, 43.10],
    (34, 68): [11.67, 17.27, 42.73, 43.43, 44.60, 44.00, 44.86],
    (34, 102): [16.57, 20.13, 42.63, 43.13, 44.27, 43.83, 44.42],
    (34, 136): [18.80, 21.97, 42.40, 43.20, 44.10, 43.77, 44.38],
}

# Prepare x and y for each curve, removing points where a later x has a higher y (make curves non-decreasing)
filtered_xs = []
filtered_ys = []

for y_curve in zip(*data.values()):
    x_curve = [b*e*20 for (b, e) in data.keys()]  # Multiply b*e by 20
    filtered_x = []
    filtered_y = []
    max_y = float('-inf')
    for xi, yi in sorted(zip(x_curve, y_curve)):
        if yi >= max_y:
            filtered_x.append(xi)
            filtered_y.append(yi)
            max_y = yi
    filtered_xs.append(filtered_x)
    filtered_ys.append(filtered_y)

for i, (x, y) in enumerate(zip(filtered_xs, filtered_ys)):
    plt.plot(x, y, marker='o', label=f'Checkpoint {i+1}')
    

plt.xlabel('tokens')
plt.ylabel('Accuracy %')
plt.xscale('log')
plt.legend(fontsize='small')
plt.title('Accuracy vs. Tokens')
plt.savefig('plot.png')
plt.show()
