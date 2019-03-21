import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
cycol = cycle('bgrcmk')

data = dict()
names = []

def get_data(filename, loss_idx, val_idx, name):
    data[name] = []
    names.append(name)
    with open(filename, 'r') as f:
        for num, line in enumerate(f):
            line = line.split(',')
            data[name].append(float(line[loss_idx]))

get_data('train-loss-1', 0, 1, 'k=1')
get_data('train-loss-0.75', 0, 1, 'k=0.75')
get_data('train-loss-0.5', 0, 1, 'k=0.5')
get_data('train-loss-0.25', 0, 1, 'k=0.25')


for name in names:
    plt.plot(data[name][::10], c=next(cycol), label=name)

plt.legend()
plt.show()

