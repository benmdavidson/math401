#!/usr/bin/env python3

import random
import numpy as np
from matplotlib import pyplot as plt

# This program makes two numpy arrays of random numbers [0,1), performs
# basic linear algebra calculations on them, and creates a plot.

x = np.random.rand(50)
y = np.random.rand(50)

added = np.add(x, y)
subtracted = np.subtract(x, y)
dotted = np.dot(x, y)

print("Added : {}".format(added))
print("Subtracted : {}".format(added))
print("Dotted : {}".format(dotted))

z = x[10:21]

plt.title("Question 6")
plt.xlabel("Index")
plt.ylabel("Value")
plt.plot(z)
plt.savefig("classwork_plot.png")
