#!/usr/bin/env python3

# Imports
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# File description

# Definitions

A = np.array([[-3, 3], [3, -8], [6, 4]])
B = np.array([[1, -5, 4], [-4, -1, 5]])

def first_linear_layer(v):
	return np.dot(A, v)

def ramp(v):
	return (v + abs(v)) / 2

def second_linear_layer(v):
	return np.dot(B, v)

x1 = sym.Symbol("x1")
x2 = sym.Symbol("x2")
v = [x1, x2]

# 1.
# a)
first_layer_result = first_linear_layer(v)
y1, y2, y3 = first_layer_result
print(first_layer_result)
print("y1 = {}\ny2 = {}\ny3 = {}".format(y1, y2, y3))

# b)
# y1 < 0 into -3*x1 + 3*x2 < 0
# y2 < 0 into 3*x1 - 8*x2 < 0
# y3 < 0 into 6*x1 + 4*x2 < 0

# c)
# There are 6 cases for the Ramp operation.
# There are 8 technical possibility cases considering parity (2^3).
# We can see, though, that 3 intersecting lines in 2d create only 6 regions.
# The solution for this system is at x1 = 0 and x2 = 0, that is, they intersect at (0, 0)

# Note that the cases [0, 0, 0] and [y1, y2, y3] are invalid
c = 6
case_1 = [y1, y2, 0]
case_2 = [y1, 0, y3]
case_3 = [y1, 0, 0]
case_4 = [0, y2, y3]
case_5 = [0, y2, 0]
case_6 = [0, 0, y3]
cases = [case_1, case_2, case_3, case_4, case_5, case_6]

# d)
results = map(second_linear_layer, cases)

print("Cases: ")
idx = 1
for case in results:
	print("Case {} -> z1 = {} , z2 = {}".format(idx, case[0], case[1]))
	idx += 1
	# This prints: 
	# Cases:
	# Case 1 -> z1 = -18*x1 + 43*x2 , z2 = 9*x1 - 4*x2
	# Case 2 -> z1 = 21*x1 + 19*x2 , z2 = 42*x1 + 8*x2
	# Case 3 -> z1 = -3*x1 + 3*x2 , z2 = 12*x1 - 12*x2
	# Case 4 -> z1 = 9*x1 + 56*x2 , z2 = 27*x1 + 28*x2
	# Case 5 -> z1 = -15*x1 + 40*x2 , z2 = -3*x1 + 8*x2
	# Case 6 -> z1 = 24*x1 + 16*x2 , z2 = 30*x1 + 20*x2

# e)
# Cases: 
# Case 1 -> -18*x1 + 43*x2 > 0 and 9*x1 - 4*x2 > 0
# Case 2 -> 21*x1 + 19*x2 > 0 and 42*x1 + 8*x2 > 0
# Case 3 -> -3*x1 + 3*x2 > 0 and 12*x1 - 12*x2 > 0
# Case 4 -> 9*x1 + 56*x2 > 0 and 27*x1 + 28*x2 > 0
# Case 5 -> -15*x1 + 40*x2 > 0 and -3*x1 + 8*x2 > 0
# Case 6 -> 24*x1 + 16*x2 > 0 and 30*x1 + 20*x2 > 01
#
# The conditions for which x1 and x2 are mapped to the first quadrant depend
# on the case of the Ramp operation because the third layer is a linear layer.
# This means that a linear transformation is being performed on the result of
# the ramp operation. Each case of the Ramp operation represents a different
# and disjoint region of 2d space. Mapping these 6 distinct regions to the
# same target region (quadrant 1) is what leads to different restrictions on
# x1 and x2 for each case.

# f)
# Using matplot lib to draw these regions

x = np.arange(-1, 1, .001)
# Case 1
# -18*x1 + 43*x2 > 0 and 9*x1 - 4*x2 > 0
plt.clf()
plt.xlim([-1, 1])
plt.ylim([-100, 100])
plt.grid()
z1 = (18 / 43)*x
z2 = (9 / 4)*x
plt.plot(x, z1, 'r')
plt.plot(x, z2, 'g')
plt.fill_between(x, z1, z2, where = x > 0) 
plt.savefig('case_1_region.png')

# Case 2
# 21*x1 + 19 * x2 > 0 and 42*x1 + 8*x2 > 0
plt.clf()
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.grid()
z1 = (-21 / 19)*x
z2 = (-42 / 8)*x
plt.plot(x, z1, 'r')
plt.plot(x, z2, 'g')
Z1 = np.maximum(z1, z2)
plt.fill_between(x, Z1, np.max(z1)) 
plt.savefig('case_2_region.png')

# Case 3
# -3*x1 + 3*x2 > 0 and 12*x1 - 12*x2 > 0 
plt.clf()
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.grid()
z1 = x
z2 = x
plt.plot(x, z1, 'r')
plt.plot(x, z2, 'g')
Z1 = np.maximum(z1, z2)
plt.fill_between(x, Z1, np.max(z1)) 
plt.savefig('case_3_region.png')