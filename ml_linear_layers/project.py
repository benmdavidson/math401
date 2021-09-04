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
for case in results:
	print(repr(case))
