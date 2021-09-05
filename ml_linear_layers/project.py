#!/usr/bin/env python3

# Imports
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

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
print("y1 = {}\ny2 = {}\ny3 = {}".format(y1, y2, y3))
# Output:
# y1 = -3*x1 + 3*x2
# y2 = 3*x1 - 8*x2
# y3 = 6*x1 + 4*x2
#
# b)
# y1 < 0:
# 	    -3*x1 + 3*x2 < 0
# 	    0 if x2 < x1
# y2 < 0:
# 	    3*x1 - 8*x2 < 0
#	    0 if x2 > 3/8 * x1
# y3 < 0:
#       6*x1 + 4*x2 < 0
#	    0 if x2 < -6/4 * x1
#
# c)
# There are 6 cases for the Ramp operation.
# There are 8 technical possibility cases considering parity (2^3).
# We can see, though, that 3 lines that intersect at a common point
#   in 2d create only 6 regions.
# Two of the 8 "possible" cases are in fact impossible.
# Note that the cases [0, 0, 0] and [y1, 0, 0] are invalid for this example.

case_1 = [y1, 0, 0]
case_2 = [y1, y2, 0]
case_3 = [0, y2, 0]
case_4 = [0, y2, y3]
case_5 = [0, 0, y3]
case_6 = [y1, 0, y3]

cases = [case_1, case_2, case_3, case_4, case_5, case_6]

# d)
results = map(second_linear_layer, cases)

print("Cases: ")
idx = 1
for case in results:
    print("Case {} -> z1 = {} , z2 = {}".format(idx, case[0], case[1]))
    idx += 1
# Output:
# Cases:
# Case 1 -> z1 = -3*x1 + 3*x2 , z2 = 12*x1 - 12*x2
# Case 2 -> z1 = -18*x1 + 43*x2 , z2 = 9*x1 - 4*x2
# Case 3 -> z1 = -15*x1 + 40*x2 , z2 = -3*x1 + 8*x2
# Case 4 -> z1 = 9*x1 + 56*x2 , z2 = 27*x1 + 28*x2
# Case 5 -> z1 = 24*x1 + 16*x2 , z2 = 30*x1 + 20*x2
# Case 6 -> z1 = 21*x1 + 19*x2 , z2 = 42*x1 + 8*x2
#
# e)
# Cases: 
# Case 1 -> -3*x1 + 3*x2 > 0 , 12*z1 - 12*x2 > 0
#			x2 > x1              y < x1
# Case 2 -> -18*x1 + 43*x2 > 0 and 9*x1 - 4*x2 > 0 
#            x2 > (18/43)*x1       x2 < (9/4)*x1
# Case 3 -> -15*x1 + 40*x2 > 0 and -3*x1 + 8*x2 > 0
#            x2 > (15/40)*x1        x2 > (3/8)*x1 
# Case 4 -> 9*x1 + 56*x2 > 0 and 27*x1 + 28*x2 > 0
#           x2 > (-9/56)*x1      x2 > (-27/28)*x1
# Case 5 -> 24*x1 + 16*x2 > 0 and 30*z1 + 20*x2 > 0
#           x2 > (-9/56)*x1       x2 > (-30/20)*x1
# Case 6 -> 21*x1 + 19*x2 > 0 and 42*z1 + 8*x2 > 0
#           x2 > (-21/19)*x1      x2 > (-21/19)*x1 
#
# The ramp operation is what the case depends on because for each case, the 
# vector (y1, y2, y3) output by the ramp function is different, consequently affecting
# the output vector after the second linear layer.
#
# f)
# Using matplotlib to draw these regions

x = np.arange(-1, 1, .001)
y1 = x
y2 = (3/8)*x
y3 = (-6/4)*x

def graph_util(z1, z2):
	plt.clf()
	plt.xlim([-1, 1])
	plt.ylim([-1, 1])
	plt.grid()
	plt.plot(x, y1, 'r')
	plt.plot(x, y2, 'g')
	plt.plot(x, y3, 'b')
	plt.plot(x, z1, 'y')
	plt.plot(x, z2, 'y')

# Case 1
# 6*x1 + 59*x2 > 0 and 39*x1 + 16*x2 > 0
# x2 > (6/59)*x1
z1 = x
# x2 > (-39/16)*x1
z2 = x
graph_util(z1, z2)
plt.fill_between(x, z1, z1, where = x > 0, color = 'grey', alpha = 0.5) 
plt.title('Case 1 Region')
plt.savefig('images/case_1_region.png')

# Case 2
# -18*x1 + 43*x2 > 0 and 9*x1 - 4*x2 > 0 
# x2 > (18/43)*x1
z1 = (18/43)*x
# x2 < (9/4)*x1
z2 = (9/4)*x
graph_util(z1, z2)
plt.fill_between(x, z1, y1, where = x < 0, color = 'grey', alpha = 0.5) 
plt.title('Case 2 Region')
plt.savefig('images/case_2_region.png')

# Case 3
# -15*x1 + 40*x2 > 0 and -3*x1 + 8*x2 > 0
# x2 > (15/40)*x1
z1 = (15/40)*x
# x2 > (3/8)*x1
z2 = (3/8)*x
graph_util(z1, z2)
plt.fill_between(x, z1, z2, color = 'grey', alpha = 0.5) 
plt.title('Case 3 Region')
plt.savefig('images/case_3_region.png')

# Case 4
# 9*x1 + 56*x2 > 0 and 27*x1 + 28*x2 > 0
# x2 > (-9/56)*x1
z1 = (-9/56)*x
# x2 > (-27/28)*x1
z2 = (-27/28)*x
graph_util(z1, z2)
plt.fill_between(x, z1, z2, where = x > 0, color = 'grey', alpha = 0.5) 
plt.title('Case 4 Region')
plt.savefig('images/case_4_region.png')

# Case 5
# 24*x1 + 16*x2 > 0 and 30*z1 + 20*x2 > 0
# x2 > (-24/16)*x1
z1 = (-24/16)*x
# x2 > (-30/20)*x1
z2 = (-30/20)*x
graph_util(z1, z2)
plt.fill_between(x, z1, z2, color = 'grey', alpha = 0.5) 
plt.title('Case 5 Region')
plt.savefig('images/case_5_region.png')

# Case 6
# 21*x1 + 19*x2 > 0 and 42*z1 + 8*x2 > 0
# x2 > (-21/19)*x1
z1 = (-21/19)*x
# x2 > (-42/8)*x1
z2 = (-42/9)*x
graph_util(z1, z2)
plt.fill_between(x, y3, z2, where = x < 0, color = 'grey', alpha = 0.5) 
plt.title('Case 6 Region')
plt.savefig('images/case_6_region.png')

# Combined Graphic

plt.clf()
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.grid()
plt.plot(x, y1, 'r')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b')

# Case 1
z1 = x
z2 = x
plt.fill_between(x, z1, z1, where = x < 0, color = 'grey', alpha = 0.5) 
# Case 2
z1 = (18/43)*x
z2 = (9/4)*x
plt.fill_between(x, z1, y1, where = x < 0, color = 'grey', alpha = 0.5) 
# Case 3
z1 = (15/40)*x
z2 = (3/8)*x
plt.fill_between(x, z1, z2, color = 'grey', alpha = 0.5) 
# Case 4
z1 = (-9/56)*x
z2 = (-27/28)*x
plt.fill_between(x, z1, z2, where = x > 0, color = 'grey', alpha = 0.5) 
# Case 5
z1 = (-24/16)*x
z2 = (-30/20)*x
plt.fill_between(x, z1, z2, color = 'grey', alpha = 0.5) 
# Case 6
z1 = (-21/19)*x
z2 = (-42/9)*x
plt.fill_between(x, y3, z2, where = x < 0, color = 'grey', alpha = 0.5) 
plt.title('Combined Graphic')
plt.savefig('images/combined_graphic.png')

# 2.
X = np.random.uniform(-1, 1, [1000, 2])
results = np.array(list(map(second_linear_layer, map(ramp, map(first_linear_layer, X)))))

plt.clf()
plt.xlim([-60, 60])
plt.ylim([-60, 60])
plt.axvline(0, 0, 1, color = 'k')
plt.axhline(0, 0, 1, color = 'k')
x_vals = results[:,0]
y_vals = results[:,1]
col = np.where(x_vals < 0, 'b', np.where(y_vals < 0, 'b', 'r'))
plt.scatter(x_vals, y_vals, c=col)
plt.title("Neural Network Output")
plt.savefig('images/scatter_plot.png')
