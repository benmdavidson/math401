#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# 1)
S = np.array([[1.14412, -0.371748], [0.511667, -0.371748],
              [0.707107, -0.973249], [1.33956, -0.973249]])

colors = np.array(["red", "red", "red", "blue"])
x = S[:,0]
y = S[:,1]
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.grid()
plt.title("Angle 0")
scatter = plt.scatter(x, y, c=colors)
plt.savefig("images/scatter_0.png")
scatter.remove()

# 2)
theta = np.pi * 2/5
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta), np.cos(theta)]])

# This function translates the points such that the pivot point is the origin,
# performs a rotation by 2pi / 5 then translates those points back.
# This effecitvely rotates the points about the pivot.
def perform_rotation(points):
    rotate = lambda x: R @ x
    translate_to_origin = lambda x: x - points[-1]
    translate_back = lambda x: x + points[-1]
    return np.array(list(map(translate_back, map(rotate, map(translate_to_origin, points)))))

def graph_rotation(title, file_name, points):
    result = perform_rotation(points) 
    x = result[:,0]
    y = result[:,1]
    plt.title(title)
    scatter = plt.scatter(x, y, c=colors)
    plt.savefig(file_name)
    scatter.remove()
    return result

result1 = graph_rotation("Angle 2$\pi$ / 5", "images/scatter_1.png", S)
result2 = graph_rotation("Angle 4$\pi$ / 5", "images/scatter_2.png", result1)
result3 = graph_rotation("Angle 6$\pi$ / 5", "images/scatter_3.png", result2)
graph_rotation("Angle 8$\pi$ / 5", "images/scatter_4.png", result3)

# 3)
offsets = np.array([[-2.995, 0.973], [-1.851, -0.6015],
                    [-1.851, 2.548], [0, 0], [0, 1.9465]])

def get_translation_matrix(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])

translation_matrices = list(map(get_translation_matrix, offsets[:,0], offsets[:,1]))
S1 = S.copy()
S1 = np.transpose(S1)
S1 = np.vstack([(S1), np.ones(len(S1[0]))])
for matrix in translation_matrices:
    S1 = matrix @ S1

plt.xlim(-8, -4)
plt.ylim(2, 6)
x = S1[0,:]
y = S1[1,:]
plt.title("Post Translation")
scatter = plt.scatter(x, y, c=colors)
plt.savefig("images/translated_scatter_0.png")

# 4)
plt.title("Post Translation Rotations")

def perform_rotation_trans(points):
    result = perform_rotation(points) 
    x = result[:,0]
    y = result[:,1]
    plt.scatter(x, y, c=colors)
    return result

S1 = S1[:-1,:]
result1 = perform_rotation_trans(np.transpose(S1))
result2 = perform_rotation_trans(result1)
result3 = perform_rotation_trans(result2)
perform_rotation_trans(result3)
plt.savefig("images/translated_scatter_1.png")
# I observe that the points are rotating around the pivot point, as expected.
# It's also to note that each step requires only one matrix multilication. You can
# dot the rotation matrix with itself "n" times to rotate the points much more efficiently.
# This is a similar concept used in markov chains and random walks. This pattern can be extended to
# perform arbitrary rotations quite efficiently.

# 5)
m = []
for i in range(-3, 4):
    for j in range(-3, 4):
        for k in range(-3, 4):
            m.append([i,j,k])

# 6)
M = np.array(m)
T = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, -1]])
points = T @ np.transpose(M)

# 7)
A = np.array([[3, 2, 1],
              [1, 0, 0],
              [0, 1, 0]])

# do gram schmidt on A and rearrange coordinates so it's [x,y,z] again
A_gs = np.linalg.qr(A)[0]
A_gs_moved = np.array([A_gs[1], A_gs[2], A_gs[0]])

# projection matrix with d = distance between (3,2,1) and (0,0,0)
P = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 0],
              [0, 0, -1/3.741657, 1]])

with_perspective = A_gs_moved @ points
points = np.vstack([with_perspective, np.ones(len(points[0]))])

filtered = np.transpose(points)
filter_points = [True] * filtered.shape[0]
for c, v in enumerate(filtered):
    if v[2] > 0:
        filter_points[c] = False
    
post_filter = np.transpose(filtered[filter_points])

proj = P @ post_filter
for i in range(len(proj[0])):
    proj[:,i] /= proj[3,i]

x = proj[0,:]
y = proj[1,:]
plt.clf()
plt.scatter(x, y)
plt.savefig("images/projection.png")