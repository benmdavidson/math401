#!/usr/bin/env python3

import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=np.inf)

# Adjacency Matrix
# {{0, 0, 0, 0, 0, 1, 1, 0},
#  {0, 0, 0, 0, 1, 1, 1, 0},
#  {0, 0, 0, 0, 1, 1, 1, 0},
#  {0, 0, 0, 0, 1, 1, 0, 1},
#  {1, 1, 1, 1, 0, 0, 0, 0},
#  {1, 0, 1, 0, 0, 0, 0, 0},
#  {0, 1, 1, 0, 0, 0, 0, 0},
#  {1, 1, 0, 1, 0, 0, 0, 0}}

adj = np.array([
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 0, 0, 0, 0]	
])

# 1)
G =nx.from_numpy_matrix(np.transpose(adj), create_using=nx.MultiDiGraph())
nx.draw(G, with_labels=True)
plt.savefig("images/directed_graph.png")

# 2)
A = np.zeros((8,8), dtype='float64')
for i in range(np.shape(adj)[1]):
    A[:,i] = adj[:,i] / np.sum(adj[:,i])

# 3)
A2 = A @ A
A4 = A2 @ A2
A8 = A4 @ A4
A16 = A8 @ A8

# 4)
eig_vals, eig_vecs = np.linalg.eig(A)
# print(eig_vals)
# print(eig_vecs)
# There is a dominant eigenvector.
# There are two eigenvalues that have absolute value of 1,

# 5)
dom_eig_vec = eig_vecs[:,0]
# dom_eig_vec = [-0.085 -0.254 -0.254 -0.621 -0.508 -0.113 -0.169 -0.423] 
dom_prob_vec = dom_eig_vec / sum(dom_eig_vec)
# dom_prob_vec = [0.035 0.105 0.105 0.256 0.209 0.047 0.07  0.174]
# The values of dom_prob_vec equal almost exactly 1/2 the entry
# in the corresponding row of A16, which means when you normalize to be a
# probability vector they are basically the same.

# 6)
low, *_, high = sorted(dom_prob_vec)
norm = mpl.colors.Normalize(vmin=low, vmax=high, clip=True)
mapper = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.coolwarm)

plt.clf()
plt.title("Frequencies according to stable eigenvector \n Red is more frequent")
nx.draw(G, with_labels=True, node_color=[mapper.to_rgba(i) 
                                         for i in dom_prob_vec])
plt.savefig("images/colored_graph.png")

# 7)
v = np.array([1, 0, 0, 0, 0, 0, 0, 0])
randoms = np.random.uniform(0, 1, 20)
# [0.287 0.182 0.041 0.351 0.03  0.037 0.109 0.672 0.445 0.126
#  0.597 0.056 0.794 0.145 0.545 0.537 0.177 0.321 0.429 0.939]
# step 1:  vertex 0
# step 2:  vertex 4
# step 3:  vertex 1
# step 4:  vertex 4
# step 5:  vertex 2
# step 6:  vertex 4
# step 7:  vertex 1
# step 8:  vertex 4
# step 9:  vertex 2
# step 10: vertex 5
# step 11: vertex 0
# step 12: vertex 5
# step 13: vertex 0
# step 14: vertex 7
# step 15: vertex 3
# step 16: vertex 7
# step 17: vertex 3
# step 18: vertex 4
# step 19: vertex 1
# step 20: vertex 5
# Frequencies:
# vertex 0 -> 3
# vertex 1 -> 3
# vertex 2 -> 2
# vertex 3 -> 2
# vertex 4 -> 5
# vertex 5 -> 3
# vertex 6 -> 0
# vertex 7 -> 2
# There doesn't seem to be that much correlation between the observed frequencies
# and the probability vector from 5. This is because the number of steps on the walk
# is relatively small and succept to outliers. My random numbers were generally quite low,
# which caused the walk to favor some nodes over the others. I presume that over time
# or with a longer walk, the frequencies would average out to correlate with the probability
# vector from 5. This is because the dominant eigenvector from 4 dictates the stable behavior
# of the walk.

# 8)
integer_walk = [0]
for i in range(100):
    random = np.random.randint(2)
    integer_walk.append(integer_walk[i] + (-1 if random == 1 else 1))

# 9)
plt.clf()
plt.plot(np.abs(integer_walk))
plt.title("Random walk over integers (abs value)")
plt.savefig("images/random_integer_walk.png")

# 10)
x = np.linspace(0, 100, 100)
plt.plot(x, x)
plt.plot(x, np.sqrt(x))
plt.title("Random walk with y=x and y=sqrt(x)")
plt.savefig("images/overlayed_random_integer_walk.png")
# It is clear that the random walk follows the trend of y = sqrt(x) very
# closely and y = x goes far higher quite quickly. This is expected. The
# long term behavior of a random walk over the integers trends towards sqrt(x)