#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:21:06 2023

@author: ruadhri
"""

import classicalsim as sim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# create a test hamiltonian for g=0 to check degeneracy
test_ham = sim.Hamiltonian(10, 0)
print(test_ham.exp_ground)


#%% check spectrum

datas = int(1e2 + 1)
g = np.linspace(-2, 2, datas)
q_bits = 2
energies = []

for i in range(datas):
    ham = sim.Hamiltonian(q_bits, g[i])
    energies.append(ham.exp_ground)

plt.figure()
plt.plot(g, energies)
plt.xlabel('Field Coupling Coefficient, g')
plt.ylabel(r'$\langle \psi_0\vert\hat H\vert\psi_0\rangle$')
plt.title(r'1D Ising $\hat{H}$ for ' + str(q_bits) + ' qubits')

#%% check if I can do inner product without for loops

test_state = np.array([[2, 1], [-1, -2]])
print(test_state.shape)

test_matrix = np.array([[1, 2], [3, 4]])

#mult = test_matrix * test_state#np.matmul(test_matrix, test_state)
mult = np.einsum('...ij,...j', test_matrix, test_state)#(test_matrix, test_state.T).T
print(mult)
#print(test_state.shape[0])
inner = np.diag(np.dot(mult, test_state.conj().T))
print(inner)