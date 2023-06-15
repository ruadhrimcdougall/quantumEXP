#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:47:07 2023

@author: ruadhri
"""

import classicalsim as sim
import numpy as np
import matplotlib.pyplot as plt

#%% Create hamiltonians for different values of x, and plot epectation of ground
# state energy as a function of this variable

# Two qubits first

g_vals = np.linspace(-2, 2)
q_bits = 5
E0 = []

for i in range(len(g_vals)):
    new_hamiltonian = sim.Hamiltonian(q_bits, g_vals[i])
    exp_min_energy = new_hamiltonian.exp_ground
    E0.append(exp_min_energy)
    print(new_hamiltonian.ground_state)
    
plt.figure()
plt.plot(g_vals, E0)
plt.xlabel('Field Coupling Coefficient, g')
plt.ylabel(r'$< \psi_0|\hat H|\psi_0>$')
plt.title(r'1D Ising $\hat{H}$ for ' + str(q_bits) + ' qubits')

#%% Min. positive val test

arr = np.array([3, -1, 5, 0, 2])
mask = arr > 0
values_greater_than_zero = arr[mask]
index = np.argmin(values_greater_than_zero)
index_of_minimum = np.nonzero(mask)[0][index]
print(index)
print(index_of_minimum)
