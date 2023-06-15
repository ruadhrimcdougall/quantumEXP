#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:47:07 2023

@author: ruadhri
"""

import classicalsim as sim
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.5)
sns.set_palette('winter')
#plt.style.use('bmh')

#%% Create hamiltonians for different values of x, and plot epectation of ground
# state energy as a function of this variable

# Two qubits first
data_pts = 50
g_vals = np.linspace(-2, 2, data_pts)
q_bits = 5
E0 = []
training_data = np.zeros((data_pts*5, 2))

for q in range(q_bits):
    #training_data[:(data_pts*(q+1)-1), 0] = q+1
    #print(data_pts*(q+1))
    #print(q+1)
    for i in range(len(g_vals)):
        new_hamiltonian = sim.Hamiltonian(q+1, g_vals[i])
        exp_min_energy = new_hamiltonian.exp_ground
        E0.append(exp_min_energy)
        training_data[q*data_pts + i, 0] = q+1
        training_data[q*data_pts + i, 1] = g_vals[i]
        #print(new_hamiltonian.ground_state)
        
#training_data[:data_pts, 0] = 1
#training_data[data_pts:2*data_pts, 0] = 2
#training_data[2*data_pts:3*data_pts, 0] = 3
#training_data[3*data_pts:4*data_pts, 0] = 4
#training_data[4*data_pts:, 0] = 5
#training_data[:data_pts, 1] = g_vals
#training_data[data_pts:2*data_pts, 1] = g_vals
#training_data[2*data_pts:3*data_pts, 1] = g_vals
#training_data[3*data_pts:4*data_pts, 1] = g_vals
#training_data[4*data_pts:, 1] = g_vals

#plt.figure()
#plt.plot(g_vals, E0)
#plt.xlabel('Field Coupling Coefficient, g')
#plt.ylabel(r'$\langle \psi_0\vert\hat H\vert\psi_0\rangle$')
#plt.title(r'1D Ising $\hat{H}$ for ' + str(q_bits) + ' qubits')

#training_data = np.vstack((g_vals, E0)).T

#%% Testing out sklearn lasso

clf = linear_model.Lasso()
clf.fit(training_data, E0)
#print(clf.predict())

#%% Min. positive val test

#arr = np.array([3, -1, 5, 0, 2])
#mask = arr > 0
#values_greater_than_zero = arr[mask]
#index = np.argmin(values_greater_than_zero)
#index_of_minimum = np.nonzero(mask)[0][index]
#print(index)
#print(index_of_minimum)
