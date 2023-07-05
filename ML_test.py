#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:47:07 2023

@author: ruadhri
"""

import classicalsim as sim
import numpy as np
import time
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.5)
sns.set_palette('winter')
#plt.style.use('bmh')

errors_lst = []
pts_lst = []

#%% Create hamiltonians for different values of x, and plot epectation of ground
# state energy as a function of this variable

# Two qubits first
data_pts = int(1e2 + 1)
g_vals = np.linspace(-2, 2, data_pts)
q_bits = 2
E0 = []
training_data = np.zeros((data_pts, 2))
#print(training_data.shape[0])

#for q in range(q_bits):
    #training_data[:(data_pts*(q+1)-1), 0] = q+1
    #print(data_pts*(q+1))
    #print(q+1)
#    for i in range(len(g_vals)):
#        new_hamiltonian = sim.Hamiltonian(q+1, g_vals[i])
#        exp_min_energy = new_hamiltonian.exp_ground
#        E0.append(exp_min_energy)
#        training_data[q*data_pts + i, 0] = q+1
#        training_data[q*data_pts + i, 1] = g_vals[i]
        #print(new_hamiltonian.ground_state)
start = time.time()
for i in range(len(g_vals)):
    new_hamiltonian = sim.Hamiltonian(q_bits, g_vals[i])
    exp_min_energy = new_hamiltonian.exp_ground
    E0.append(exp_min_energy)
    training_data[i, 1] = g_vals[i]
    if g_vals[i] == 0:
        print(exp_min_energy)
end = time.time()
print('runtime')
print(end-start)

training_data[:, 0] = q_bits
        
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

plt.figure()
plt.plot(g_vals, E0)
plt.xlabel('Field Coupling Coefficient, g')
plt.ylabel(r'$\langle \psi_0\vert\hat H\vert\psi_0\rangle$')
plt.title(r'1D Ising $\hat{H}$ for ' + str(q_bits) + ' qubits')

#training_data = np.vstack((g_vals, E0)).T

#%% Testing out sklearn lasso

clf = linear_model.Lasso()
clf.fit(training_data, np.real(E0))
#print(clf.predict())

#%% Check errors
# predict for a random data point
random_data = np.zeros((1,2))
qubits_rand = q_bits#np.random.randint(1, q_bits)
g_rand = (np.random.rand() - 1)*4
random_data[:,0] = qubits_rand
random_data[:,1] = g_rand
test_ham = sim.Hamiltonian(qubits_rand, g_rand)
print('Parameters for test_g within training data range (of +/- 2)')
print('Qubits: '+str(qubits_rand))
print('Couping Coeff (g): '+str(g_rand))
print('No. Training Data points: '+str(data_pts))
print()

E0_pred = clf.predict(random_data)[0]
print('Predicted E0: ' + str(E0_pred))
E0_check = test_ham.exp_ground
print('Expected E0: ' + str(E0_check))
E0_diff = E0_pred - E0_check
E0_err = np.abs(E0_pred - E0_check) / np.abs(E0_pred)
print('E0 Error: ' + str(E0_err))

#%% appending to lists

errors_lst.append(E0_err)
pts_lst.append(data_pts)

#%%

plt.figure()
plt.semilogx(pts_lst, errors_lst)
plt.xlabel('Training Data Points')
plt.ylabel('% Error')
plt.title('Qubits: '+str(q_bits)+', within g-range -2 to 2')


