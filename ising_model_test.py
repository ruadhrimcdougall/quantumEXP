#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 09:11:14 2023

@author: ruadhri
"""

#import classicalsim as sim
import numpy as np

pauli_X = np.array([[0, 1], [1, 0]])
pauli_Z = np.array([[1, 0], [0, -1]])
up = np.array([1, 0])
down = np.array([0, 1])

# one qubit
print("1st Order Pauli X on |0>")
print(np.matmul(pauli_X, up))
print()

#as expecxted up to here
#%% Two qubits 

#print(np.matmul(pauli_X, down))
up_up = np.kron(up, up)
down_down = np.kron(down, down)
up_down = np.kron(up, down)
down_up = np.kron(down, up)

pauli_X_1 = np.kron(pauli_X, np.identity(2))
pauli_X_2 = np.kron(np.identity(2), pauli_X)


print("|00>")
print(up_up)
print()
print("|11>")
print(down_down)
print()
print("|01>")
print(up_down)
print()
print("|10>")
print(down_up)
print()

#print(pauli_X_1)
#print()
#print(pauli_X_2)
#print()

print("Pauli X_1 on |00>")
print(np.matmul(pauli_X_1, up_up))
print()
print("Pauli X_2 on |00>")
print(np.matmul(pauli_X_2, up_up))
print()

print(up_up.shape)

#%%

#print(np.kron(1, pauli_X))
#print(np.matmul(sim.pauli_Xi(2, 1), up_up))

#Z1 = sim.pauli_Zi(2, 0)
#Z2 = sim.pauli_Zi(2, 1)
#Z1Z2 = np.matmul(Z1, Z2)

#print(np.matmul(Z1Z2, up_up))

#%% testing hamiltonian method

#h1 = sim.Hamiltonian(3, 1)
#print(h1.hamiltonian)
#test_hamiltonian = h1.hamiltonian


#%% Testing ground state

#ground = h1.ground_state
#print(ground)

#exp_ground = np.matmul(ground.conj().T, np.matmul(test_hamiltonian, ground))
#print(exp_ground)
