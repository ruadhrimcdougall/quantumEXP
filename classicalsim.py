#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:34:00 2023

@author: ruadhri
"""

import numpy as np

pauli_X = np.array([[0, 1], [1, 0]])
pauli_Z = np.array([[1, 0], [0, -1]])
up = np.array([1, 0])
down = np.array([0, 1])

# maybe an overarching "Operator" class, which the Hamiltonian class inherits from?

class Hamiltonian:
    
    def __init__(self, qubits, x, form='ising1d'):
        self.__qubits = qubits
        self.__x = x
        if form =='ising1d':
            self.hamiltonian = self.ising_hamiltonian()
        self.ground_state = self.find_ground_state()
        self.exp_ground = self.compute_observable(self.ground_state, 
                                                  self.hamiltonian)
        
    def ising_hamiltonian(self):
        Z_term = 0
        X_term = 0
        new_Z = self.pauli_Zi(0)
        for i in range(self.__qubits):
            Z_i = new_Z
            if self.__qubits == i+1:
                new_Z = np.zeros(new_Z.shape)
            else:
                new_Z = self.pauli_Zi(i+1)
            Z_term += np.matmul(Z_i, new_Z)
            X_term += self.pauli_Xi(i)
        total_hamiltonian = Z_term + self.__x * X_term
        return total_hamiltonian
        
    def pauli_Xi(self, i):
        if i<0 or i>=self.__qubits:
            raise ValueError('i must take integer values between 0 (for the first qubit) up to qubits-1')
        elif isinstance(i, int) == 0:
            raise TypeError('i must be an integer')
        X = 1
        for q in range(self.__qubits):
            if q == i:
                X = np.kron(X, pauli_X)
            else:
                X = np.kron(X, np.identity(2))
        return X
    
    def pauli_Zi(self, i):
        if i<0 or i>=self.__qubits:
            raise ValueError('i must take integer values between 0 (for the first qubit) up to qubits-1')
        elif isinstance(i, int) == 0:
            raise TypeError('i must be an integer')
        Z = 1
        for q in range(self.__qubits):
            if q == i:
                Z = np.kron(Z, pauli_Z)
            else:
                Z = np.kron(Z, np.identity(2))
        return Z
    
    def find_ground_state(self):
        vals, vecs = np.linalg.eig(self.hamiltonian)
        lowest_ind = np.where(vals == vals.min())[0]
        #mask = vals > 0
        #pos_vals = vals[mask]
        #min_pos_val = np.argmin(pos_vals)
        #lowest_ind = np.nonzero(mask)[0][min_pos_val]
        min_energy_state = vecs[lowest_ind,:]#.flatten()
        #print(min_energy_state.ndim)
        return min_energy_state
    
    def compute_observable(self, state, operator):
        if state.ndim == 1:
            return np.matmul(state.conj().T, 
                             np.matmul(operator, 
                                       state)
                             )
        elif state.ndim == 2:
            # not sure this is quite working, unsure
            no_states = state.shape[0]
            #inner = 0
            operator_on_state = np.dot(operator, state.T).T
            inner_prod = np.diag(np.dot(operator_on_state, state.T))
            trace = np.sum(inner_prod) / no_states
            #for i in range(no_states):
            #    inner += np.matmul(state[i].conj().T, 
            #                     np.matmul(operator, 
            #                               state[i])
            #                     )
            #trace = inner / no_states
            return trace
                
        else:
            raise ValueError('state input must be a 1d array, or a 2d array of 1d arrays')
        

class State:
    
    def __init__(self):
        self.__up = np.array([1, 0])
        self.__down = np.array([0, 1])

        
        

