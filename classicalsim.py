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
        

class State:
    
    def __init__(self):
        self.__up = np.array([1, 0])
        self.__down = np.array([0, 1])

        
        

