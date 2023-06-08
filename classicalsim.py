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
    
    def __init__(self, qubits, form='ising1d'):
        self.__qubits = qubits
        if form =='ising1d':
            self.hamiltonian = self.ising_hamiltonian()
    
    def ising_hamiltonian(self):
        pass
        

class State:
    
    def __init__(self):
        self.__up = np.array([1, 0])
        self.__down = np.array([0, 1])

def pauli_Z(order):
    pass
        

