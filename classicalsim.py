#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:34:00 2023

@author: ruadhri
"""

import numpy as np

pauli_X = np.array([[0, 1], [1, 0]])
pauli_Z = np.array([[1, 0], [0, -1]])

class Hamiltonian:
    
    def __init__(self, qubits):
        self.__qubits = qubits