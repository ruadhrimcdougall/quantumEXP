#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 10:34:00 2023

@author: ruadhri
"""

import numpy as np

# pauli spin matrix definitions
pauli_X = np.array([[0, 1], [1, 0]])
pauli_Z = np.array([[1, 0], [0, -1]])

# spin up (+1) and spin down (-1) definitions
up = np.array([1, 0])
down = np.array([0, 1])

# maybe an overarching "Operator" class, which the Hamiltonian class inherits from?

class Hamiltonian:
    
    def __init__(self, qubits, g, form='ising1d'):
        '''
        
        A classical model for the quantum hamiltonian
        
        Current functionality models the 1d quantum ising model to describe
        a chain of near-neighbour interactions.
        
        Parameters
        ----------
        qubits : TYPE int
            DESCRIPTION. The number of qubits in the monatomic chain
        g : TYPE
            DESCRIPTION.
        form : TYPE string, optional
            DESCRIPTION. The default is 'ising1d'.

        Returns
        -------
        None.

        '''
        if isinstance(qubits, int):
            self.__qubits = qubits
        else:
            raise TypeError('"qubits" field must be an integer')
        self.__g = g
        if form =='ising1d':
            self.hamiltonian = self.ising_hamiltonian()
        else:
            raise ValueError('The string "ising1d" is currently the only allowed value for "form" field')
        self.ground_state = self.find_ground_state()
        self.exp_ground = self.compute_observable(self.ground_state, 
                                                  self.hamiltonian)
        
    def ising_hamiltonian(self):
        '''
        Determines the matrix form of the 1d ising hamiltonian for "n" qubits
        
        Modeled using fixed boundary conditions (i.e. non-periodic);
        
        --------------------------------------------------------------------
        |              --- n-2                  --- n-1                    |
        |          H =  >      Z_i . Z_i+1  +    >      g . X_i            |
        |              --- i=0                  --- i=0                    |
        --------------------------------------------------------------------

        such that for the case of two qubits in the chain (n=2), we get,
        
                        H = Z_0 . Z_1 + g . (X_0 + X1)
        
        where "i" indicates the position of the qubit in the chain, "Z_i, X_i"
        are the respective pauli spin matrices, and "g" is the magnetic field
        coupling coefficient.

        Returns
        -------
        total_hamiltonian : TYPE np.ndarray
            DESCRIPTION. Matrix form of the 1d ising hamiltonian (2d numpy array)
        '''
        Z_term = 0
        X_term = 0
        new_Z = self.pauli_Zi(0)
        for i in range(self.__qubits):
            Z_i = new_Z
            if i == self.__qubits-1:
                new_Z = np.zeros(new_Z.shape)
            else:
                new_Z = self.pauli_Zi(i+1)
            Z_term += np.matmul(Z_i, new_Z)
            X_term += self.pauli_Xi(i)
        total_hamiltonian = Z_term + self.__g * X_term
        return total_hamiltonian
        
    def pauli_Xi(self, i):
        '''
        The pauli Z matix for the qubit at position i in the 1d sing chain

        Parameters
        ----------
        i : TYPE int
            DESCRIPTION. The qubit position, can only take values from 0 to n-1,
                         where n is the number of qubits.

        Raises
        ------
        ValueError
            DESCRIPTION. Error if i is outside of the 0 to n-1 range
        TypeError
            DESCRIPTION. Error if i is not an integer

        Returns
        -------
        X : TYPE np.ndarray
            DESCRIPTION. The pauli spin X matrix for a given qubit position 
                         (2d array)
        '''
        if i<0 or i>=self.__qubits:
            raise ValueError('i must take integer values between 0 (for the first qubit) up to n-1')
        elif isinstance(i, int):
            X = 1
            for q in range(self.__qubits):
                if q == i:
                    X = np.kron(X, pauli_X)
                else:
                    X = np.kron(X, np.identity(2))
            return X
        else:
            raise TypeError('i must be an integer')
    
    def pauli_Zi(self, i):
        '''
        The pauli Z matix for the qubit at position i in the 1d sing chain
        
        Parameters
        ----------
        i : TYPE int
            DESCRIPTION. The qubit position, can only take values from 0 to n-1,
                         where n is the number of qubits.

        Raises
        ------
        ValueError
            DESCRIPTION. Error if i is outside of the 0 to n-1 range
        TypeError
            DESCRIPTION. Error if i is not an integer

        Returns
        -------
        Z : TYPE np.ndarray
            DESCRIPTION. The pauli Z spin matrix for a given qubit position
                         (2d array)
        '''
        if i<0 or i>=self.__qubits:
            raise ValueError('i must take integer values between 0 (for the first qubit) up to n-1')
        elif isinstance(i, int):
            Z = 1
            for q in range(self.__qubits):
                if q == i:
                    Z = np.kron(Z, pauli_Z)
                else:
                    Z = np.kron(Z, np.identity(2))
            return Z
        else:
            raise TypeError('i must be an integer')
    
    def find_ground_state(self):
        '''
        Ground state of the hamiltonian is defined as the state(s) with the lowest
        eigenvalue.
        
        This finds the lowest eigenvalue and the corresponding state(s) for a
        complex hermitian matrix using the np.linalg.eigh function

        Returns
        -------
        min_energy_state : TYPE np.ndarray
            DESCRIPTION. A 1d array defining the ground state, or 2d for the 
                         case of degenerate ground states, interpreted as a list
                         of the 1d states.
                         
                         i.e. has shape (n_degstates, 2^{qubits}) where
                         n_degstates is the number of degenerate eigenstates
                         
                         e.g. for two qubits and a single ground state the 
                         min_energy_state shape is (4) and for two it is (2, 4).
                                    
                         Added .flatten() in 'if' statement to give output 
                         described above for single ground state (has no effect 
                         on results, but gives a slight speed-up)
        '''
        vals, vecs = np.linalg.eigh(self.hamiltonian)
        lowest_ind = np.where(vals == vals.min())[0]
        min_energy_state = vecs[:, lowest_ind].T
        if min_energy_state.shape[0] == 1:
            return min_energy_state.flatten()
        else:
            return min_energy_state
    
    def compute_observable(self, state, operator):
        '''
        Computes the expected observable for a given state and operator.
        
        Expected value is given by the inner product <state|operator|state>
        (or s - state and O - operator)
        
        Can be shown that this is equivalent to Tr(sO), which is the same as
        the sum over the inner products of each of the degenerate eigenstates 
        with the operator, divided by the number of degenerate states.
        
        Tr(sO) = (1/N) x (<s_1|O|s_1> + ... + <s_N|O|s_N>)

        Parameters
        ----------
        state : TYPE np.ndarray
            DESCRIPTION. If one degenerate state, this is a 1d array. If multiple
                         degenerate states this this is a 2d array, interpreted
                         as a list of the 1d array states.
                         
                         Has shape (n_degstates, 2^{qubits})
                         
                         (For the case of one ground state, input shapes (2^{qubits}) 
                          and (1, 2^{qubits}) give identical results)
                         
        operator : TYPE np.ndarray
            DESCRIPTION. The operator for which the observable is computed for.
                         (e.g. hamiltonian to compute energy)

        Raises
        ------
        ValueError
            DESCRIPTION. Error raised if state is not a 1d or 2d numpy array

        Returns
        -------
        TYPE
            DESCRIPTION. a constant float for the expected observable

        '''
        # Note that an ndim of 1 or 2 checked to give exact same output data
        # By flattening the single ground state to a 1d array and using the 'if' statement below gives a slight speedup
        if state.ndim == 1:
            return np.matmul(state.conj().T, 
                             np.matmul(operator, 
                                       state)
                             )
        elif state.ndim == 2:
            no_states = state.shape[0]
            operator_on_state = np.dot(operator, state.T).T
            inner_prod = np.diag(np.dot(operator_on_state, state.conj().T))
            trace = np.sum(inner_prod) / no_states
            return trace
        else:
            raise ValueError('state input must be a 1d array, or a 2d array interpretted as a list of 1d state arrays')
        

class State:
    '''
    A class for quantum states. (will probably delete)
    '''
    def __init__(self):
        self.__up = np.array([1, 0])
        self.__down = np.array([0, 1])

        
        

