#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:16:20 2023

@author: Ruadhri McDougall
"""

Welcome quantumEXP, a library to classically simulate quantum experiments, and 
use a machine learning model to make predictive estimates about such systems. 

A UROP project with i-X, supervised by Roberto Bondesan.

Here are all the relevant files for the project.
    
1) classicalsim.py
        
        This module contains all classes and functions used for the classical
        simulation of the 1d quantum ising model

        The Hamiltonian class determines the hamiltonian, and computes the
        ground state and ground state energy, accounting for degeneracy.
    
2) ML_test.py
        
        The main test file used to trial the code in classicalsim.py
        
        Contains scripts to check and plot the expected ground state energy
        for varying magnetic field coupling coefficient values. This can easilly
        be adapted for different numbers of qubits in the ising chain.
        
        It also has a section for when I have had a play with the
        sklearn linear_learn.Lasso functionality. I have messed around with 
        varying qubit number and magnetic fields to see how this affects the 
        error on predictions.
    
3) ising_model_test.py
        
        First test script used to check that the classical model was working as
        a first pass. 
        
        (Not particularly relevant, included for record of testing).

4) degeneracy_test.py
        
        Used to verify the results for the case of degenerate ground states,
        when there is no magnetic field i.e. g=0.and
        
        (Not particularly relevant, included for record of testing).