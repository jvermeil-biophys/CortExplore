# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:57:51 2023

@author: JosephVermeil
"""

# %% Imports and constants

import numpy as np

# 1mm -> N45
# 3mm -> N45
# 4mm -> N42
# 5mm -> N42 or N50
# 6mm -> N42
# 7mm -> N42
# 10mm -> N42
# 12mm -> N48


# 1. Compute the reference magnetization for neodyme magnets

# DO NOT CHANGE !

µ0 = 4*np.pi*1e-7

N = 12 # Nombre d'aimants
R_ref = 55e-3/2 # m - Rayon du cercle reliant les centres des aimants
B_ref = 30e-3 # T - 
a_ref = 7e-3 # m
m_ref = B_ref * (np.pi*R_ref**3) / ((3/8) * N * µ0)
M_ref = m_ref / (a_ref**3)
print(M_ref)

# Resulting value : M_ref = 1010538.14382896 A/m
# M_ref is the reference magnetization for N42 supermagnete cubes
# By definition the magnetic moment m of an N42 magnet is m = M_ref x V (V is the cubic magnet volume)
# M the magnetization is expressed in A/m
# m the magnetic moment is expressed in A.m² = N.m/T = J/T

# %% # 2. Compute the B field in the center of a specific geometry
# %%% Halbach Spinning - 5/7 nested - OUTSIDE ring
# ID = 58.5
# MD = 66
# OD =

N = 16
a = 4e-3 # m
R = 66e-3/2 # m
m = M_ref * a**3
B = (3/8) * N * µ0 * m / (np.pi * R**3)

B = B*1000 # mT

print(B)

# %%% Halbach Spinning - 5/7 nested - INSIDE ring
# ID = 42.1
# MD = 49.5
# OD = 58.5

N = 16
a = 3e-3 # m
R = 49.5e-3/2 # m
m = M_ref * a**3
B = (3/8) * N * µ0 * m / (np.pi * R**3)

B = B*1000 # mT

print(B)

# %% 3. Compute the parameters needed to obtain a specific B field

B_goal = 5e-3 # T

# %%% 3.1 Compute R

N = 12
a = 3e-3 # m
m = M_ref * a**3

R = ((3/8) * N * µ0 * m / (np.pi * B_goal))**(1/3)

R = R*100 # cm
print(R)

# %% 3.2 Compute a

N = 12
R = 8e-2 # m

a = (B_goal * np.pi * R**3 / ((3/8) * N * µ0 * M_ref)) **(1/3)

A = a*1000 # mm
print(A)

# %% 3.3 Compute N

R = 5e-2 # m
a = 5e-3 # m
m = M_ref * a**3

N = B_goal * np.pi * R**3 / ((3/8) * µ0 * m)

print(N)