# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:47:19 2024

@author: JosephVermeil
"""

# %% Imports & Utility functions

#### Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate

from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from matplotlib import ticker

import UtilityFunctions as ufun
import GraphicStyles as gs

#### Utility functions
def my_sqrt(x):
    """
    The solver likes to put negative values in square roots.
    But then it complains and whines that there is an error.
    I want to tell him: 'It's your fault solver ! *You* put this negative number!'
    But you can't talk to a solver. So I wrote this little function.
    It prevents the number x that you wanna square root to be lower than 1e-10.
    """
    return(max(1e-10, x)**0.5)

# %% Solver functions

# %%% Solver for Parallel Plates

# %%% Solver for Spherical Indenter - 1 domain [phi <= pi/2]

def CSS_sphere_1domain(R1, H0, Rp, delta_values,
                        initial_guess = None,
                        print_tests = False):
    """
    Cell Shape Solver for a cell indented by a spherical indenter
    1 domain version <=> phi (= cell-substrate contact angle) has to stay below pi/2
    
    The initial geometry is parametrized by the 3 inputs:
        - R1 is the radius of the cell-substrate contact
          (note: R1 is constant cause the cell is assumed to be adhered on a pattern)
        - H0 is the initial height of the cell
        - Rp is the radius of the spherical probe of the indenter
    
    Here the solving is done for r1, phi:
        - r1 is the radius of the cell-indenter contact
        - phi is the cell-substrate contact angle
        -> The vectorial writing of the solution is: [r1, phi] = r1_phi
        
    The two equations solved are:
        - Set indentation: Delta_set - Delta(r1, phi) = 0
        - Volume conservation: V0 - V(r1, phi) = 0
        
    Having solved these for a given delta, one can compute all the relevant parameters:
        - Contour of the cell z(r)
        - Surface Area of the cell A 
    """
    
    #### 1. Compute other geometrical parameters in the initial state
    signe_h = (H0-R1)/np.abs(H0-R1)
    phi0 = np.arcsin(2*H0*R1 / (H0**2 + R1**2)) * (-signe_h) + np.pi * (H0>R1)
    R00 = (H0**2 + R1**2) / (2*H0)
    # phi0_deg = phi0 * 180/np.pi
    A0 = 2*np.pi*R00*H0
    V0 = (np.pi/3)*H0**2*(3*R00 - H0)
    
    #### 2. Define the functions for the solver  
    # Simple geometry
    def h_cap(r1_phi):
        """
        Height of a spherical cap of main radius Rp and base radius r1 = r1_phi[0]
        """
        return(Rp - my_sqrt(Rp**2 - r1_phi[0]**2))

    def A_cap(r1_phi):
        """
        Area of a spherical cap of main radius Rp and base radius r1 = r1_phi[0]
        """
        return(2*np.pi*Rp*(Rp - my_sqrt(Rp**2 - r1_phi[0]**2)))

    def V_cap(r1_phi):
        """
        Volume of a spherical cap of main radius Rp and base radius r1 = r1_phi[0]
        """
        return((np.pi/3) * (Rp - my_sqrt(Rp**2 - r1_phi[0]**2))**2 * (3*Rp - (Rp - my_sqrt(Rp**2 - r1_phi[0]**2))))
    

    # Shape function u and its coefficient A & B
    def A_sphere(r1_phi):
        """
        A = (R1.sin(phi) + r1²/Rp) / (R1² - r1²)
        """
        return((R1*np.sin(r1_phi[1]) + ((r1_phi[0]**2)/Rp))/(R1**2 - r1_phi[0]**2))
        
    def B_sphere(r1_phi):
        """
        B = -r1².A - r1²/Rp
        """
        return(-(r1_phi[0]**2)*(R1*np.sin(r1_phi[1]) + (r1_phi[0]**2/Rp))/(R1**2 - r1_phi[0]**2) - (r1_phi[0]**2)/Rp)

    def u_sphere(r, r1_phi):
        """
        DEFINITION
        u(r) = sin(gamma(r))
        with gamma(r) the angle between the vertical axis 
        and the vector normal to the cell surface at location r.
        
        FUNDAMENTAL EQUATION
        u(r) is the relevant shape function, which verifies:
        dP/T = C1 + C2 = du/dr + u/r
        where dP is the pressure difference, T is the (uniform) tension 
        and C1 & C2 are the (uniform) principal curvatures.
        
        SOLUTION OF THE FUNDAMENTAL EQUATION
        u(r) = A.r + B/r
        where A & B are defined above
        """
        return((A_sphere(r1_phi)*r) + (B_sphere(r1_phi)/r))
    

    # Altitude of the peeling point z1, Indentation delta & their common integrand
    def integrand_z_sphere(r, r1_phi):
        return(u_sphere(r, r1_phi) / (my_sqrt(1 - u_sphere(r, r1_phi)**2)))

    def z1_sphere_fun(r1_phi):
        """
        z1 = integral from r1 to R1 of integrand_z_sphere
        """
        return(integrate.quad(integrand_z_sphere, r1_phi[0], R1, args=r1_phi, limit = 300, maxp1 = 300)[0])

    def delta_sphere_fun(r1_phi):
        """
        delta = H0 - z1 + h_cap
        """
        return(H0 - z1_sphere_fun(r1_phi) + h_cap(r1_phi))
    
    
    # Volume V and its integrand
    def integrand_V_sphere(r, r1_phi):
        return(r**2 * (u_sphere(r, r1_phi) / (my_sqrt(1 - u_sphere(r, r1_phi)**2))))

    def V_sphere_fun(r1_phi):
        """
        V = integral from r1 to R1 of integrand_V_sphere
        """
        return(np.pi * integrate.quad(integrand_V_sphere, r1_phi[0], R1, args=r1_phi, limit = 300, maxp1 = 300)[0] - V_cap(r1_phi))
    
    
    #### OPTIONAL TESTS
    if print_tests:
        r1_phi_test = [Rp*0.3, phi0+0.1]
        print("V_cap", V_cap(r1_phi_test))
        print("V_sphere_fun", V_sphere_fun(r1_phi_test))
        print("")
        print("delta_sphere_fun", delta_sphere_fun(r1_phi_test))
        print("z1_sphere_fun", z1_sphere_fun(r1_phi_test))
    
    
    #### 3. Solve shit
    N = len(delta_values)
    r1_phi_sol = np.zeros((N, 2))
    error_list = np.ones(N)
    
    if initial_guess == None:
        r1_phi_guess = [0.1, phi0]
    else:
        r1_phi_guess = initial_guess

    for i in range(N):
        delta = delta_values[i]
        
        def delta_sphere_0(r1_phi):
            return(delta - delta_sphere_fun(r1_phi))
        
        def V_sphere_0(r1_phi):
            return(V0 - V_sphere_fun(r1_phi))
        
        def fun_solve(r1_phi):
            return(delta_sphere_0(r1_phi), V_sphere_0(r1_phi))
        
        r1_phi_sol[i], infodict, error_list[i], mesg = fsolve(fun_solve, r1_phi_guess, full_output = True)
        r1_phi_guess = r1_phi_sol[i]
        
    
    #### 4. Define the functions for the outputs
    def integrand_A_sphere(r, r1_phi):
        return(r / ((1-u_sphere(r, r1_phi)**2)**0.5))

    def Area_sphere_fun(r1_phi):
        """
        A = 2*pi * integral from r1 to R1 of integrand_A_sphere
        """
        return(2*np.pi*integrate.quad(integrand_A_sphere, r1_phi[0], R1, args=r1_phi, limit = 300, maxp1 = 300)[0] + A_cap(r1_phi))

    def Curvature(r1_phi):
        return(A_sphere(r1_phi))

    # def z_sphere_fun(r, r1_phi):
    #     """
    #     z(r) = integral from r to R1 of integrand_z_sphere
    #     """
    #     return(integrate.quad(integrand_z_sphere, r, R1, args=r1_phi, limit = 300, maxp1 = 300)[0])
    
    #### 5. Give the results
    area = np.array([Area_sphere_fun(r1_phi) for r1_phi in r1_phi_sol])
    alpha = np.array([(A-A0)/A0 for A in area])
    curvature = np.array([Curvature(r1_phi) for r1_phi in r1_phi_sol])
    R0_blank = R1*np.ones(N)
    case_array = np.array(['1domain'] * N)
    
    res = {'R1':R1,
           'H0':H0,
           'Rp':Rp,
           'phi0':phi0,
           'V0':V0,
           'A0':A0,
           'delta':delta_values,
           'r1':r1_phi_sol[:,0],
           'R0':R0_blank,
           'phi':r1_phi_sol[:,1],
           'area':area,
           'alpha':alpha,
           'curvature':curvature,
           'case':case_array,
           'error':error_list-1,
           }
    
    return(res)


# Test
# R1 = 9
# H0 = 5.5
# Rp = 3
# N_points = 200

# # delta_values = np.linspace(0, H0/3, N_points) + 0.01
# delta_values = np.array([])
# res = CSS_sphere_1domain(R1, H0, Rp, delta_values)





# %%% Solver for Spherical Indenter - 2 domains [phi > pi/2]

def CSS_sphere_2domains(R1, H0, Rp, delta_values,
                        initial_guess = None,
                        print_tests = False):
    """
    Cell Shape Solver for a cell indented by a spherical indenter
    2 domains version <=> phi (= cell-substrate contact angle) has to stay above pi/2
    
    The initial geometry is parametrized by the 3 inputs:
        - R1 is the radius of the cell-substrate contact
          (note: R1 is constant cause the cell is assumed to be adhered on a pattern)
        - H0 is the initial height of the cell
        - Rp is the radius of the spherical probe of the indenter
    
    Here the solving is done for r1, R0, phi:
        - r1 is the radius of the cell-indenter contact
        - R0 is the maximum radius of the cell in the plane normal to the indentation
        - phi is the cell-substrate contact angle
        -> The vectorial writing of the solution is: [r1, R0, phi] = rRp
        
    The two equations solved are:
        - Set indentation: Delta_set - Delta(r1, R0, phi) = 0
        - Volume conservation: V0 - V(r1, R0, phi) = 0
        
    Having solved these for a given delta, one can compute all the relevant parameters:
        - Contour of the cell z(r)
        - Surface Area of the cell A 
    """
    
    #### 1. Compute other geometrical parameters in the initial state
    signe_h = (H0-R1)/np.abs(H0-R1)
    phi0 = np.arcsin(2*H0*R1 / (H0**2 + R1**2)) * (-signe_h) + np.pi * (H0>R1)
    R00 = (H0**2 + R1**2) / (2*H0)
    # phi0_deg = phi0 * 180/np.pi
    A0 = 2*np.pi*R00*H0
    V0 = (np.pi/3)*H0**2*(3*R00 - H0)
    
    #### 2. Define the functions for the solver    
    # Simple geometry
    def h_cap(rRp):
        """
        Height of a spherical cap of main radius Rp and base radius r1 = rRp[0]
        """
        return(Rp - my_sqrt(Rp**2 - rRp[0]**2))

    def A_cap(rRp):
        """
        Area of a spherical cap of main radius Rp and base radius r1 = rRp[0]
        """
        return(2*np.pi*Rp*(Rp - my_sqrt(Rp**2 - rRp[0]**2)))

    def V_cap(rRp):
        """
        Volume of a spherical cap of main radius Rp and base radius r1 = rRp[0]
        """
        return((np.pi/3) * (Rp - my_sqrt(Rp**2 - rRp[0]**2))**2 * (3*Rp - (Rp - my_sqrt(Rp**2 - rRp[0]**2))))


    #### Domaine 1 (above equator)
    # Shape function u and its coefficient A & B
    def A1_sphere(rRp):
        """
        A1 = (R0 + r1²/Rp) / (R0² - r1²)
        """
        return((rRp[1] + ((rRp[0]**2)/Rp))/(rRp[1]**2 - rRp[0]**2))
        
    def B1_sphere(rRp):
        """
        B1 = -r1².A1 - r1²/Rp
        """
        return(-(rRp[0]**2) * (rRp[1] + ((rRp[0]**2)/Rp))/(rRp[1]**2 - rRp[0]**2) - (rRp[0]**2)/Rp)

    def u1_sphere(r, rRp):
        """
        DEFINITION
        u1(r) = sin(gamma(r))
        with gamma(r) the angle between the vertical axis 
        and the vector normal to the cell surface at location r.
        
        FUNDAMENTAL EQUATION
        u1(r) is the relevant shape function, which verifies:
        dP/T = C1 + C2 = du1/dr + u1/r
        where dP is the pressure difference, T is the (uniform) tension 
        and C1 & C2 are the (uniform) principal curvatures.
        
        SOLUTION OF THE FUNDAMENTAL EQUATION
        u1(r) = A1.r + B1/r
        where A1 & B1 are defined above
        """
        return((A1_sphere(rRp)*r) + (B1_sphere(rRp)/r))

    # Altitude of the peeling point above the equatorial plane = z1 and its integrand
    def integrand_z1(r, rRp):
        return(u1_sphere(r, rRp) / (my_sqrt(1 - u1_sphere(r, rRp)**2)))

    def z1_fun(rRp):
        """
        z1 = integral from r1 to R0 of integrand_z1
        """
        return(integrate.quad(integrand_z1, rRp[0], rRp[1], args=rRp, points=[rRp[0], rRp[1]], 
                              limit = 300, maxp1 = 300)[0])
    
    # Volume V and its integrand
    def integrand_V1(r, rRp):
        return(r**2 * (u1_sphere(r, rRp) / (my_sqrt(1 - u1_sphere(r, rRp)**2))))

    def V1_fun(rRp):
        """
        V1 = integral from r1 to R0 of integrand_V1
        """
        return(np.pi * integrate.quad(integrand_V1, rRp[0], rRp[1], args=rRp, points=[rRp[0], rRp[1]], 
                                      limit = 300, maxp1 = 300)[0])


    #### Domaine 2 (below equator)
    def A2_sphere(rRp):
        """
        A2 = (R0 - R1*sin(phi)) / (R0² - R1²)
        """
        return((rRp[1] - (R1*np.sin(rRp[2])))/(rRp[1]**2 - R1**2))
        
    def B2_sphere(rRp):
        """
        B2 = -R0².A2 + R0
        """
        return(-(rRp[1]**2) * ((rRp[1] - (R1*np.sin(rRp[2])))/(rRp[1]**2 - R1**2)) + rRp[1])

    def u2_sphere(r, rRp):
        """
        DEFINITION
        u2(r) = sin(gamma(r))
        with gamma(r) the angle between the vertical axis 
        and the vector normal to the cell surface at location r.
        
        FUNDAMENTAL EQUATION
        u2(r) is the relevant shape function, which verifies:
        dP/T = C1 + C2 = du2/dr + u2/r
        where dP is the pressure difference, T is the (uniform) tension 
        and C1 & C2 are the (uniform) principal curvatures.
        
        SOLUTION OF THE FUNDAMENTAL EQUATION
        u2(r) = A2.r + B2/r
        where A2 & B2 are defined above
        """
        return((A2_sphere(rRp)*r) + (B2_sphere(rRp)/r))
    
    # Altitude of the equatorial plane = z2 and its integrand
    def integrand_z2(r, rRp):
        return(u2_sphere(r, rRp) / (my_sqrt(1 - u2_sphere(r, rRp)**2)))

    def z2_fun(rRp):
        """
        z2 = integral from R1 to R0 of integrand_z2
        """
        return(integrate.quad(integrand_z2, R1, rRp[1], args=rRp, limit = 300, maxp1 = 300)[0])
    
    # Volume V and its integrand
    def integrand_V2(r, rRp):
        return(r**2 * (u2_sphere(r, rRp) / (my_sqrt(1 - u2_sphere(r, rRp)**2))))

    def V2_fun(rRp):
        """
        V2 = integral from R1 to R0 of integrand_V2
        """
        return(np.pi * integrate.quad(integrand_V2, R1, rRp[1], args=rRp, limit = 300, maxp1 = 300)[0])


    #### Total
    def ztot_fun(rRp):
        """
        ztot is the total altitude of the peeling point
        ztot = z1 + z2
        """
        return(z1_fun(rRp) + z2_fun(rRp))

    def delta_sphere_fun(rRp):
        """
        delta = H0 - ztot + h_cap
        """
        return(H0 - ztot_fun(rRp) + h_cap(rRp))

    def Vtot_sphere_fun(rRp):
        """
        Vtot = V1 + V2 - V_cap
        """
        return(V1_fun(rRp) + V2_fun(rRp) - V_cap(rRp))

    def junction_equation(rRp):
        """
        This equation translate the fact that the curvature are the same in both domains.
        Hence, for r = R0: 
            du1/dr + u1/r = C1 + C2 = du2/dr + u2/r
        Substituting u1 and u2 by their expressions, you get:
            A1 = A2
        This 'junction equation' state simply that A1(r1, R0, phi) - A2(r1, R0, phi) = 0
        """
        return(((rRp[1] - R1*np.sin(rRp[2]))/(rRp[1]**2 - R1**2)) - ((rRp[1] + (rRp[0]**2)/Rp)/(rRp[1]**2 - rRp[0]**2)))
    
    
    #### OPTIONAL TESTS
    if print_tests:
        rRp_test = [1, R00 + 0.1, phi0+1]
        print("V_cap", V_cap(rRp_test))
        print("")
        print("z1_fun", z1_fun(rRp_test))
        print("V1_fun", V1_fun(rRp_test))
        print("")
        print("z2_fun", z2_fun(rRp_test))
        print("V2_fun", V2_fun(rRp_test))
        print("")
        print("delta_sphere_fun", delta_sphere_fun(rRp_test))
        print("Vtot_sphere_fun", Vtot_sphere_fun(rRp_test))
    
    
    #### 3. Solve shit
    N = len(delta_values)
    rRp_sol = np.zeros((N, 3))
    error_list = np.ones(N)
    
    if initial_guess == None:
        rRp_guess = [0.1, R00+0.01, phi0+0.001]
    else:
        rRp_guess = initial_guess
        print('imported guess')

    for i in range(N):
        delta = delta_values[i]
        
        def delta_fun_0(rRp):
            return(delta - delta_sphere_fun(rRp))
        
        def V_fun_0(rRp):
            return(V0 - Vtot_sphere_fun(rRp))
        
        def fun_solve(rRp):
            return(delta_fun_0(rRp), V_fun_0(rRp), junction_equation(rRp))
        
        rRp_sol[i], infodict, error_list[i], mesg = fsolve(fun_solve, rRp_guess, full_output = True) 
        rRp_guess = rRp_sol[i]
        
    
    #### 4. Define the functions for the outputs
    def integrand_A1_sphere(r, rRp):
        return(r / (my_sqrt(1-u1_sphere(r, rRp)**2)))

    def Area1_sphere_fun(rRp):
        """
        A1 = 2*pi * integral from r1 to R0 of integrand_A1_sphere
        """
        return(2*np.pi*integrate.quad(integrand_A1_sphere, rRp[0], rRp[1], args=rRp, limit = 300, maxp1 = 300)[0])

    def integrand_A2_sphere(r, rRp):
        return(r / (my_sqrt(1-u2_sphere(r, rRp)**2)))

    def Area2_sphere_fun(rRp):
        """
        A2 = 2*pi * integral from R1 to R0 of integrand_A2_sphere
        """
        return(2*np.pi*integrate.quad(integrand_A2_sphere, R1, rRp[1], args=rRp, limit = 300, maxp1 = 300)[0])

    def Areatot_sphere_fun(rRp):
        """
        Atot = A1 + A2 + A_cap
        """
        return(Area1_sphere_fun(rRp) + Area2_sphere_fun(rRp) + A_cap(rRp))
    
    def Curvature(rRp):
        return(A1_sphere(rRp))

    
    #### 5. Give the results
    area = np.array([Areatot_sphere_fun(rRp) for rRp in rRp_sol])
    alpha = np.array([(A-A0)/A0 for A in area])
    curvature = np.array([Curvature(rRp) for rRp in rRp_sol])
    case_array = np.array(['2domains'] * N)
    
    res = {'R1':R1,
           'H0':H0,
           'Rp':Rp,
           'phi0':phi0,
           'V0':V0,
           'A0':A0,
           'delta':delta_values,
           'r1':rRp_sol[:,0],
           'R0':rRp_sol[:,1],
           'phi':rRp_sol[:,2],
           'area':area,
           'alpha':alpha,
           'curvature':curvature,
           'case':case_array,
           'error':error_list-1,
           }
    
    return(res)

# R1 = 9.715
# H0 = 13
# Rp = 26.5
# delta_values = np.linspace(0, H0/3, 200) + 0.01

# res = CSS_sphere_2domains(R1, H0, Rp, delta_values)

# %%% Solver for Spherical Indenter - general case (calls the two above)

def CSS_sphere_limitCase(R1, H0, Rp):
    """
    Cell Shape Solver for a cell indented by a spherical indenter
    Limit case version <=> phi = pi/2
    
    In short: this is equivalent to the 1 domain solver, 
    where phi is no longer unknown but fixed at pi/2
    
    The initial geometry is parametrized by the 3 inputs:
        - R1 is the radius of the cell-substrate contact
          (note: R1 is constant cause the cell is assumed to be adhered on a pattern)
        - H0 is the initial height of the cell
        - Rp is the radius of the spherical probe of the indenter
    
    Here the solving is done for r1 only:
        - r1 is the radius of the cell-indenter contact
        - phi, the cell-substrate contact angle, is by definition equal to pi/2 in this case.
        
    The one equation solved is:
        - Volume conservation: V0 - V(r1, phi) = 0
        
    Having solved these for a given delta, one can compute all the relevant parameters:
        - Value of delta_lim, the indentation for which this limit case is reached
        - Contour of the cell z(r)
        - Surface Area of the cell A 
    """
    #### 1. Compute parameters
    signe_H = (H0-R1)/np.abs(H0-R1)
    phi0 = np.arcsin(2*H0*R1 / (H0**2 + R1**2)) * (-signe_H) + np.pi * (H0>R1)
    R00 = (H0**2 + R1**2) / (2*H0)
    A0 = 2*np.pi*R00*H0
    V0 = (np.pi/3)*H0**2*(3*R00 - H0)
    
    
    #### 2. Define functions
    # Geometry
    def h_cap(r1):
        return(Rp - my_sqrt(Rp**2 - r1**2))

    def A_cap(r1):
        return(2*np.pi*Rp*(Rp - my_sqrt(Rp**2 - r1**2)))

    def V_cap(r1):
        return((np.pi/3) * (Rp - my_sqrt(Rp**2 - r1**2))**2 * (3*Rp - (Rp - my_sqrt(Rp**2 - r1**2))))
    
    # Shape function    
    def A_lim(r1):
        return((R1 + ((r1**2)/Rp))/(R1**2 - r1**2))
        
    def B_lim(r1):
        return(-(r1**2)*(R1 + ((r1**2)/Rp))/(R1**2 - r1**2) - (r1**2)/Rp)

    def u_lim(r, r1):
        return((A_lim(r1)*r) + (B_lim(r1)/r))
    
    # z and V    
    def integrand_z_lim(r, r1):
        return(u_lim(r, r1) / (my_sqrt(1 - u_lim(r, r1)**2)))

    def integrand_V_lim(r, r1):
        return(r**2 * (u_lim(r, r1) / (my_sqrt(1 - u_lim(r, r1)**2))))

    def z_lim_fun(r1):
        return(integrate.quad(integrand_z_lim, r1, R1, args=r1, limit = 300, maxp1 = 300)[0])

    def V_lim_fun(r1):
        return(np.pi * integrate.quad(integrand_V_lim, r1, R1, args=r1, limit = 300, maxp1 = 300)[0] - V_cap(r1))
    
    # def z_lim_fun2(r, r1):
    #     return(integrate.quad(integrand_z_lim, r, R1, args=r1)[0])
    
    
    #### 3. Solve shit
    r1_lim = 0
    r1_guess = R1/2
    def V_lim_0(r1):
        return(V0 - V_lim_fun(r1))
    [r1_lim], infodict, ier, mesg = fsolve(V_lim_0, r1_guess, full_output = True)
    
    
    #### 4. Define the functions for the outputs
    def delta_lim_fun(r1):
        return(H0 - integrate.quad(integrand_z_lim, r1, R1, args=r1, limit = 300, maxp1 = 300)[0] + h_cap(r1))
    
    def integrand_A_sphere(r, r1):
        return(r / (my_sqrt(1-u_lim(r, r1)**2)))

    def Area_lim_sphere_fun(r1):
        return(2*np.pi*integrate.quad(integrand_A_sphere, r1, R1, args=r1, limit = 300, maxp1 = 300)[0] + A_cap(r1))
    
    def Curvature(r1):
        return(A_lim(r1))

    #### 5. Make results
    delta_lim = delta_lim_fun(r1_lim)
    area_lim = Area_lim_sphere_fun(r1_lim)
    alpha_lim = (area_lim-A0)/A0
    curvature_lim = Curvature(r1_lim)
    
    res_lim = {'R1':R1,
               'H0':H0,
               'Rp':Rp,
               'phi0':phi0,
               'V0':V0,
               'A0':A0,
               'delta':delta_lim,
               'r1':r1_lim,
               'R0':R1,
               'phi':np.pi/2,
               'area':area_lim,
               'alpha':alpha_lim,
               'curvature':curvature_lim,
               'case':'limit',
               'error':ier-1,
               }
    
    return(delta_lim, res_lim)



def CSS_sphere_general(R1, H0, Rp, delta_values):
    #### 1. Compute other geometrical parameters in the initial state
    signe_h = (H0-R1)/np.abs(H0-R1)
    phi0 = np.arcsin(2*H0*R1 / (H0**2 + R1**2)) * (-signe_h) + np.pi * (H0>R1)
    # R00 = (H0**2 + R1**2) / (2*H0)
    # phi0_deg = phi0 * 180/np.pi
    # A0 = 2*np.pi*R00*H0
    # V0 = (np.pi/3)*H0**2*(3*R00 - H0)
    
    limit_reached = False
    initial_guess1 = None
    initial_guess2 = None
    
    #### 2. Figure out which algorithm to use for
    # if phi0 > pi/2, always 2 domains
    if phi0 > np.pi/2:
        delta_values_1 = np.array([])
        delta_values_2 = delta_values
        
    else:
        delta_lim, res_lim = CSS_sphere_limitCase(R1, H0, Rp)
        # print(res_lim)
        
        # if phi < pi/2 for all delta, always 1 domain
        if delta_lim > np.max(delta_values):
            delta_values_1 = delta_values
            delta_values_2 = np.array([])
        
        # if the delta_lim is reached, there is a switch between the two
        else:
            delta_values_1 = delta_values[delta_values <=delta_lim]
            delta_values_2 = delta_values[delta_values > delta_lim]
            limit_reached = True
            initial_guess2 = [res_lim['r1']+0.01, res_lim['R0']+0.01, res_lim['phi']+0.01] # r1, R0, phi
            
    #### 3. Run the algo on their respective array of deltas
    print(gs.GREEN + "\nStart Solving" + gs.NORMAL)
    print(gs.CYAN + f"Phase 1 - {len(delta_values_1):.0f} deltas" + gs.RED)
    res1 = CSS_sphere_1domain(R1, H0, Rp, delta_values_1,
                              initial_guess = initial_guess1)
    print(gs.BRIGHTORANGE + "\n" + f"Phase 2 - {len(delta_values_2):.0f} deltas" + gs.RED)
    res2 = CSS_sphere_2domains(R1, H0, Rp, delta_values_2,
                               initial_guess = initial_guess2)
    print(gs.NORMAL + '\n')
    
    res_dict = {**res1}
    if limit_reached:
        for k in res_dict.keys():
            try:
                res_dict[k] = np.concatenate((res_dict[k], [res_lim[k]]))
            except:
                pass
    for k in res_dict.keys():
        try:
            res_dict[k] = np.concatenate((res_dict[k], res2[k]))
        except:
            pass
        
    table_cols = ['delta', 'r1', 'R0', 'phi', 'area', 'alpha', 'curvature', 'case', 'error']
    res_df = pd.DataFrame({k:res_dict[k] for k in table_cols})
    
    return(res_dict, res_df)



# %% Plot functions

def plot_contours(res_dict, res_df,
                  initial_contour = True, limit_contour = True, deltas_contour = True,
                  save = False, save_path=''):
    
    [R1, H0, Rp, phi0, V0, A0] = [res_dict[k] for k in ['R1', 'H0', 'Rp', 'phi0', 'V0', 'A0']]
    R00 = (H0**2 + R1**2) / (2*H0)
    
    figs, fig_names = [], []
    
    if initial_contour:
        CenterCell = H0 - R00
        AnglesCell = np.linspace(np.pi/2 - phi0, np.pi/2 + phi0, 200)
        RRCell = R00 * np.cos(AnglesCell)
        ZZCell = R00 * np.sin(AnglesCell) + CenterCell
        
        AnglesNotCell = np.linspace(-(3/2)*np.pi + phi0, np.pi/2 - phi0, 200)
        RRNotCell = R00 * np.cos(AnglesNotCell)
        ZZNotCell = R00 * np.sin(AnglesNotCell) + CenterCell

        CenterIndenter = H0 + Rp
        AnglesIndenter = np.linspace(0, 2*np.pi, 200)
        RRIndenter = Rp * np.cos(AnglesIndenter)
        ZZIndenter = Rp * np.sin(AnglesIndenter) + CenterIndenter

        fig_i, ax_i = plt.subplots(1,1, figsize = (8, 8))
        suptitle = 'Cell contour in the initial geometry' + '\n'
        suptitle += f'$R_1$ = {R1:.1f}, $H_0$ = {H0:.1f}, $R_p$ = {Rp:.1f} (µm)'
        fig_i.suptitle(suptitle, size = 14)
        
        ax = ax_i
        ax.axhline(0, c='k', lw=5)
        ax.plot(RRCell, ZZCell, 'b-')
        ax.plot(RRIndenter, ZZIndenter, 'r-')
            
        ax.grid()
        ax.set_xlabel('r (µm)')
        ax.set_ylabel('z (µm)')
        ax.axis('equal')
        w = max(R1, H0)
        ax.set_xlim([-w-0.5, +w+0.5])
        ax.set_ylim([H0-w-0.5, H0+w+0.5])
        
        fig_i.tight_layout()
        figs.append(fig_i)
        fig_names.append('contour_init')
    
    if limit_contour and ('limit' in res_df['case'].values):
        r1_lim = res_df.loc[res_df['case'] == 'limit', 'r1'].values[0]
        delta_lim = res_df.loc[res_df['case'] == 'limit', 'delta'].values[0]
        
        def A_lim(r1):
            return((R1 + ((r1**2)/Rp))/(R1**2 - r1**2))
            
        def B_lim(r1):
            return(-(r1**2)*(R1 + ((r1**2)/Rp))/(R1**2 - r1**2) - (r1**2)/Rp)

        def u_lim(r, r1):
            return((A_lim(r1)*r) + (B_lim(r1)/r))
        
        def integrand_z_lim(r, r1):
            return(u_lim(r, r1) / (my_sqrt(1 - u_lim(r, r1)**2)))
        
        def z_lim_fun(r, r1):
            return(integrate.quad(integrand_z_lim, r, R1, args=r1)[0])
        
        rr = np.linspace(r1_lim, R1, num=200)
        zz = np.array([z_lim_fun(r, r1_lim) for r in rr])
        
        CenterCell = H0 - R00
        AnglesCell = np.linspace(np.pi/2 - phi0, np.pi/2 + phi0, 200)
        RRCell = R00 * np.cos(AnglesCell)
        ZZCell = R00 * np.sin(AnglesCell) + CenterCell
        
        CenterIndenter = H0 + Rp - delta_lim
        AnglesIndenter = np.linspace(0, 2*np.pi, 100)
        RRIndenter = Rp * np.cos(AnglesIndenter)
        ZZIndenter = Rp * np.sin(AnglesIndenter) + CenterIndenter
        
        fig_lim, ax_lim = plt.subplots(1, 1, figsize = (8, 8))
        suptitle = 'Cell contour in the limit geometry' + '\n'
        suptitle += f'$r_1$ = {r1_lim:.2f}, $\\delta_l$ = {delta_lim:.2f} (µm)'
        fig_lim.suptitle(suptitle, size = 14)
        
        ax = ax_lim
        ax.axhline(0, c='k', lw=5)
        ax.plot(rr, zz, 'b-')
        ax.plot(-rr, zz, 'b-')
        ax.plot(RRCell, ZZCell, 'b--', lw=0.8)
        ax.plot(RRIndenter, ZZIndenter, 'r-')
        
        ax.grid()
        ax.set_xlabel('r (µm)')
        ax.set_ylabel('z (µm)')
        ax.axis('equal')
        w = max(R1, H0)
        ax.set_xlim([-w-0.5, +w+0.5])
        ax.set_ylim([H0-w-0.5, H0+w+0.5])
        
        fig_lim.tight_layout()
        figs.append(fig_lim)
        fig_names.append('contour_limit')
        
    if deltas_contour:
        N = len(res_df['delta'])
        
        # ax.set_title('Cell contours for various $\\xi = z_p / R_c$')
        # fig.suptitle("Reproduction of the method by A. Janshoff et al. (PRL 2020)", size = 12)
        # ax.set_xlabel('$r$ / $R_c$')
        # ax.set_ylabel('$z$ / $R_c$')
        
        Nplots = 10
        Ncols = 5
        Nrows = ((Nplots-1)//Ncols) + 1
        fig_d, axes_dM = plt.subplots(Nrows, Ncols, figsize = (3.5*Ncols, 3.5*Nrows))
        suptitle = 'Cell contours for growing values of $\\delta$' + '\n'
        suptitle += f'$R_1$ = {R1:.1f}, $H_0$ = {H0:.1f}, $R_p$ = {Rp:.1f} (µm)'
        fig_d.suptitle(suptitle, size = 14)
        axes_d = axes_dM.flatten()
        cList = plt.cm.viridis(np.linspace(0, 0.9, Nplots))
        index_df = np.linspace(0, N-1, Nplots, dtype=int)
        for i in range(Nplots):
            ax = axes_d[i]
            c = cList[i]
            ii = index_df[i]
            
            [delta, r1, R0, phi, solver_case] = res_df.loc[ii, ['delta', 'r1', 'R0', 'phi', 'case']].values
            
            if solver_case == '1domain':
                # Shape function u and its coefficient A & B
                def A_sphere(r1_phi):
                    return((R1*np.sin(r1_phi[1]) + ((r1_phi[0]**2)/Rp))/(R1**2 - r1_phi[0]**2))
                    
                def B_sphere(r1_phi):
                    return(-(r1_phi[0]**2)*(R1*np.sin(r1_phi[1]) + (r1_phi[0]**2/Rp))/(R1**2 - r1_phi[0]**2) - (r1_phi[0]**2)/Rp)

                def u_sphere(r, r1_phi):
                    return((A_sphere(r1_phi)*r) + (B_sphere(r1_phi)/r))
                
                def integrand_z_sphere(r, r1_phi):
                    return(u_sphere(r, r1_phi) / (my_sqrt(1 - u_sphere(r, r1_phi)**2)))

                def z_sphere_fun(r, r1_phi):
                    return(integrate.quad(integrand_z_sphere, r, R1, args=r1_phi, limit = 300, maxp1 = 300)[0])

                r1_phi_ii = [r1, phi]
                
                rr = np.linspace(r1, R1, num=200)
                zz = np.array([z_sphere_fun(r, r1_phi_ii) for r in rr])
            

            elif solver_case == '2domains':
                #### Domaine 1 (above equator)
                def A1_sphere(rRp):
                    return((rRp[1] + ((rRp[0]**2)/Rp))/(rRp[1]**2 - rRp[0]**2))
                    
                def B1_sphere(rRp):
                    return(-(rRp[0]**2) * (rRp[1] + ((rRp[0]**2)/Rp))/(rRp[1]**2 - rRp[0]**2) - (rRp[0]**2)/Rp)

                def u1_sphere(r, rRp):
                    return((A1_sphere(rRp)*r) + (B1_sphere(rRp)/r))

                def integrand_z1(r, rRp):
                    return(u1_sphere(r, rRp) / (my_sqrt(1 - u1_sphere(r, rRp)**2)))

                def z1_fun(r, rRp):
                    return(integrate.quad(integrand_z1, r, rRp[1], args=rRp, points=[rRp[0], rRp[1]], 
                                          limit = 300, maxp1 = 300)[0])
                
                #### Domaine 2 (below equator)
                def A2_sphere(rRp):
                    return((rRp[1] - (R1*np.sin(rRp[2])))/(rRp[1]**2 - R1**2))
                    
                def B2_sphere(rRp):
                    return(-(rRp[1]**2) * ((rRp[1] - (R1*np.sin(rRp[2])))/(rRp[1]**2 - R1**2)) + rRp[1])

                def u2_sphere(r, rRp):
                    return((A2_sphere(rRp)*r) + (B2_sphere(rRp)/r))
                
                def integrand_z2(r, rRp):
                    return(u2_sphere(r, rRp) / (my_sqrt(1 - u2_sphere(r, rRp)**2)))

                def z2_fun(r, rRp):
                    return(integrate.quad(integrand_z2, R1, r, args=rRp, limit = 300, maxp1 = 300)[0])
                
                rRp_ii = [r1, R0, phi]
                Z0 = z2_fun(R0, rRp_ii)
                
                rr1 = np.linspace(r1, R0, num=100)
                zz1 = np.array([z1_fun(r, rRp_ii) + Z0 for r in rr1])
                
                rr2 = np.linspace(R0, R1, num=100)
                zz2 = np.array([z2_fun(r, rRp_ii) for r in rr2])
                
                rr = np.concatenate((rr1, rr2))
                zz = np.concatenate((zz1, zz2))
                
                
            elif solver_case == 'limit':
                def A_lim(r1):
                    return((R1 + ((r1**2)/Rp))/(R1**2 - r1**2))
                    
                def B_lim(r1):
                    return(-(r1**2)*(R1 + ((r1**2)/Rp))/(R1**2 - r1**2) - (r1**2)/Rp)

                def u_lim(r, r1):
                    return((A_lim(r1)*r) + (B_lim(r1)/r))
                
                def integrand_z_lim(r, r1):
                    return(u_lim(r, r1) / (my_sqrt(1 - u_lim(r, r1)**2)))
                
                def z_lim_fun(r, r1):
                    return(integrate.quad(integrand_z_lim, r, R1, args=r1)[0])
                
                rr = np.linspace(r1_lim, R1, num=200)
                zz = np.array([z_lim_fun(r, r1_lim) for r in rr])
            
            ax.axhline(0, c='k', lw=5)
            ax.plot(rr, zz, c = c, zorder = 6, label = f'$\\delta$ = {delta:.2f} µm')
            ax.plot(-rr, zz, c = c, zorder = 6)
            
            CenterCell = H0 - R00
            AnglesCell = np.linspace(np.pi/2 - phi0, np.pi/2 + phi0, 200)
            RRCell = R00 * np.cos(AnglesCell)
            ZZCell = R00 * np.sin(AnglesCell) + CenterCell
            ax.plot(RRCell, ZZCell, 'k--', lw=0.6)
            
            CenterIndenter = H0 + Rp - delta
            AnglesIndenter = np.linspace(0, 2*np.pi, 100)
            RRIndenter = Rp * np.cos(AnglesIndenter)
            ZZIndenter = Rp * np.sin(AnglesIndenter) + CenterIndenter
            ax.plot(RRIndenter, ZZIndenter, 'r-')
            
        for ax in axes_d:
            ax.legend(loc = 'upper right', fontsize = 9)
            locator = ticker.MultipleLocator(5)
            ax.tick_params(axis='both', labelsize=8)
            ax.grid()
            ax.set_aspect('equal', adjustable='box')
            w = max(R1, H0)
            ax.set_ylim([-1, +2*w+1])
            ax.set_xlim([-w-1, +w+1])
            
        
        fig_d.tight_layout()
        figs.append(fig_d)
        fig_names.append('contour_deltas')
    
    if save:
        for fig, name in zip(figs, fig_names):
            ufun.archiveFig(fig, name = name, figDir = save_path)

    plt.show()



def plot_curves(res_dict, res_df, return_polynomial_fits = True,
                save = False, save_path=''):
    
    [R1, H0, Rp, phi0, V0, A0] = [res_dict[k] for k in ['R1', 'H0', 'Rp', 'phi0', 'V0', 'A0']]
    # limit_reached = ('limit' in res_df['case'].values)
    
    #### fig01 - fit params + curvature
    fig01, axes01 = plt.subplots(2, 2, figsize = (12, 12))
    axes01_flat = axes01.flatten()
    suptitle = 'Fitted parameters as functions of $\\delta$' + '\n'
    suptitle += f'$R_1$ = {R1:.1f}, $H_0$ = {H0:.1f}, $R_p$ = {Rp:.1f} (µm)'
    fig01.suptitle(suptitle, size = 14)
    Xp = 'delta'
    Xa = '$\\delta$ (µm)'
    Yparams = ['r1', 'R0', 'phi', 'curvature']
    Yaxis = ['$r_1$ (µm)', '$R_0$ (µm)', '$\\phi$ (rad)', 'Mean curvature $(µm^{-1})$']
    
    N = res_df.shape[0]
    dict_c = {'1domain':'w', 'limit':'darkred', '2domains':'w'}
    error_array = (res_df['error'] > 0)
    color_array = np.array([str(dict_c[C]) for C in res_df['case'].values])
    color_array[error_array] = 'k'
    
    dict_ec = {'1domain':'deepskyblue', 'limit':'darkred', '2domains':'orange'}
    edgecolor_array = np.array([str(dict_ec[C]) for C in res_df['case'].values])
    
    size_array = np.ones(N) * 15
    
    for k in range(4):
        ax = axes01_flat[k]
        Yp = Yparams[k]
        Ya = Yaxis[k]
        
        ax.scatter(res_df[Xp], res_df[Yp], c=color_array, edgecolors=edgecolor_array, s=size_array, marker='o')#, edgecolors='k')
        ax.scatter([], [], c=dict_c['1domain'], edgecolors=dict_ec['1domain'], marker='o', label = '1 domain')
        ax.scatter([], [], c=dict_c['2domains'], edgecolors=dict_ec['2domains'], marker='o', label = '2 domains')
        ax.scatter([], [], c=dict_c['limit'], edgecolors=dict_ec['limit'], marker='o', label = 'Limit')
        ax.scatter([], [], c='k', edgecolors=None, marker='o', label = 'Error')
        ax.set_xlabel(Xa)
        ax.set_ylabel(Ya)
        ax.grid()
        if k == 0:
            ax.legend(loc='upper left', fontsize = 8)
        
    fig01.tight_layout()
    
    
    #### fig02 - Area & alpha
    def cubic_for_fit(x, K3, K2, K1):
        return(K3*x**3 + K2*x**2 + K1*x**1)
    
    f0 = (res_df['error']==0)
    Xraw, Yraw = res_df[f0]['delta'], res_df[f0]['alpha']
    
    K_alpha, covM = curve_fit(cubic_for_fit, Xraw, Yraw, p0=(1,1,1))
    P_alpha = np.poly1d([k for k in K_alpha] + [0])
    Yfit = np.polyval(P_alpha, res_df[f0]['delta'])
    
    residuals = Yraw - Yfit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Yraw-np.mean(Yraw))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    s = f'Fit y = {K_alpha[0]:.3e}$x^3$ + {K_alpha[1]:.3e}$x^2$ + {K_alpha[2]:.3e}$x$' + '\n'
    s += f'$R^2$ = {r_squared:.3f}'
    
    fig02, ax02 = plt.subplots(1, 1, figsize = (6, 6))
    suptitle = 'Area & $\\alpha$ as functions of $\\delta$' + '\n'
    suptitle += f'$R_1$ = {R1:.1f}, $H_0$ = {H0:.1f}, $R_p$ = {Rp:.1f} (µm)'
    fig02.suptitle(suptitle, size = 14)
    
    ax02_bis = ax02.twinx()
    
    ax = ax02
    ax.scatter(res_df['delta'], res_df['area'], c=color_array, edgecolors=edgecolor_array, s=size_array, marker='o')
    ax.set_xlabel('$\\delta$ (µm)')
    ax.set_ylabel('$Area$ (µm²)')
    
    ax = ax02_bis
    ax.grid()
    ax.scatter(res_df['delta'], res_df['alpha']*100, c=color_array, edgecolors=edgecolor_array, s=size_array, marker='o')
    ax.scatter([], [], c=dict_c['1domain'], edgecolors=dict_ec['1domain'], marker='o', label = '1 domain')
    ax.scatter([], [], c=dict_c['2domains'], edgecolors=dict_ec['2domains'], marker='o', label = '2 domains')
    ax.scatter([], [], c=dict_c['limit'], edgecolors=dict_ec['limit'], marker='o', label = 'Limit')
    ax.scatter([], [], c='k', edgecolors=None, marker='o', label = 'Error')
    ax.plot(res_df['delta'], np.polyval(P_alpha, res_df['delta'])*100, c='k', ls='--', lw=1.0, label=s)
    ax.set_ylabel('$\\alpha$ (%)')

    ax.legend(loc = 'upper left', fontsize = 8)
    
    fig02.tight_layout()
    
    
    #### fig03 - Lc
    Lc_raw = (R1**2 * res_df['curvature'].values - R1*np.sin(res_df['phi'].values))
    Lc_f = Lc_raw[f0]
    # res_df['Lc'] = Lc_raw
    
    def cubic_for_fit(x, K3, K2, K1):
        return(K3*x**3 + K2*x**2 + K1*x**1)
    
    f0 = (res_df['error']==0)
    Xraw, Yraw = res_df[f0]['delta'], Lc_f
    
    K_Lc, covM = curve_fit(cubic_for_fit, Xraw, Yraw, p0=(1,1,1))
    P_Lc = np.poly1d([k for k in K_Lc] + [0])
    Yfit = np.polyval(P_Lc, res_df[f0]['delta'])
    
    residuals = Yraw - Yfit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Yraw-np.mean(Yraw))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    s = f'Fit y = {K_Lc[0]:.3e}$x^3$ + {K_Lc[1]:.3e}$x^2$ + {K_Lc[2]:.3e}$x$' + '\n'
    s += f'$R^2$ = {r_squared:.3f}'
    
    fig03, ax03 = plt.subplots(1, 1, figsize = (6, 6))
    suptitle = '$L_c$ as a function of $\\delta$' + '\n'
    suptitle += f'$R_1$ = {R1:.1f}, $H_0$ = {H0:.1f}, $R_p$ = {Rp:.1f} (µm)'
    fig03.suptitle(suptitle, size = 14)
    
    ax = ax03
    ax.scatter(res_df['delta'], Lc_raw, c=color_array, edgecolors=edgecolor_array, s=size_array, marker='o')
    ax.scatter([], [], c=dict_c['1domain'], edgecolors=dict_ec['1domain'], marker='o', label = '1 domain')
    ax.scatter([], [], c=dict_c['2domains'], edgecolors=dict_ec['2domains'], marker='o', label = '2 domains')
    ax.scatter([], [], c=dict_c['limit'], edgecolors=dict_ec['limit'], marker='o', label = 'Limit')
    ax.scatter([], [], c='k', edgecolors=None, marker='o', label = 'Error')
    ax.plot(res_df['delta'], np.polyval(P_Lc, res_df['delta']), c='k', ls='--', lw=1.0, label=s)
    ax.grid()
    ax.set_xlabel('$\\delta$ (µm)')
    ax.set_ylabel('$L_c$ (µm)')

    ax.legend(loc = 'upper left', fontsize = 8)
    
    fig03.tight_layout()
    
    plt.show()
    
    if save:
        fig_names = ['delta-shape_fitting_parms', 'delta-alpha', 'delta-Lc']
        for fig, name in zip([fig01, fig02, fig03], fig_names):
            ufun.archiveFig(fig, name = name, figDir = save_path)
    
    poly_dict = {'P_alpha':P_alpha, 'P_Lc':P_Lc, }
    
    if return_polynomial_fits:
        return(poly_dict)
        
    


# %% Run the schmuck

# %%% Case 1

# R1 = 9
# H0 = 7.5
# Rp = 3
# N_points = 200
# delta_values = np.linspace(0, H0/2, N_points) + 0.01

# res_dict_01, res_df_01 = CSS_sphere_general(R1, H0, Rp, delta_values)
       
# plot_contours(res_dict_01, res_df_01)
# plot_curves(res_dict_01, res_df_01)

# %%% Case 2

# R1 = 9
# H0 = 7.5
# Rp = 25
# N_points = 200
# delta_values = np.linspace(0, H0/2, N_points) + 0.01

# res_dict_02, res_df_02 = CSS_sphere_general(R1, H0, Rp, delta_values)  
         
# plot_contours(res_dict_02, res_df_02)
# plot_curves(res_dict_02, res_df_02)

# %%% Case 3

# R1 = 9
# H0 = 12
# Rp = 3
# N_points = 200
# delta_values = np.linspace(0, H0/2, N_points) + 0.01

# res_dict_03, res_df_03 = CSS_sphere_general(R1, H0, Rp, delta_values)   
        
# plot_contours(res_dict_03, res_df_03)
# plot_curves(res_dict_03, res_df_03)

# %%% Case 4

# R1 = 9
# H0 = 12
# Rp = 25
# N_points = 200
# delta_values = np.linspace(0, H0/2, N_points) + 0.01

# res_dict_04, res_df_04 = CSS_sphere_general(R1, H0, Rp, delta_values)     
      
# plot_contours(res_dict_04, res_df_04)
# plot_curves(res_dict_04, res_df_04)

# %%% Case 5

# R1 = 9
# H0 = 6
# Rp = 3
# N_points = 200
# delta_values = np.linspace(0, H0/2, N_points) + 0.01

# res_dict_05, res_df_05 = CSS_sphere_general(R1, H0, Rp, delta_values)     
      
# plot_contours(res_dict_05, res_df_05)
# plot_curves(res_dict_05, res_df_05)

# %%% Case 6

# R1 = 9
# H0 = 6
# Rp = 25
# N_points = 200
# delta_values = np.linspace(0, H0/2, N_points) + 0.01

# res_dict_06, res_df_06 = CSS_sphere_general(R1, H0, Rp, delta_values)  
         
# plot_contours(res_dict_06, res_df_06)
# plot_curves(res_dict_06, res_df_06)
