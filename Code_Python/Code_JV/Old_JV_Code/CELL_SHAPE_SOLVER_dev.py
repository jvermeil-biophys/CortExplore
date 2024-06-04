# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 14:04:34 2024

@author: JosephVermeil
"""

# %% Imports

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

import scipy.integrate as integrate
from scipy.optimize import fsolve
from scipy.optimize import curve_fit

# %% Parallel plates

# %%% Constants

Rc = 1
V0 = (4/3) * np.pi * Rc**3
A0 = 4 * np.pi * Rc**2

# IMPORTANT NOTATION
# RR = [Ri, R0]
# Ri = RR[0]
# R0 = RR[1]

# %%% Main functions

def A_flat(RR):
    return(RR[1]/(RR[1]**2 - RR[0]**2))
    
def B_flat(RR):
    return(-RR[0]**2 * RR[1]/(RR[1]**2 - RR[0]**2))

def u_flat(r, RR):
    return((A_flat(RR)*r) + (B_flat(RR)/r))

def integrand_Xi(r, RR):
    return(u_flat(r, RR) / ((1-u_flat(r, RR)**2)**0.5))

def integrand_V(r, RR):
    return(r**2 * (u_flat(r, RR) / ((1-u_flat(r, RR)**2)**0.5)))

def Xi_fun(RR):
    return((2/Rc) * integrate.quad(integrand_Xi, RR[0], RR[1], args=RR)[0])

def V_fun(RR):
    return(2*np.pi * integrate.quad(integrand_V, RR[0], RR[1], args=RR)[0])


# %%% Solving shit

Xi_values = np.arange(2.0, 1.0, -0.01) - 0.0001
N = len(Xi_values)

RR_sol = np.zeros((N, 2))

RR_guess = [0.1, 1.1]

for i in range(N):
    Xi = Xi_values[i]
    
    def Xi_fun_0(RR):
        return(Xi - Xi_fun(RR))
    
    def V_0(RR):
        return(V0 - V_fun(RR))
    
    def fun_solve(RR):
        return(Xi_fun_0(RR), V_0(RR))
    
    RR_sol[i] = fsolve(fun_solve, RR_guess)
    RR_guess = RR_sol[i]
    
    
# %%% Other functions

def g_fun(RR):
    return(2*np.pi * RR[1]*RR[0]**2 / (Rc*(RR[1]**2 - RR[0]**2)))

def integrand_A(r, RR):
    return(r / ((1-u_flat(r, RR)**2)**0.5))

def A_fun(RR):
    return(2*np.pi*(RR[0]**2) + (4*np.pi) * integrate.quad(integrand_A, RR[0], RR[1], args=RR)[0])

def alpha_fun(RR):
    return((A_fun(RR) - A0) / A0)

def z_fun(r, RR):
    return((1/Rc) * integrate.quad(integrand_Xi, RR[0], r, args=RR)[0])
    

# %%% Tests Simples

RR_test = [0, 1]

Xi_fun_test = Xi_fun(RR_test)
V_test = V_fun(RR_test)
A_test = A_fun(RR_test)

# RRi = RR_sol[-1]
# r_test = np.mean(RRi[1]-0.1)
# z = z_fun(r_test, RRi)

# %%% Plots

def verification_plot(Xi_values, RR_sol):
    try:
        Xi_values = Xi_values[:50]
    except:
        pass
    N = len(Xi_values)
    fig, axes = plt.subplots(1,3, figsize = (18, 6))
    
    #### Data
    g_sol = np.zeros_like(Xi_values)
    A_sol = np.zeros_like(Xi_values)
    alpha_sol = np.zeros_like(Xi_values)
    
    for i in range(N):
        g_sol[i] = g_fun(RR_sol[i])
        A_sol[i] = A_fun(RR_sol[i])
        alpha_sol[i] = alpha_fun(RR_sol[i])
        
    Xi_plot = 2 - Xi_values
    
    #### Fits
    def cubic_for_fit(x, K3, K2, K1):
        return(K3*x**3 + K2*x**2 + K1*x**1)
        
    K_g_paper = [0.9144, 2.3115, 0.9775, 0]
    P_g_paper = np.poly1d(K_g_paper)
    K_g, covM = curve_fit(cubic_for_fit, Xi_plot, g_sol, p0=(1,1,1))
    P_g = np.poly1d([k for k in K_g] + [0])
    
    K_alpha, covM = curve_fit(cubic_for_fit, Xi_plot, alpha_sol, p0=(1,1,1))
    P_alpha = np.poly1d([k for k in K_alpha] + [0])
    
    K_A_paper = [0.9638, 0.4443, 0.0014, 0]
    P_A_paper = np.poly1d(K_A_paper)
    K_A, covM = curve_fit(cubic_for_fit, Xi_plot, A_sol-4*np.pi, p0=(1,1,1))
    P_A = np.poly1d([k for k in K_A] + [0])

    #### Plots
    ax = axes[0]
    ax.plot(Xi_plot, g_sol, c='w', marker='o', mec='k', ms=8)
    
    s = f'y = {K_g_paper[0]:.3f}$x^3$ + {K_g_paper[1]:.3f}$x^2$ + {K_g_paper[2]:.3f}$x$ - From A.J. et al.'
    ax.plot(Xi_plot, np.polyval(P_g_paper, Xi_plot), c='c', ls='-', lw=3.0, label=s)
    
    s = f'y = {K_g[0]:.3f}$x^3$ + {K_g[1]:.3f}$x^2$ + {K_g[2]:.3f}$x$ - My fit'
    ax.plot(Xi_plot, np.polyval(P_g, Xi_plot), c='r', ls='-', lw=1.2, label=s)
    
    ax.legend(loc='upper left', fontsize = 8)
    ax.set_title('Shape function $g(\\xi)$')
    ax.set_xlabel('$\\xi = z_p / R_c$')
    ax.set_ylabel('$g(\\xi)$')
    ax.grid()
    
    
    ax = axes[1]
    ax.plot(Xi_plot, alpha_sol * 100, c='w', marker='o', mec='k', ms=8)

    s = f'y = {K_alpha[0]:.3f}$x^3$ + {K_alpha[1]:.3f}$x^2$ + {K_alpha[2]:.3f}$x$ - My fit'
    ax.plot(Xi_plot, np.polyval(P_alpha, Xi_plot)*100, c='r', ls='-', lw=1.2, label=s)
    
    ax.legend(loc='upper left', fontsize = 8)
    ax.set_title('Areal strain function $\\alpha(\\xi)=\\Delta$A/$A_0$')
    ax.set_xlabel('$\\xi = z_p / R_c$')
    ax.set_ylabel('$\\alpha(\\xi)$ (%)')
    ax.grid()
    
    
    ax = axes[2]
    ax.plot(Xi_plot, A_sol, c='w', marker='o', mec='k', ms=8)
    
    s = f'y = {K_A_paper[0]:.3f}$x^3$ + {K_A_paper[1]:.3f}$x^2$ + {K_A_paper[2]:.3f}$x$ - From A.J. et al.'
    ax.plot(Xi_plot, 4*np.pi + np.polyval(P_A_paper, Xi_plot), c='c', ls='-', lw=3.0, label=s)
    
    s = f'y = {K_A[0]:.3f}$x^3$ + {K_A[1]:.3f}$x^2$ + {K_A[2]:.3f}$x$ - My fit'
    ax.plot(Xi_plot, 4*np.pi + np.polyval(P_A, Xi_plot), c='r', ls='-', lw=1.2, label=s)
    
    ax.legend(loc='upper left', fontsize = 8)
    ax.set_title('Total surface area $A(\\xi)$ - given $R_c$ = 1µm')
    ax.set_xlabel('$\\xi = z_p / R_c$')
    ax.set_ylabel('Area ($µm^2$)')
    ax.grid()
    
    fig.suptitle("Reproduction of the method by A. Janshoff et al. (PRL 2020)", size = 16)
    
    plt.show()
    
    return(K_g, K_A, K_alpha)


def cell_contour_plot(Xi_values, RR_sol):
    N = len(Xi_values)
    fig, ax = plt.subplots(1,1, figsize = (8, 8))
    ax.axhline(0, c='k', lw=5)
    ax.grid()
    ax.set_title('Cell contours for various $\\xi = z_p / R_c$')
    fig.suptitle("Reproduction of the method by A. Janshoff et al. (PRL 2020)", size = 12)
    ax.set_xlabel('$r$ / $R_c$')
    ax.set_ylabel('$z$ / $R_c$')
    
    #### Data
    # g_sol = np.zeros_like(Xi_values)
    # A_sol = np.zeros_like(Xi_values)
    # alpha_sol = np.zeros_like(Xi_values)
    
    # for i in range(N):
    #     g_sol[i] = g_fun(RR_sol[i])
    #     A_sol[i] = A_fun(RR_sol[i])
    #     alpha_sol[i] = alpha_fun(RR_sol[i])
        
    Nc = N//10
    cList = plt.cm.viridis(np.linspace(0, 1, Nc))
    for i in range(0, N, 10):
        c = cList[i//10]
        [Ri, R0] = RR_sol[i]
        Xi = Xi_values[i]
        
        rr = np.concatenate([np.linspace(Ri, 0.9*R0, num=100), np.linspace(0.9*R0, R0, num=100)])
        zz = np.array([z_fun(r, [Ri, R0]) for r in rr])
        
        plt.plot(rr, zz, c = c, label=f'$\\xi$ = {Xi:.2f}')
        plt.plot(-rr, zz, c = c)
        plt.plot(rr, Xi-zz, c = c)
        plt.plot(-rr, Xi-zz, c = c)
        plt.hlines(Xi, -Ri, Ri, colors = c)
        plt.hlines(Xi, -Ri, Ri, colors = c)
        plt.hlines(0, -Ri, Ri, colors = c)
        
    ax.axis('equal')
    plt.legend(loc='upper right', fontsize =6)
    plt.show()

K_g, K_A, K_alpha = verification_plot(Xi_values, RR_sol)

cell_contour_plot(Xi_values, RR_sol)





# %% Spherical indenter, phi0 < pi/2

# %%% Constants

# Measured parameters
R1 = 9
h0 = 5.5
Rp = 3

# Computed parameters
signe_h = (h0-R1)/np.abs(h0-R1)
phi0 = np.arcsin(2*h0*R1 / (h0**2 + R1**2)) * (-signe_h) + np.pi * (h0>R1)
R00 = (h0**2 + R1**2) / (2*h0)
phi0_deg = phi0 * 180/np.pi
A0 = 2*np.pi*R00*h0
V0 = (np.pi/3)*h0**2*(3*R00 - h0)

# phi0 = np.pi/2 + np.pi/6
# signe_phi0 = int((phi0 - np.pi/2) / np.abs(phi0 - np.pi/2))
# R0 = R1/np.sin(phi0)
# h0 = R0 + signe_phi0 * (R0**2 - R1**2)**0.5

# IMPORTANT NOTATION
# r1_phi = [r1, phi]
# r1 = r1_phi[0]
# phi = r1_phi[1]


# %%% Plotting the initial geometry
def initial_cell_contour_plot(R1=8, h0=6, Rp=2, 
                              fig = None, ax = None, 
                              annotations = True, plot_circle = True):
    signe_h = (h0-R1)/np.abs(h0-R1)
    phi0 = np.arcsin(2*h0*R1 / (h0**2 + R1**2)) * (-signe_h) + np.pi * (h0>R1)
    R00 = (h0**2 + R1**2) / (2*h0)
    phi0_deg = phi0 * 180/np.pi

    CenterCell = h0 - R00
    AnglesCell = np.linspace(np.pi/2 - phi0, np.pi/2 + phi0, 200)
    RRCell = R00 * np.cos(AnglesCell)
    ZZCell = R00 * np.sin(AnglesCell) + CenterCell
    
    AnglesNotCell = np.linspace(-(3/2)*np.pi + phi0, np.pi/2 - phi0, 200)
    RRNotCell = R00 * np.cos(AnglesNotCell)
    ZZNotCell = R00 * np.sin(AnglesNotCell) + CenterCell

    CenterIndenter = h0 + Rp
    AnglesIndenter = np.linspace(0, 2*np.pi, 200)
    RRIndenter = Rp * np.cos(AnglesIndenter)
    ZZIndenter = Rp * np.sin(AnglesIndenter) + CenterIndenter
    
    if ax == None:
        fig, ax = plt.subplots(1,1, figsize = (8, 8))
        
    ax.axhline(0, c='k', lw=5)
    ax.plot(RRCell, ZZCell, 'b-')
    ax.plot(RRIndenter, ZZIndenter, 'r-')
    if plot_circle:
        ax.plot(RRNotCell, ZZNotCell, 'k--')
        ax.plot(0, CenterCell, 'ko')
    if annotations:
        # Centers
        ax.plot(0, CenterCell, 'ko')
        ax.plot(0, CenterIndenter, 'ro')
        
        # R0
        ax.text((2**0.5)*R00/2 + 0.5, CenterCell + (2**0.5)*R00/2 + 0.2, f'R0 = {R00:.2f}', color='c')
        ax.annotate("", size = 10,
            xy=(0, CenterCell), #xycoords='data',
            xytext=((2**0.5)*R00/2, CenterCell + (2**0.5)*R00/2), #textcoords='data',
            arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0.0", color='c'))
        # R1
        ax.text(- R1 , 0 - 0.5, f'R1 = {R1:.1f}', color='c', ha='left', va='top')
        ax.annotate("", size = 10,
            xy=(0, 0 - 0.25), #xycoords='data',
            xytext=(-R1, 0 - 0.25), #textcoords='data',
            arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0.0", color='c'))
        # Rp
        ax.text(RRIndenter[155] + 1.5, ZZIndenter[155] + 0.2, f'Rp = {Rp:.1f}', ha='left', va='top', color='orange')
        ax.annotate("", size = 10,
            xy=(0, CenterIndenter), #xycoords='data',
            xytext=(RRIndenter[155], ZZIndenter[155]), #textcoords='data',
            arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0.0", color='orange'))
        # Phi
        d = 2
        x1, y1 = R1, 0
        x2, y2 = x1 - d*np.cos(phi0), y1 + d*np.sin(phi0)
        d2 = 1.5
        x10, y10 = R1-1, 0
        x20, y20 = x1 - d2*np.cos(phi0), y1 + d2*np.sin(phi0)
        ax.plot([x1, x2], [y1, y2], 'c-')
        ax.plot([R1-d, R1], [0, 0], 'c-')
        # ax.plot([x10, x20], [y10, y20], 'co', ms=5)
        ax.text((x10+x20)*0.5 - 0.5, (y10+y20)*0.5 + 0.5, f'$\\Phi$ = {phi0_deg:.2f}°', color='c', ha='right')
        ax.annotate("", size = 10,
            xy=(x10, y10), #xycoords='data',
            xytext=(x20, y20), #textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle=f"arc3,rad={phi0/2}", color='c'))
        # connectionstyle=f"arc3,angleA={90-phi0_deg},angleB={360-phi0_deg},rad={phi0_deg/3}"
        
    ax.grid()
    fig.suptitle("Cell contours in the initial geometry", size = 16)
    ax.set_xlabel('r (µm)')
    ax.set_ylabel('z (µm)')
        
    ax.axis('equal')
    ax.axis('equal')
    ax.set_xlim([-R1-0.5, +R1+0.5])
    ax.set_ylim([h0-R1-0.5, h0+R1+0.5])
    
    plt.tight_layout()
    plt.show()


# fig, axes = plt.subplots(1, 2, figsize = (16, 8), sharey=True)

# initial_cell_contour_plot(R1 = 8.5, h0 = 7, Rp = 3, fig=fig, ax=axes[0], 
#                           annotations = True, plot_circle = False)

# initial_cell_contour_plot(R1 = 8.5, h0 = 7, Rp = 25, fig=fig, ax=axes[1],
#                           annotations = True, plot_circle = False)

# %%% Main functions

def my_sqrt(x):
    return(max(1e-10, x)**0.5)

def A_sphere(r1_phi):
    return((R1*np.sin(r1_phi[1]) + ((r1_phi[0]**2)/Rp))/(R1**2 - r1_phi[0]**2))
    
# def B1_sphere(r1_phi):
#     return(- r1_phi[0]**2 * ((1/Rp) + ((R1*np.sin(r1_phi[1]) + (r1_phi[0]**2/Rp))/(R1**2 - r1_phi[0]**2))))
def B_sphere(r1_phi):
    return(-(r1_phi[0]**2)*(R1*np.sin(r1_phi[1]) + (r1_phi[0]**2/Rp))/(R1**2 - r1_phi[0]**2) - (r1_phi[0]**2)/Rp)

def u_sphere(r, r1_phi):
    return((A_sphere(r1_phi)*r) + (B_sphere(r1_phi)/r))

# def h_cap(r1_phi):
#     return(Rp - (Rp**2 - r1_phi[0]**2)**0.5)

# def A_cap(r1_phi):
#     return(2*np.pi*Rp*(Rp - (Rp**2 - r1_phi[0]**2)**0.5))

# def V_cap(r1_phi):
#     return((np.pi/3) * (Rp - (Rp**2 - r1_phi[0]**2)**0.5)**2 * (3*Rp - (Rp - (Rp**2 - r1_phi[0]**2)**0.5)))

def A_cap(r1_phi):
    return(2*np.pi*Rp*(Rp - my_sqrt(Rp**2 - r1_phi[0]**2)))

def V_cap(r1_phi):
    return((np.pi/3) * (Rp - my_sqrt(Rp**2 - r1_phi[0]**2))**2 * (3*Rp - (Rp - my_sqrt(Rp**2 - r1_phi[0]**2))))

# def integrand_Delta(r, r1_phi):
#     return(u_sphere(r, r1_phi) / ((1 - u_sphere(r, r1_phi)**2)**0.5))

# def integrand_Vsphere(r, r1_phi):
#     return(r**2 * (u_sphere(r, r1_phi) / ((1 - u_sphere(r, r1_phi)**2)**0.5)))

def integrand_Delta(r, r1_phi):
    return(u_sphere(r, r1_phi) / (my_sqrt(1 - u_sphere(r, r1_phi)**2)))

def integrand_Vsphere(r, r1_phi):
    return(r**2 * (u_sphere(r, r1_phi) / (my_sqrt(1 - u_sphere(r, r1_phi)**2))))

def z_sphere_fun(r1_phi):
    return(integrate.quad(integrand_Delta, r1_phi[0], R1, args=r1_phi)[0])

# def Delta_fun(r1_phi):
#     return(h0 - integrate.quad(integrand_Delta, r1_phi[0], R1, args=r1_phi)[0] + Rp - (Rp**2 - r1_phi[0]**2)**0.5)

def Delta_fun(r1_phi):
    return(h0 - integrate.quad(integrand_Delta, r1_phi[0], R1, args=r1_phi)[0] + Rp - my_sqrt(Rp**2 - r1_phi[0]**2))

def Vsphere_fun(r1_phi):
    return(np.pi * integrate.quad(integrand_Vsphere, r1_phi[0], R1, args=r1_phi)[0] - V_cap(r1_phi))


# %%% Tests Simples

r1_phi_test = [Rp*0.3, phi0+0.1]

print("V_cap", V_cap(r1_phi_test))
print("Vsphere_fun", Vsphere_fun(r1_phi_test))
print("")
print("Delta_fun", Delta_fun(r1_phi_test))
print("z_sphere_fun", z_sphere_fun(r1_phi_test))


# def V_cap_test(R0, R1):
#     return((np.pi/3) * (R0 - (R0**2 - R1**2)**0.5)**2 * (3*R0 - (R0 - (R0**2 - R1**2)**0.5)))

# print("V_cap_test", V_cap_test(R00, R1))


# %%% Solving shit

### In Delta

Delta_values = np.linspace(0, h0/3, 200) + 0.01
# Delta_values = np.array([0.2*Rp])
N = len(Delta_values)

r1_phi_sol = np.zeros((N, 2))

r1_phi_guess = [0.1, phi0]

for i in range(N):
    Delta = Delta_values[i]
    
    def Delta_fun_0(r1_phi):
        return(Delta - Delta_fun(r1_phi))
    
    def Vsphere_0(r1_phi):
        return(V0 - Vsphere_fun(r1_phi))
    
    def fun_solve(r1_phi):
        return(Delta_fun_0(r1_phi), Vsphere_0(r1_phi))
    
    r1_phi_sol[i] = fsolve(fun_solve, r1_phi_guess)
    r1_phi_guess = r1_phi_sol[i]
    
### In Z

# Z_values = np.linspace(h0, h0 - 0.2*Rp, 10) - 0.01
# # Z_values = np.array([h0 - 0.1])
# N = len(Z_values)

# r1_phi_sol = np.zeros((N, 2))

# r1_phi_guess = [0.1, phi0]

# for i in range(N):
#     Z = Z_values[i]
    
#     def Z_fun_0(r1_phi):
#         return(Z - z_sphere_fun(r1_phi))
    
#     def Vsphere_0(r1_phi):
#         return(V0 - Vsphere_fun(r1_phi))
    
#     def fun_solve(r1_phi):
#         return(Z_fun_0(r1_phi), Vsphere_0(r1_phi))
    
#     r1_phi_sol[i] = fsolve(fun_solve, r1_phi_guess)
#     r1_phi_guess = r1_phi_sol[i]
    
# %%% Other functions

def integrand_Asphere(r, r1_phi):
    return(r / ((1-u_sphere(r, r1_phi)**2)**0.5))

def Area_sphere_fun(r1_phi):
    return(2*np.pi*integrate.quad(integrand_Asphere, r1_phi[0], R1, args=r1_phi)[0] + A_cap(r1_phi))

def z_sphere_fun2(r, r1_phi):
    return(integrate.quad(integrand_Delta, r, R1, args=r1_phi)[0])
    
def alpha_sphere_fun(r1_phi):
    return((Area_sphere_fun(r1_phi) - A0) / A0)

# %%% contour plots

def cell_contour_plot2(Delta_values, r1_phi_sol):
    N = len(Delta_values)
    
    # ax.set_title('Cell contours for various $\\xi = z_p / R_c$')
    # fig.suptitle("Reproduction of the method by A. Janshoff et al. (PRL 2020)", size = 12)
    # ax.set_xlabel('$r$ / $R_c$')
    # ax.set_ylabel('$z$ / $R_c$')
        
    Nc = N//10
    fig, m_axes = plt.subplots(2,Nc//2, figsize = (3*Nc/2, 6))
    axes = m_axes.flatten()
    cList = plt.cm.viridis(np.linspace(0, 0.9, Nc))
    for i in range(0, N, 10):
        ax = axes[i//10]
        
        
        ax.axhline(0, c='k', lw=5)
        ax.grid()
        c = cList[i//10]
        [r1, phi] = r1_phi_sol[i]
        Delta = Delta_values[i]
        
        rr = np.linspace(r1, R1, num=100)
        zz = np.array([z_sphere_fun2(r, [r1, phi]) for r in rr])
        
        ax.plot(rr, zz, c = c)
        ax.plot(-rr, zz, c = c)
        
        CenterIndenter = h0 + Rp - Delta
        AnglesIndenter = np.linspace(0, 2*np.pi, 100)
        RRIndenter = Rp * np.cos(AnglesIndenter)
        ZZIndenter = Rp * np.sin(AnglesIndenter) + CenterIndenter
        ax.plot(RRIndenter, ZZIndenter, 'r-')
        # plt.plot(rr, Xi-zz, c = c)
        # plt.plot(-rr, Xi-zz, c = c)
        # plt.hlines(Xi, -Ri, Ri, colors = c)
        # plt.hlines(Xi, -Ri, Ri, colors = c)
        # plt.hlines(0, -Ri, Ri, colors = c)
        
    
    for ax in axes:
        ax.axis('equal')
        ax.set_xlim([-R1-0.5, +R1+0.5])
        ax.set_ylim([h0-R1-0.5, h0+R1+0.5])
        
    plt.show()

cell_contour_plot2(Delta_values, r1_phi_sol)


# %%% numeric plots

#### TBD



# %% Spherical indenter, phi0 > pi/2

# %%% Constants

# Measured parameters

# Funny test
# R1 = 2.5
# h0 = 17
# Rp = 500

# 24-02-26_M2_C4_I3
R1 = 9.8
h0 = 13
Rp = 3

# 24-02-26_M2_C1_I3
# R1 = 9.59
# h0 = 11.24
# Rp = 26.5

# Computed parameters
signe_h = (h0-R1)/np.abs(h0-R1)
phi0 = np.arcsin(2*h0*R1 / (h0**2 + R1**2)) * (-signe_h) + np.pi * (h0>R1)
R00 = (h0**2 + R1**2) / (2*h0)
phi0_deg = phi0 * 180/np.pi
A0 = 2*np.pi*R00*h0
V0 = (np.pi/3)*h0**2*(3*R00 - h0)
Rc_eq = (V0*(3/4)*(1/np.pi))**(1/3) 

# phi0 = np.pi/2 + np.pi/6
# signe_phi0 = int((phi0 - np.pi/2) / np.abs(phi0 - np.pi/2))
# R0 = R1/np.sin(phi0)
# h0 = R0 + signe_phi0 * (R0**2 - R1**2)**0.5

# IMPORTANT NOTATION
# r1_phi = [r1, phi]
# r1 = r1_phi[0]
# phi = r1_phi[1]


# %%% Plotting the initial geometry
def initial_cell_contour_plot(R1=8, h0=6, Rp=2, 
                              fig = None, ax = None, 
                              annotations = True, plot_circle = True):
    signe_h = (h0-R1)/np.abs(h0-R1)
    phi0 = np.arcsin(2*h0*R1 / (h0**2 + R1**2)) * (-signe_h) + np.pi * (h0>R1)
    R00 = (h0**2 + R1**2) / (2*h0)
    phi0_deg = phi0 * 180/np.pi

    CenterCell = h0 - R00
    AnglesCell = np.linspace(np.pi/2 - phi0, np.pi/2 + phi0, 200)
    RRCell = R00 * np.cos(AnglesCell)
    ZZCell = R00 * np.sin(AnglesCell) + CenterCell
    
    AnglesNotCell = np.linspace(-(3/2)*np.pi + phi0, np.pi/2 - phi0, 200)
    RRNotCell = R00 * np.cos(AnglesNotCell)
    ZZNotCell = R00 * np.sin(AnglesNotCell) + CenterCell

    CenterIndenter = h0 + Rp
    AnglesIndenter = np.linspace(0, 2*np.pi, 200)
    RRIndenter = Rp * np.cos(AnglesIndenter)
    ZZIndenter = Rp * np.sin(AnglesIndenter) + CenterIndenter
    
    if ax == None:
        fig, ax = plt.subplots(1,1, figsize = (8, 8))
        
    ax.axhline(0, c='k', lw=5)
    ax.plot(RRCell, ZZCell, 'b-')
    ax.plot(RRIndenter, ZZIndenter, 'r-')
    if plot_circle:
        ax.plot(RRNotCell, ZZNotCell, 'k--')
        ax.plot(0, CenterCell, 'ko')
    if annotations:
        # Centers
        ax.plot(0, CenterCell, 'ko')
        ax.plot(0, CenterIndenter, 'ro')
        
        # R0
        ax.text((2**0.5)*R00/2 + 0.5, CenterCell + (2**0.5)*R00/2 + 0.2, f'$R_0$ = {R00:.2f}', color='c')
        ax.annotate("", size = 10,
            xy=(0, CenterCell), #xycoords='data',
            xytext=((2**0.5)*R00/2, CenterCell + (2**0.5)*R00/2), #textcoords='data',
            arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0.0", color='c'))
        # R1
        ax.text(- R1 , 0 - 0.8, f'$R_1$ = {R1:.1f}', color='c', ha='left', va='top')
        ax.annotate("", size = 10,
            xy=(0, 0 - 0.5), #xycoords='data',
            xytext=(-R1, 0 - 0.5), #textcoords='data',
            arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0.0", color='c'))
        # Rp
        iRp = 175
        ax.text(RRIndenter[iRp] + 0.5, ZZIndenter[iRp] + 0.2, f'$R_p$ = {Rp:.1f}', ha='left', va='top', color='orange')
        ax.annotate("", size = 10,
            xy=(0, CenterIndenter), #xycoords='data',
            xytext=(RRIndenter[iRp], ZZIndenter[iRp]), #textcoords='data',
            arrowprops=dict(arrowstyle="<-", connectionstyle="arc3,rad=0.0", color='orange'))
        # Phi
        d = 2
        x1, y1 = R1, 0
        x2, y2 = x1 - d*np.cos(phi0), y1 + d*np.sin(phi0)
        d2 = 1.5
        x10, y10 = R1-1, 0
        x20, y20 = x1 - d2*np.cos(phi0), y1 + d2*np.sin(phi0)
        ax.plot([x1, x2], [y1, y2], 'c-')
        ax.plot([R1-d, R1], [0, 0], 'c-')
        # ax.plot([x10, x20], [y10, y20], 'co', ms=5)
        ax.text((x10+x20)*0.5 - 1, (y10+y20)*0.5 + 1, f'$\\Phi$ = {phi0_deg:.2f}°', color='c', ha='right')
        ax.annotate("", size = 10,
            xy=(x10, y10), #xycoords='data',
            xytext=(x20, y20), #textcoords='data',
            arrowprops=dict(arrowstyle="->", connectionstyle=f"arc3,rad={phi0/2}", color='c'))
        # connectionstyle=f"arc3,angleA={90-phi0_deg},angleB={360-phi0_deg},rad={phi0_deg/3}"
        
    ax.grid()
    fig.suptitle("Cell contour in the initial geometry", size = 16)
    ax.set_xlabel('r (µm)')
    ax.set_ylabel('z (µm)')
        
    
    w = max(R1, h0)
    ax.set_xlim([-w-0.5, +w+0.5])
    ax.set_ylim([h0-w-0.5, h0+w+0.5])
    ax.axis('equal')
    
    plt.tight_layout()
    plt.show()


initial_cell_contour_plot(R1 = R1, h0 = h0, Rp = Rp, 
                          annotations = True, plot_circle = False)


# %%% Main functions

# rRp = [r1, R0, phi]


#### Auxiliary
def my_sqrt(x):
    return(max(1e-10, x)**0.5)

def h_cap(rRp):
    return(Rp - my_sqrt(Rp**2 - rRp[0]**2))

def A_cap(rRp):
    return(2*np.pi*Rp*(Rp - my_sqrt(Rp**2 - rRp[0]**2)))

def V_cap(rRp):
    return((np.pi/3) * (Rp - my_sqrt(Rp**2 - rRp[0]**2))**2 * (3*Rp - (Rp - my_sqrt(Rp**2 - rRp[0]**2))))


#### Domaine 1 (above equator)
def A1_sphere(rRp):
    return((rRp[1] + ((rRp[0]**2)/Rp))/(rRp[1]**2 - rRp[0]**2))
    
def B1_sphere(rRp):
    return(-(rRp[0]**2) * (rRp[1] + ((rRp[0]**2)/Rp))/(rRp[1]**2 - rRp[0]**2) - (rRp[0]**2)/Rp)

def u1_sphere(r, rRp):
    return((A1_sphere(rRp)*r) + (B1_sphere(rRp)/r))

def integrand_z1(r, rRp):
    return(u1_sphere(r, rRp) / (my_sqrt(1 - u1_sphere(r, rRp)**2)))

def integrand_V1(r, rRp):
    return(r**2 * (u1_sphere(r, rRp) / (my_sqrt(1 - u1_sphere(r, rRp)**2))))

def z1_fun2(r, rRp):
    return(integrate.quad(integrand_z1, r, rRp[1], args=rRp, limit = 300, maxp1 = 300)[0])

def V1_fun(rRp):
    return(np.pi * integrate.quad(integrand_V1, rRp[0], rRp[1], args=rRp, limit = 300, maxp1 = 300)[0])


#### Domaine 2 (below equator)
def A2_sphere(rRp):
    return((rRp[1] - (R1*np.sin(rRp[2])))/(rRp[1]**2 - R1**2))
    
def B2_sphere(rRp):
    return(-(rRp[1]**2) * ((rRp[1] - (R1*np.sin(rRp[2])))/(rRp[1]**2 - R1**2)) + rRp[1])

def u2_sphere(r, rRp):
    return((A2_sphere(rRp)*r) + (B2_sphere(rRp)/r))

def integrand_z2(r, rRp):
    return(u2_sphere(r, rRp) / (my_sqrt(1 - u2_sphere(r, rRp)**2)))

def integrand_V2(r, rRp):
    return(r**2 * (u2_sphere(r, rRp) / (my_sqrt(1 - u2_sphere(r, rRp)**2))))

def z2_fun2(r, rRp):
    return(integrate.quad(integrand_z2, R1, r, args=rRp, limit = 100, maxp1 = 100)[0])

def V2_fun(rRp):
    return(np.pi * integrate.quad(integrand_V2, R1, rRp[1], args=rRp, limit = 100, maxp1 = 100)[0])


#### Total
def z_fun(rRp):
    return(z1_fun2(rRp[0], rRp) + z2_fun2(rRp[1], rRp))

def Delta_fun(rRp):
    return(h0 - z_fun(rRp) + h_cap(rRp))

def V_fun(rRp):
    return(V1_fun(rRp) + V2_fun(rRp) - V_cap(rRp))

def junction_equation(rRp):
    return(((rRp[1] - R1*np.sin(rRp[2]))/(rRp[1]**2 - R1**2)) - ((rRp[1] + (rRp[0]**2)/Rp)/(rRp[1]**2 - rRp[0]**2)))


# %%% Tests Simples

rRp_test = [1, R00 + 0.1, phi0+1]

print("V_cap", V_cap(rRp_test))
print("")
print("z1_fun2", z1_fun2(rRp_test[0], rRp_test))
print("V1_fun", V1_fun(rRp_test))
print("")
print("z2_fun2", z2_fun2(rRp_test[1], rRp_test))
print("V2_fun", V2_fun(rRp_test))
print("")
print("Delta_fun", Delta_fun(rRp_test))
print("V_fun", V_fun(rRp_test))


# def V_cap_test(R0, R1):
#     return((np.pi/3) * (R0 - (R0**2 - R1**2)**0.5)**2 * (3*R0 - (R0 - (R0**2 - R1**2)**0.5)))

# print("V_cap_test", V_cap_test(R00, R1))


# %%% Solving shit

### In Delta

Delta_values = np.linspace(0, h0/3, 200) + 0.01
# Delta_values = np.array([h0/4])
N = len(Delta_values)

rRp_sol = np.zeros((N, 3))

rRp_guess = [0.1, R00+0.1, phi0]

for i in range(N):
    Delta = Delta_values[i]
    
    def Delta_fun_0(rRp):
        return(Delta - Delta_fun(rRp))
    
    def V_fun_0(rRp):
        return(V0 - V_fun(rRp))
    
    def fun_solve(rRp):
        return(Delta_fun_0(rRp), V_fun_0(rRp), junction_equation(rRp))
    
    rRp_sol[i] = fsolve(fun_solve, rRp_guess)
    rRp_guess = rRp_sol[i]
    
# %%% Other functions

# def integrand_Asphere(r, rRp):
#     return(r / ((1-u_sphere(r, rRp)**2)**0.5))

# def Asphere_fun(rRp):
#     return(2*np.pi*integrate.quad(integrand_Asphere, rRp[0], R1, args=rRp)[0] + A_cap(rRp))

# def z_sphere_fun2(r, rRp):
#     return(integrate.quad(integrand_Delta, r, R1, args=rRp)[0])
    
# def alpha_sphere_fun(rRp):
#     return((Asphere_fun(rRp) - A0) / A0)

# %%% contour plots


def cell_contour_plot2(Delta_values, rRp_sol):
    N = len(Delta_values)
    
    # ax.set_title('Cell contours for various $\\xi = z_p / R_c$')
    # fig.suptitle("Reproduction of the method by A. Janshoff et al. (PRL 2020)", size = 12)
    # ax.set_xlabel('$r$ / $R_c$')
    # ax.set_ylabel('$z$ / $R_c$')
    f = 10
    Nc = N//f
    fig, m_axes = plt.subplots(2,Nc//2, figsize = (3.5*Nc/2, 7))
    title = 'Cell Axial Indentation'
    title += '\n'
    title += f'$R_1$ = {R1:.1f} µm, $h_0$ = {h0:.1f} µm, $R_p$ = {Rp:.1f} µm'
    fig.suptitle(title)
    axes = m_axes.flatten()
    cList = plt.cm.viridis(np.linspace(0, 0.9, Nc))
    for i in range(0, N, f):
        ax = axes[i//f]
        ax.axhline(0, c='k', lw=5)
        ax.grid()
        
        c = cList[i//f]
        [r1, R0, phi] = rRp_sol[i]
        Delta = Delta_values[i]
        
        Z0 = z2_fun2(R0, [r1, R0, phi])
        
        rr1 = np.linspace(r1, R0, num=100)
        zz1 = np.array([z1_fun2(r, [r1, R0, phi]) + Z0 for r in rr1])
        
        rr2 = np.linspace(R1, R0, num=100)
        zz2 = np.array([z2_fun2(r, [r1, R0, phi]) for r in rr2])
        
        ax.plot(rr1, zz1, c = c, zorder = 6, label = f'$\\delta$ = {Delta:.2f} µm')
        ax.plot(-rr1, zz1, c = c, zorder = 6)
        ax.plot(rr2, zz2, c = c, zorder = 6)
        ax.plot(-rr2, zz2, c = c, zorder = 6)
        
        CenterCell = h0 - R00
        AnglesCell = np.linspace(np.pi/2 - phi0, np.pi/2 + phi0, 200)
        RRCell = R00 * np.cos(AnglesCell)
        ZZCell = R00 * np.sin(AnglesCell) + CenterCell
        ax.plot(RRCell, ZZCell, 'k--', lw=0.6)
        
        CenterIndenter = h0 + Rp - Delta
        AnglesIndenter = np.linspace(0, 2*np.pi, 100)
        RRIndenter = Rp * np.cos(AnglesIndenter)
        ZZIndenter = Rp * np.sin(AnglesIndenter) + CenterIndenter
        ax.plot(RRIndenter, ZZIndenter, 'r-')
        
    for ax in axes:
        ax.legend(loc = 'upper right', fontsize = 9)
        locator = ticker.MultipleLocator(5)
        # ax.xaxis.set_major_locator(locator)
        # ax.yaxis.set_major_locator(locator)
        ax.tick_params(axis='both', labelsize=8)
        ax.axis('equal')
        w = max(R1, h0)
        ax.set_xlim([-w-1, +w+1])
        ax.set_ylim([-1, +2*w+2])
        # ax.axis('equal')
    
    plt.tight_layout()
    plt.show()
    

cell_contour_plot2(Delta_values, rRp_sol)


# %%% numeric plots

#### TBD



# %% Spherical indenter, general case

# %%% Constants

# Measured parameters
R1 = 9.59
h0 = 11.24
Rp = 25

# Computed parameters
signe_h = (h0-R1)/np.abs(h0-R1)
phi0 = np.arcsin(2*h0*R1 / (h0**2 + R1**2)) * (-signe_h) + np.pi * (h0>R1)
R00 = (h0**2 + R1**2) / (2*h0)
phi0_deg = phi0 * 180/np.pi
A0 = 2*np.pi*R00*h0
V0 = (np.pi/3)*h0**2*(3*R00 - h0)

# %%% Compute the delta-limit

def delta_limite(R1=8.5, h0=7, Rp=25,
                 plot = True):
    # Computed parameters
    signe_h = (h0-R1)/np.abs(h0-R1)
    phi0 = np.arcsin(2*h0*R1 / (h0**2 + R1**2)) * (-signe_h) + np.pi * (h0>R1)
    R00 = (h0**2 + R1**2) / (2*h0)
    V0 = (np.pi/3)*h0**2*(3*R00 - h0)
    
    # Define functions
    def my_sqrt(x):
        return(max(1e-10, x)**0.5)

    def A_lim(r1):
        return((R1 + ((r1**2)/Rp))/(R1**2 - r1**2))
        
    def B_lim(r1):
        return(-(r1**2)*(R1 + ((r1**2)/Rp))/(R1**2 - r1**2) - (r1**2)/Rp)

    def u_lim(r, r1):
        return((A_lim(r1)*r) + (B_lim(r1)/r))

    def A_cap(r1):
        return(2*np.pi*Rp*(Rp - my_sqrt(Rp**2 - r1**2)))

    def V_cap(r1):
        return((np.pi/3) * (Rp - my_sqrt(Rp**2 - r1**2))**2 * (3*Rp - (Rp - my_sqrt(Rp**2 - r1**2))))
    
    def integrand_z_lim(r, r1):
        return(u_lim(r, r1) / (my_sqrt(1 - u_lim(r, r1)**2)))

    def integrand_V_lim(r, r1):
        return(r**2 * (u_lim(r, r1) / (my_sqrt(1 - u_lim(r, r1)**2))))

    def z_lim_fun(r1):
        return(integrate.quad(integrand_z_lim, r1, R1, args=r1)[0])

    def Delta_lim_fun(r1):
        return(h0 - integrate.quad(integrand_z_lim, r1, R1, args=r1)[0] + Rp - my_sqrt(Rp**2 - r1**2))

    def V_lim_fun(r1):
        return(np.pi * integrate.quad(integrand_V_lim, r1, R1, args=r1)[0] - V_cap(r1))
    
    def z_lim_fun2(r, r1):
        return(integrate.quad(integrand_z_lim, r, R1, args=r1)[0])
    
    ### Solve shit
    r1_lim = 0
    r1_guess = R1/2
    
    def V_lim_0(r1):
        return(V0 - V_lim_fun(r1))
    
    r1_lim = fsolve(V_lim_0, r1_guess)[0]
    Delta_lim = Delta_lim_fun(r1_lim)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize = (8, 8))
        fig.suptitle(f'Configuration Limite - $\\Phi$ = $\\pi$/2 - $\\delta_l$ = {Delta_lim:.2f}')
        
        ax.axhline(0, c='k', lw=5)
        ax.grid()
        
        rr = np.linspace(r1_lim, R1, num=200)
        zz = np.array([z_lim_fun2(r, r1_lim) for r in rr])
        
        ax.plot(rr, zz, 'b-')
        ax.plot(-rr, zz, 'b-')
        
        CenterCell = h0 - R00
        AnglesCell = np.linspace(np.pi/2 - phi0, np.pi/2 + phi0, 200)
        RRCell = R00 * np.cos(AnglesCell)
        ZZCell = R00 * np.sin(AnglesCell) + CenterCell
        ax.plot(RRCell, ZZCell, 'b--', lw=0.8)
        
        CenterIndenter = h0 + Rp - Delta_lim
        AnglesIndenter = np.linspace(0, 2*np.pi, 100)
        RRIndenter = Rp * np.cos(AnglesIndenter)
        ZZIndenter = Rp * np.sin(AnglesIndenter) + CenterIndenter
        ax.plot(RRIndenter, ZZIndenter, 'r-')

        ax.axis('equal')
        ax.set_xlim([-R1-0.5, +R1+0.5])
        ax.set_ylim([h0-R1-0.5, h0+R1+0.5])
        
        fig.tight_layout()
        plt.show()
        
    return(Delta_lim)


Delta_lim = delta_limite(R1=R1, h0=h0, Rp=Rp, plot = True)