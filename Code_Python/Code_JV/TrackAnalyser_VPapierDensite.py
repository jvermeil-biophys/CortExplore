# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 10:18:47 2022

@author: anumi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 13:07:45 2022

@author: JosephVermeil & AnumitaJawahar
"""

# %% (0) Imports and settings

import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import scipy.interpolate as si

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import re
import os
import sys
import time
import random
import numbers
import warnings
import itertools


from copy import copy, deepcopy
from cycler import cycler
from datetime import date
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


#### Local Imports

import sys
import CortexPaths as cp
sys.path.append(cp.DirRepoPython)

import GraphicStyles as gs
import ArticlePlotMaker as apm
import UtilityFunctions as ufun


#### Errors for chi2

err_chi2_H = 7 # nm
err_chi2_F = 10 # pN
err_chi2_Stress = 100 # Pa
err_chi2_Strain = 0.01 # %

# %%% Smaller settings

# Pandas settings
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.max_columns', None)

# Plot settings
gs.set_default_options_jv()


####

dictSubstrates = {}
for i in range(5,105,5):
    dictSubstrates['disc' + str(i) + 'um'] = str(i) + 'um fibronectin discs'
    dictSubstrates['disc{:02.0f}um'.format(i)] = str(i) + 'um fibronectin discs'



# %% (2) Compressions experiments

#### Workflow
# * analyseTimeSeries_meca() analyse 1 file and return the dict (with the results of the analysis)
# * buildDf_meca() call the previous function on the given list of files and concatenate the results
# * computeGlobalTable_meca() call the previous function and convert the dict to a DataFrame



# %%%% Mechanical models

def VWC(h, K, Y, H0):
    f  = 3.1416*2250*(K/6*(H0**3*h**-2+2*h-3*H0)+Y/3*H0*(1-h/H0)**2) #including factor 3 (K/6)
    return f

def chadwickModel(h, E, H0, DIAMETER):
    """
    Implement the Chadwick formula with force as a function of thickness.

    Parameters
    ----------
    h : numpy array
        Array of cortical thickness in µm.
    E : float
        Elastic modulus of the cortex, in Pa.
    H0 : float
        Thickness of the cortex in the non-deformed configuration, in µm.
    DIAMETER : float
        Diameter of the beads indenting the cortex, in µm.

    Returns
    -------
    f : numpy array
        Array of pinching forces in pN.
    
    Reference
    -------
    Axisymmetric Indentation of a Thin Incompressible Elastic Layer, 
    R. S. Chadwick, 2002, https://doi.org/10.1137/S0036139901388222

    Note
    -------
    Compatible units: µm, pN, Pa ; or nm, pN, µPa.
    """
    
    R = DIAMETER/2
    f = (np.pi*E*R*((H0-h)**2))/(3*H0)
    return(f)


def inversedChadwickModel(f, E, H0, DIAMETER):
    """
    Implement the Chadwick formula with thickness as a function of force.

    Parameters
    ----------
    f : numpy array
        Array of pinching forces in pN.
    E : float
        Elastic modulus of the cortex, in Pa.
    H0 : float
        Thickness of the cortex in the non-deformed configuration, in µm.
    DIAMETER : float
        Diameter of the beads indenting the cortex, in µm.

    Returns
    -------
    h : numpy array
        Array of cortical thickness in µm.
        
    Reference
    -------
    Axisymmetric Indentation of a Thin Incompressible Elastic Layer, 
    R. S. Chadwick, 2002, https://doi.org/10.1137/S0036139901388222

    Note
    -------
    Compatible units: µm, pN, Pa ; or nm, pN, µPa.
    """
    R = DIAMETER/2
    h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
    return(h)


def dimitriadisModel(h, E, H0, DIAMETER, v = 0, order = 3):
    """
    Implement the Chadwick formula with force as a function of thickness.

    Parameters
    ----------
    h : numpy array
        Array of cortical thickness in µm.
    E : float
        Elastic modulus of the cortex, in Pa.
    H0 : float
        Thickness of the cortex in the non-deformed configuration, in µm.
    DIAMETER : float
        Diameter of the beads indenting the cortex, in µm.
    v : float, optional
        Poisson modulus of the cortex.
    order : int, optional
        Order of the polynomia used in the model. The default is 2.

    Returns
    -------
    f : numpy array
        Array of pinching forces in pN.
    
    Reference
    -------
    Determination of Elastic Moduli of Thin Layers of Soft Material Using the Atomic Force Microscope, 
    E. K. Dimitriadis et al., 2002, https://doi.org/10.1016/S0006-3495(02)75620-8
    
    Note
    -------
    Compatible units: µm, pN, Pa ; or nm, pN, µPa.
    """
    R = DIAMETER/2
    delta = H0-h
    X = np.sqrt(R*delta)/h
    ks = ufun.getDimitriadisCoefs(v, order)
    poly = np.zeros_like(X)
    for i in range(order+1):
        poly = poly + ks[i] * X**i
    f = ((4 * E * R**0.5 * delta**1.5)/(3 * (1 - v**2))) * poly
    return(f)


def constitutiveRelation(strain, K, stress0):
    """
    Implement the stress-strain locally affine function: stress = (K * strain) + stress0

    Parameters
    ----------
    strain : numpy array
        Array of strain.
    K : float
        Tangeantial modulus of the cortex, in Pa.
    stress0 : float
        Y-intercept for the stress.

    Returns
    -------
    stress : numpy array
        Array of stress in Pa.
    """
    
    stress = (K * strain) + stress0
    return(stress)


def inversedConstitutiveRelation(stress, K, strain0):
    """
    Implement the strain-stress locally affine function: strain = (stress / K) + strain0

    Parameters
    ----------
    stress : numpy array
        Array of stress in Pa.
    K : float
        Tangeantial modulus of the cortex, in Pa.
    strain0 : float
        Y-intercept for the strain.

    Returns
    -------
    strain : numpy array
        Array of strain.
    """
    
    strain = (stress / K) + strain0
    return(strain)

    

# %%%% General fitting functions

def fitVWC_hf(h, f, D):
    """
    Fit the VWC model on a force-thickness curve.
    This means the X-variable is h and the Y-variable is f.

    Parameters
    ----------
    h : numpy array
        Array of cortical thickness in nm.
    f : numpy array
        Array of pinching forces in pN.


    Returns
    -------
    params : (3 x 1) numpy array
        Parameters values as: [K, Y, H0]. [Van Wyk, Chadwick, H0]
    ses : (3 x 1) numpy array
        Standard errors for the parameters: [se(K), se(Y), se(H0)].
    error : bool
        Error during the fit.
        
    Note
    -------
    Units in the fits: nm, pN, µPa; that is why the modulus will be multiplied by 1e6.
    """
    
    Npts = len(h)
    error = False
    
    R = D/2
    
    def VWC(h, K, Y, H0):
        f = np.pi*R * (K/6*(H0**3 * h**(-2) + 2*h - 3*H0) + Y/3 * H0 * (1 - h/H0)**2) #including factor 3 (K/6)
        return(f)
    
    try:
        # some initial parameter values - must be within bounds\
       
        initH0 = h[0] + 10
        initK = 0.2*1e-3
        initY = 1*1e-3
        
        initialParameters = [initK, initY, initH0]
    
        # bounds on parameters - initial parameters must be within these
        lowerBounds = (0, 0, h[0]) #K, Y, H0
        upperBounds = (1e3, 1e6, 2*h[0]) #K, Y, H0
        parameterBounds = [lowerBounds, upperBounds]


        # params = [K, Y, H0] ; ses = [seK, seY, seH0]
        params, covM = curve_fit(VWC, h, f, p0=initialParameters, bounds = parameterBounds, method = 'trf')
        ses = np.array([covM[0,0]**0.5, covM[1,1]**0.5, covM[2,2]**0.5])
        params[0], params[1] = params[0]*1e6, params[1]*1e6
        ses[0], ses[1] = ses[0]*1e6, ses[1]*1e6  # Convert E & seE to Pa
        
    except:
        error = True
        params = np.ones(3) * np.nan
        ses = np.ones(3) * np.nan    
    res = (params, ses, error)
        
    return(res)
        

def fitChadwick_hf(h, f, D):
    """
    Fit the Chadwick model on a force-thickness curve, using the inversed model.
    This means the X-variable is f and the Y-variable is h.

    Parameters
    ----------
    h : numpy array
        Array of cortical thickness in µm.
    f : numpy array
        Array of pinching forces in pN.
    D : float
        Diameter of the beads indenting the cortex, in µm.

    Returns
    -------
    params : (2 x 1) numpy array
        Parameters values as: [E, H0].
    ses : (2 x 1) numpy array
        Standard errors for the parameters: [se(E), se(H0)].
    error : bool
        Error during the fit.
        
    Note
    -------
    Units in the fits: nm, pN, µPa; that is why the modulus will be multiplied by 1e6.
    """
    
    R = D/2
    Npts = len(h)
    error = False
    
    def chadwickModel(h, E, H0):
        f = (np.pi*E*R*((H0-h)**2))/(3*H0)
        return(f)

    def inversedChadwickModel(f, E, H0):
        h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
        return(h)

    try:
        # some initial parameter values - must be within bounds
        initH0 = max(h) # H0 ~ h_max
        initE = (3*max(h)*max(f))/(np.pi*(R)*(max(h)-min(h))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
        
        initialParameters = [initE, initH0]
        
        # bounds on parameters - initial parameters must be within these
        lowerBounds = (0, 0)
        upperBounds = (np.inf, np.inf)
        parameterBounds = [lowerBounds, upperBounds]
        
        
        # params = [E, H0] ; ses = [seE, seH0]
        params, covM = curve_fit(inversedChadwickModel, f, h, p0=initialParameters, bounds = parameterBounds)
        ses = np.array([covM[0,0]**0.5, covM[1,1]**0.5])
        params[0], ses[0] = params[0]*1e6, ses[0]*1e6 # Convert E & seE to Pa
        
    except:
        error = True
        params = np.ones(2) * np.nan
        ses = np.ones(2) * np.nan
        
    res = (params, ses, error)
        
    return(res)
        


def fitDimitriadis_hf(h, f, D, order = 2):
    """
    Fit the Dimitriadis model on a force-thickness curve.
    The X-variable is h and the Y-variable is f.

    Parameters
    ----------
    h : numpy array
        Array of cortical thickness in µm.
    f : numpy array
        Array of pinching forces in pN.
    D : float
        Diameter of the beads indenting the cortex, in µm.
    order : int, optional
        Order of the polynomia used in the model. The default is 2.

    Returns
    -------
    params : (2 x 1) numpy array
        Parameters values as: [E, H0].
    ses : (2 x 1) numpy array
        Standard errors for the parameters: [se(E), se(H0)].
    error : bool
        Error during the fit.
        
    Note
    -------
    Units in the fits: nm, pN, µPa; that is why the modulus will be multiplied by 1e6.
    """
    
    R = D/2
    error = False
    v = 0 # Poisson
    
    def dimitriadisModel(h, E, H0):
        
        delta = H0-h
        X = np.sqrt(R*delta)/h
        ks = ufun.getDimitriadisCoefs(v, order)
        
        poly = np.zeros_like(X)
        
        for i in range(order+1):
            poly = poly + ks[i] * X**i
            
        f = ((4 * E * (R**0.5) * (delta**1.5))/(3 * (1 - (v**2)))) * poly
        return(f)

    try:
        # some initial parameter values - must be within bounds
        # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
        initE = (3*max(h)*max(f))/(np.pi*(R)*(max(h)-min(h))**2) 
        # H0 ~ h_max
        initH0 = max(h) 
    
        # initH0, initE = initH0*(initH0>0), initE*(initE>0)
        
        initialParameters = [initE, initH0]
    
        # bounds on parameters - initial parameters must be within these
        lowerBounds = (0, max(h))
        upperBounds = (np.inf, np.inf)
        parameterBounds = [lowerBounds, upperBounds]

    
        # params = [E, H0] ; ses = [seE, seH0]
        params, covM = curve_fit(dimitriadisModel, h, f, p0=initialParameters, bounds = parameterBounds)
        ses = np.array([covM[0,0]**0.5, covM[1,1]**0.5])
        params[0], ses[0] = params[0]*1e6, ses[0]*1e6 # Convert E & seE to Pa
        
    except:
        error = True
        params = np.ones(2) * np.nan
        ses = np.ones(2) * np.nan
        
    res = (params, ses, error)
        
    return(res)



def fitLinear_ss(stress, strain, weights = []):
    """
    Linear fit on a stress-strain curve, with stress as the x- and strain as the y-variable.
    "_ss" stands for "Stress-Strain".

    Parameters
    ----------
    stress : (N x 1) numpy array
        Array of stress in Pa.
    strain : (N x 1) numpy array
        Array of strain.
    weights : (N x 1) numpy array, optional
        Array of weights. Can be interpreted in two ways, which condition how this function will work.
        
        * If weigths is an array of boolean-like values, it is interpreted as a mask, 
          to select only a part of the stress-strain curve.
        * If weigths is an array of float-like values, it is interpreted as weigths, 
          that apply to the stress-strain curve during the fit. This is typically used with gaussian weights.
        * If weigths = [], it is set as an array of True, 
          and the whole stress-strain curve is fitted without mask or weights.
        The default is [].

    Returns
    -------
    params : (2 x 1) numpy array
        Parameters values as: [K, strain0].
    ses : (2 x 1) numpy array
        Standard errors for the parameters: [se(K), se(strain0)].
    error : bool
        Error during the fit.
        
    Note
    -------
    Units in the fits: Pa.
    """
    
    error = False
    
    if len(weights) == 0:
        weights = np.ones_like(stress, dtype = bool)
    
    # Are all the weights equal to 0 or 1 ? 
    # If so it is a mask.
    masked = (np.all(weights.astype(bool) == weights))
    # If not they are interpreted as weigths.
    weighted = not masked 
    
    try:
    # if np.sum(weights) > 0:
        if masked:
            ols_model = sm.OLS(strain[weights], sm.add_constant(stress[weights]))
            results_ols = ols_model.fit()
            S0, K = results_ols.params[0], 1/results_ols.params[1]
            seS0, seK = results_ols.HC3_se[0], results_ols.HC3_se[1]*K**2 # See note below
        
        if weighted:
            wls_model = sm.WLS(strain, sm.add_constant(stress), weights=weights)
            results_wls = wls_model.fit()
            S0, K = results_wls.params[0], 1/results_wls.params[1]
            seS0, seK = results_wls.HC3_se[0], results_wls.HC3_se[1]*K**2 # See note below
            
            # NB : Here we fit strain = L * stress + S0 ; Let's write: L = 1/K
            # To find K, we just take 1/params[1] = 1/L
            # To find seK, there is a trick : se(K) / K = relative error = se(L) / L ; so se(K) = se(L) * K**2
            
        params = np.array([K, S0])
        ses = np.array([seK, seS0])
        
    except:
    # else:
        error = True
        params = np.ones(2) * np.nan
        ses = np.ones(2) * np.nan

    res = (params, ses, error)
        
    return(res)

def fitLinear_ss_i(strain, stress, weights = []):
    """
    Linear fit on a stress-strain curve, with strain as the x- and stress as the y-variable.
    "_ss_i" stands for "Stress-Strain, Inversed".

    Parameters
    ----------
    strain : (N x 1) numpy array
        Array of strain.
    stress : (N x 1) numpy array
        Array of stress in Pa.
    weights : (N x 1) numpy array, optional
        Array of weights. Can be interpreted in two ways, which condition how this function will work.
        
        * If weigths is an array of boolean-like values, it is interpreted as a mask, 
          to select only a part of the stress-strain curve.
        * If weigths is an array of float-like values, it is interpreted as weigths, 
          that apply to the stress-strain curve during the fit. This is typically used with gaussian weights.
        * If weigths = [], it is set as an array of True, 
          and the whole stress-strain curve is fitted without mask or weights.
        The default is [].

    Returns
    -------
    params : (2 x 1) numpy array
        Parameters values as: [K, strain0].
    ses : (2 x 1) numpy array
        Standard errors for the parameters: [se(K), se(strain0)].
    error : bool
        Error during the fit.
        
    Note
    -------
    Units in the fits: Pa.
    """
    
    error = False
    
    if len(weights) == 0:
        weights = np.ones_like(strain, dtype = bool)
    
    # Are all the weights equal to 0 or 1 ? 
    # If so it is a mask.
    masked = (np.all(weights.astype(bool) == weights))
    # If not they are interpreted as weigths.
    weighted = not masked 
    
    try:
    # if np.sum(weights) > 0:
        if masked:
            ols_model = sm.OLS(stress[weights], sm.add_constant(strain[weights]))
            results_ols = ols_model.fit()
            S0, K = results_ols.params[0], results_ols.params[1]
            seS0, seK = results_ols.HC3_se[0], results_ols.HC3_se[1]
        
        if weighted:
            wls_model = sm.WLS(stress, sm.add_constant(strain), weights=weights)
            results_wls = wls_model.fit()
            S0, K = results_wls.params[0], results_wls.params[1]
            seS0, seK = results_wls.HC3_se[0], results_wls.HC3_se[1]
            
            
        params = np.array([K, S0])
        ses = np.array([seK, seS0])
        
    except:
    # else:
        error = True
        params = np.ones(2) * np.nan
        ses = np.ones(2) * np.nan

    res = (params, ses, error)
        
    return(res)


def fitChadwick_hf_fixedH0(h, f, D, H0):
    """
    Fit the Chadwick model on a force-thickness curve, using the inversed model.
    This means the X-variable is f and the Y-variable is h.

    Parameters
    ----------
    h : numpy array
        Array of cortical thickness in µm.
    f : numpy array
        Array of pinching forces in pN.
    D : float
        Diameter of the beads indenting the cortex, in µm.

    Returns
    -------
    params : (2 x 1) numpy array
        Parameters values as: [E, H0].
    ses : (2 x 1) numpy array
        Standard errors for the parameters: [se(E), se(H0)].
    error : bool
        Error during the fit.
        
    Note
    -------
    Units in the fits: nm, pN, µPa; that is why the modulus will be multiplied by 1e6.
    """
    
    R = D/2
    Npts = len(h)
    error = False
    
    def chadwickModel(h, E):
        f = (np.pi*E*R*((H0-h)**2))/(3*H0)
        return(f)

    def inversedChadwickModel(f, E):
        h = H0 - ((3*H0*f)/(np.pi*E*R))**0.5
        return(h)

    try:
        # some initial parameter values - must be within bounds
        # initH0 = max(h) # H0 ~ h_max
        initE = (3*max(h)*max(f))/(np.pi*(R)*(max(h)-min(h))**2) # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
        
        initialParameters = [initE]
    
        # bounds on parameters - initial parameters must be within these
        lowerBounds = (0)
        upperBounds = (np.inf)
        parameterBounds = [lowerBounds, upperBounds]


        # params = [E, H0] ; ses = [seE, seH0]
        params, covM = curve_fit(inversedChadwickModel, f, h, p0=initialParameters, bounds = parameterBounds)
        
        ses = np.array([covM[0,0]**0.5])
        params[0], ses[0] = params[0]*1e6, ses[0]*1e6 # Convert E & seE to Pa
        
    except:
        error = True
        params = np.ones(1) * np.nan
        ses = np.ones(1) * np.nan
    
    res = (params, ses, error)

    return(res)


#%%%% makeDictFit


def makeDictFit_CVW_hf(params, ses, error, 
                   x, y, yPredict,  kPredict,  ePredict,
                   err_chi2, fitValidationSettings):
    """
    Take multiple inputs related to the fit of a **force-thickness** curve, 
    and compute detailed results contained in a dict.

    Parameters
    ----------
    params : (3 x 1) numpy array
        Parameters values as: [K, Y, H0]. 
    ses : (3 x 1) numpy array
        Standard errors for the parameters: [se(K), se(Y), se(H0)].
    error : bool
        Error during the fit. From the function fitVWC_hf().
    x : (N x 1) numpy array
        The x-variable values array used for the fit.
    y : (N x 1) numpy array
        The y-variable values array used for the fit.
    yPredict : (N x 1) numpy array
        The x-variable values array predicted from the fit.
    err_chi2 : float
        The typical error on the y-variable used to compute the chi2.
        See in the top of this code for default values of errors used in Chi2 computations.
    fitValidationSettings : dict
        Dictionary that contains the validation criteria for nbPts, R2 and Chi2.

    Returns
    -------
    res : dict, contains the following fields : 
        * 'error' : bool, error of the fit as given in input.
        * 'nbPts' : int, number of points fitted.
        * 'E', 'seE', 'H0', 'seH0' : float, params and ses as given in input.
        * 'R2', 'Chi2' : float, R2 and Chi2 as computed using inputs x, y, and yPredict.
        * 'ciwE', 'ciwH0' : float, Confidence Interval Width for the parameters.
        * 'x', 'y', 'yPredict' : numpy array, the arrays given as input.
        * 'valid': bool, wether or not the fit is validated with respect to the criteria in fitValidationSettings.
        * 'issue': string, a text describing the reasons why a fit was not validated if it is the case.
    
    Note
    -------
    1. The inputs params, ses, error should be taken from the output of the functions 
       **fitChadwick_hf()** or **fitDimitriadis_hf()** or **fitVWC_hf**.
    
    2. How to compute confidence intervals of fitted parameters with (1-alpha) confidence:
        i) from scipy import stats
        ii) df = nb_pts - nb_parms ; se = diag(cov)**0.5
        iii) Student t coefficient : q = stat.t.ppf(1 - alpha / 2, df)
        iv) ConfInt = [params - q*se, params + q*se]

    """
    if not error:
        K, Y, H0 = params
        seK, seY, seH0 = ses

        alpha = 0.975
        dof = len(y)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(y, yPredict)
        Chi2 = ufun.get_Chi2(y, yPredict, dof, err_chi2)        

        ciwK = q*seK
        ciwY = q*seY
        ciwH0 = q*seH0
        
        nbPts = len(y)
        
        isValidated = (K > 0 and Y > 0 and
                       nbPts >= fitValidationSettings['crit_nbPts'] and
                       R2 >= fitValidationSettings['crit_R2'] and 
                       Chi2 <= fitValidationSettings['crit_Chi2'])
        issue = ''
        if isValidated:
            issue += 'none'
        else:
            if not K > 0:
                issue += 'K<0_'
            if not Y > 0:
                issue += 'Y<0_'
            if not nbPts >= fitValidationSettings['crit_nbPts']:
                issue += 'nbPts<{:.0f}_'.format(fitValidationSettings['crit_nbPts'])
            if not R2 >= fitValidationSettings['crit_R2']:
                issue += 'R2<{:.2f}_'.format(fitValidationSettings['crit_R2'])
            if not Chi2 >= fitValidationSettings['crit_Chi2']:
                issue += 'Chi2>{:.1f}_'.format(fitValidationSettings['crit_Chi2'])
    
    else:
        K, Y, H0, seK, seY, seH0 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        R2, Chi2 = np.nan, np.nan
        ciwK, ciwY, ciwH0 = np.nan, np.nan, np.nan
        isValidated = False
        issue = 'error'
    

    res =  {'error': error,
            'nbPts':len(y),
            'K':K, 'seK':seK,
            'Y':Y, 'seY':seY,
            'H0':H0, 'seH0':seH0,
            'R2':R2, 'Chi2':Chi2,
            'ciwK':ciwK, 
            'ciwY':ciwY, 
            'ciwH0':ciwH0,
            'x': x,
            'y': y,
            'yPredict': yPredict,
            'kPredict': kPredict,
            'ePredict': ePredict,
            'valid': isValidated,
            'issue': issue
            }
    
    return(res)

def makeDictFit_hf(params, ses, error, 
                   x, y, yPredict, 
                   err_chi2, fitValidationSettings):
    """
    Take multiple inputs related to the fit of a **force-thickness** curve, 
    and compute detailed results contained in a dict.

    Parameters
    ----------
    params : (2 x 1) numpy array
        Parameters values as: [E, H0]. 
    ses : (2 x 1) numpy array
        Standard errors for the parameters: [se(E), se(H0)].
    error : bool
        Error during the fit. From the function fitChadwick_hf().
    x : (N x 1) numpy array
        The x-variable values array used for the fit.
    y : (N x 1) numpy array
        The y-variable values array used for the fit.
    yPredict : (N x 1) numpy array
        The x-variable values array predicted from the fit.
    err_chi2 : float
        The typical error on the y-variable used to compute the chi2.
        See in the top of this code for default values of errors used in Chi2 computations.
    fitValidationSettings : dict
        Dictionary that contains the validation criteria for nbPts, R2 and Chi2.

    Returns
    -------
    res : dict, contains the following fields : 
        * 'error' : bool, error of the fit as given in input.
        * 'nbPts' : int, number of points fitted.
        * 'E', 'seE', 'H0', 'seH0' : float, params and ses as given in input.
        * 'R2', 'Chi2' : float, R2 and Chi2 as computed using inputs x, y, and yPredict.
        * 'ciwE', 'ciwH0' : float, Confidence Interval Width for the parameters.
        * 'x', 'y', 'yPredict' : numpy array, the arrays given as input.
        * 'valid': bool, wether or not the fit is validated with respect to the criteria in fitValidationSettings.
        * 'issue': string, a text describing the reasons why a fit was not validated if it is the case.
    
    Note
    -------
    1. The inputs params, ses, error should be taken from the output of the functions 
       **fitChadwick_hf()** or **fitDimitriadis_hf()**.
    
    2. How to compute confidence intervals of fitted parameters with (1-alpha) confidence:
        i) from scipy import stats
        ii) df = nb_pts - nb_parms ; se = diag(cov)**0.5
        iii) Student t coefficient : q = stat.t.ppf(1 - alpha / 2, df)
        iv)  ConfInt = [params - q*se, params + q*se]

    """
    if not error:
        E, H0 = params
        seE, seH0 = ses

        alpha = 0.975
        dof = len(y)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(y, yPredict)
        Chi2 = ufun.get_Chi2(y, yPredict, dof, err_chi2)        

        ciwE = q*seE
        ciwH0 = q*seH0
        
        nbPts = len(y)
        
        isValidated = (E > 0 and
                       nbPts >= fitValidationSettings['crit_nbPts'] and
                       R2 >= fitValidationSettings['crit_R2'] and 
                       Chi2 <= fitValidationSettings['crit_Chi2'])
        issue = ''
        if isValidated:
            issue += 'none'
        else:
            if not E > 0:
                issue += 'E<0_'
            if not nbPts >= fitValidationSettings['crit_nbPts']:
                issue += 'nbPts<{:.0f}_'.format(fitValidationSettings['crit_nbPts'])
            if not R2 >= fitValidationSettings['crit_R2']:
                issue += 'R2<{:.2f}_'.format(fitValidationSettings['crit_R2'])
            if not Chi2 >= fitValidationSettings['crit_Chi2']:
                issue += 'Chi2>{:.1f}_'.format(fitValidationSettings['crit_Chi2'])
    
    else:
        E, seE, H0, seH0 = np.nan, np.nan, np.nan, np.nan
        R2, Chi2 = np.nan, np.nan
        ciwE, ciwH0 = np.nan, np.nan
        isValidated = False
        issue = 'error'

    res =  {'error': error,
            'nbPts':len(y),
            'E':E, 'seE':seE,
            'H0':H0, 'seH0':seH0,
            'R2':R2, 'Chi2':Chi2,
            'ciwE':ciwE, 
            'ciwH0':ciwH0,
            'x': x,
            'y': y,
            'yPredict': yPredict,
            'valid': isValidated,
            'issue': issue
            }
    
    return(res)


def makeDictFit_ss(params, ses, error, 
                   center, halfWidth, x, y, yPredict, 
                   err_chi2, fitValidationSettings):
    """
    Take multiple inputs related to the fit of a **stress-strain** curve, 
    and compute detailed results contained in a dict.

    Parameters
    ----------
    params : (2 x 1) numpy array
        Parameters values as: [K, s0]. 
    ses : (2 x 1) numpy array
        Standard errors for the parameters: [se(K), se(s0)].
    error : bool
        Error during the fit. From the function fitChadwick_hf().
    x : (N x 1) numpy array
        The x-variable values array used for the fit.
    y : (N x 1) numpy array
        The y-variable values array used for the fit.
    yPredict : (N x 1) numpy array
        The x-variable values array predicted from the fit.
    err_chi2 : float
        The typical error on the y-variable used to compute the chi2.
        See in the top of this code for default values of errors used in Chi2 computations.
    fitValidationSettings : dict
        Dictionary that contains the validation criteria for nbPts, R2 and Chi2.

    Returns
    -------
    res : dict, contains the following fields : 
        * 'error' : bool, error of the fit as given in input.
        * 'nbPts' : int, number of points fitted.
        * 'K', 'seK' : float, params and ses as given in input.
        * 'R2', 'Chi2' : float, R2 and Chi2 as computed using inputs x, y, and yPredict.
        * 'ciwK' : float, Confidence Interval Width for the parameters.
        * 'center_x', 'halfWidth_x', 'center_y' : float, centers and half-width of the arrays given as input.
        * 'x', 'y', 'yPredict' : numpy array, the arrays given as input.
        * 'valid': bool, wether or not the fit is validated with respect to the criteria in fitValidationSettings.
        * 'issue': string, a text describing the reasons why a fit was not validated if it is the case.
    
    Note
    -------
    1. The inputs params, ses, error should be taken from the output of the function
       **fitLinear_ss()**.
    
    2. How to compute confidence intervals of fitted parameters with (1-alpha) confidence:
        i) from scipy import stats
        ii) df = nb_pts - nb_parms ; se = diag(cov)**0.5
        iii) Student t coefficient : q = stat.t.ppf(1 - alpha / 2, df)
        iv) ConfInt = [params - q*se, params + q*se]
    """
    
    nbPts = len(y)

    if (not error) and (nbPts > 0):
        K, S0 = params
        seK, seS0 = ses

        alpha = 0.975
        dof = len(y)-len(params)
        q = st.t.ppf(alpha, dof) # Student coefficient
        R2 = ufun.get_R2(y, yPredict)
        Chi2 = ufun.get_Chi2(y, yPredict, dof, err_chi2)        

        ciwK = q*seK
        
        center_y = np.median(y)
        
        isValidated = (K > 0 and
                       nbPts >= fitValidationSettings['crit_nbPts'] and
                       R2 >= fitValidationSettings['crit_R2'] and 
                       Chi2 <= fitValidationSettings['crit_Chi2'])
        issue = ''
        if isValidated:
            issue += 'none'
        else:
            if not K > 0:
                issue += 'K<0_'
            if not nbPts >= fitValidationSettings['crit_nbPts']:
                issue += 'nbPts<{:.0f}_'.format(fitValidationSettings['crit_nbPts'])
            if not R2 >= fitValidationSettings['crit_R2']:
                issue += 'R2<{:.2f}_'.format(fitValidationSettings['crit_R2'])
            if not Chi2 <= fitValidationSettings['crit_Chi2']:
                issue += 'Chi2>{:.1f}_'.format(fitValidationSettings['crit_Chi2'])
    
    else:
        error = True
        K, seK = np.nan, np.nan
        R2, Chi2 = np.nan, np.nan
        nbPts = np.nan
        ciwK = np.nan
        center_y = np.nan
        isValidated = False
        issue = 'error'

    res =  {'error': error,
            'nbPts': nbPts,
            'K': K, 'seK': seK,
            'R2': R2, 'Chi2': Chi2,
            'ciwK': ciwK,
            'center_x': center,
            'halfWidth_x': halfWidth,
            'center_y': center_y,
            'x': x,
            'y': y,
            'yPredict': yPredict,
            'valid': isValidated,
            'issue': issue
            }
    
    return(res)


def nestedDict_to_DataFrame(D):
    """
    Convert D, a dict of dicts, into a pandas DataFrame, after removing non-numeric values.
    
    D = {'d1' : d1, 'd2' : d2, ..., 'dn' : dn} where each di = {'k1' : val1, 'k2' : val2, ..., 'kn' : valn},
    and the keys k1 ... kn are the same for all dicts d1 ... dn.
    
    In the resulting DataFrame, k1 ... kn will be the columns, and d1 ... dn the rows.
    
    This function remove any couple 'ki':val_i where val_i is not a number (e.g. a numy array).
    """
    if len(D) > 0:
        fits =  list(D.keys())
        Nfits = len(fits)
        f0 = fits[0]
        d0 = D[f0]
        
        cols_others = [k for k in d0.keys() if (k in ['valid', 'issue'])]
        cols_num = [k for k in d0.keys() if (isinstance(d0[k], numbers.Number)) \
                                            and (k not in cols_others)]
        

        D2 = {}
        index = [] # will contains all the names of the small dicts 'd'
        
        for c in cols_num:
            D2[c] = np.zeros(Nfits)
        for c in cols_others:
            D2[c] = []
    
        for jj in range(Nfits):
            f = fits[jj]
            d = D[f]
            for c in cols_others:
                D2[c].append(d[c])
            for c in cols_num:
                D2[c][jj] = d[c]
            index.append(f)
            
    else:
        D2 = {}
        index = []
        
    df = pd.DataFrame(D2)
    df.insert(0, 'id', index)
    return(df)



        

# %%%% Classes
        

class CellCompression:
    """
    This class deals with all that requires the whole array of compressions.
    
    ==========
    Attributes
    ==========
    
    Parameters of the constructor
    -----------------------------
    
    cellID : string
        Id of the analysed cell, in the format 'yy-mm-dd_M#_P#_C#'
    tsDf : pandas DataFrame
        TimeSeries DataFrame containing the data for the given cell
    expDf : pandas DataFrame
        TimeSeries DataFrame containing the experimental conditions
    fileName : string
        Name of the .csv file imported to make tsDf
    
    Attributes filled during construction
    -------------------------------------
    
    Ncomp : int
        Number of compressions in tsDf.
    DIAMETER : float
        Diameter of the beads used, in nm. Imported from expDf.
    EXPTYPE : string
        Type of experiment. Imported from expDf.
    normalField : float
        Magnetic field applied between compressions. Imported from expDf.
    minCompField : float
        Min magnetic field applied during compressions. Imported from expDf.
    maxCompField : float
        Max magnetic field applied during compressions. Imported from expDf.
    loopStruct : string
        Text describing the loop structure, in the format 'N1_N2', 
        where N1 is the total number of images per loop, 
        and N2 is the number of compression images per loop. Imported from expDf.
    nUplet : int
        Number of images within the Z-stacks outside of compressions.
    loop_totalSize : int
        Value of N1 from loopStruct, ie the total number of images per loop.
    loop_rampSize : int
        Value of N2 from loopStruct, ie the number of compression images per loop.
    loop_ctSize : int
        Value of N1-N2 from loopStruct, ie the number of non-compression images per loop
    
    Attributes filled by other methods
    ----------------------------------
    
    
    listIndent : list
        List that will contains pointers toward all the IndentCompressions objects
        created in the main function. Filled in the main function analyseTimeSeries_meca().
    listJumpsD3 : (N x 1) numpy array of floats
        
    df_mainResults : pandas DataFrame
        
    df_stressRegions : pandas DataFrame
        
    df_stressGaussian : pandas DataFrame
        
    df_nPoints : pandas DataFrame
            
    
    =======
    Methods
    =======
    
    
    
    =============
    In development
    ==============
    
    
	
    
    
    """
    
    def __init__(self, cellID, timeseriesDf, thisExpDf, fileName):
        self.tsDf = timeseriesDf
        self.expDf = thisExpDf
        self.cellID = cellID
        self.fileName = fileName
        
        Ncomp = max(timeseriesDf['idxAnalysis'])
        
        try:
            DIAMETER = int(thisExpDf.at[thisExpDf.index.values[0], 'bead diameter'])
        except:
            D1 = int(thisExpDf.at[thisExpDf.index.values[0], 'inside bead diameter'])
            D2 = int(thisExpDf.at[thisExpDf.index.values[0], 'outside bead diameter'])
            DIAMETER = (D1 + D2)/2.
        
        EXPTYPE = str(thisExpDf.at[thisExpDf.index.values[0], 'experimentType'])
        
        # Field infos
        normalField = int(thisExpDf.at[thisExpDf.index.values[0], 'normal field'])
        
        compField = thisExpDf.at[thisExpDf.index.values[0], 'ramp field'].split('_')
        minCompField = float(compField[0])
        maxCompField = float(compField[1])
        
        nUplet = thisExpDf.at[thisExpDf.index.values[0], 'normal field multi images']
        
        self.Ncomp = Ncomp
        self.DIAMETER = DIAMETER
        self.EXPTYPE = EXPTYPE
        self.normalField = normalField
        self.minCompField = minCompField
        self.maxCompField = maxCompField
        self.nUplet = nUplet
        
        
        # Loop structure infos
        try:
            loopStruct = thisExpDf.at[thisExpDf.index.values[0], 'loop structure'].split('_')
            loop_totalSize = int(loopStruct[0])
            loop_rampSize = int(loopStruct[1])
            loop_ctSize = int((loop_totalSize - loop_rampSize)/nUplet)
            
            self.loopStruct = loopStruct
            self.loop_totalSize = loop_totalSize
            self.loop_rampSize = loop_rampSize
            self.loop_ctSize = loop_ctSize
        except:
            try:
                t1 = self.tsDf['idxLoop'].size/np.max(self.tsDf['idxLoop'])
                t2 = self.tsDf['idxLoop'].size//np.max(self.tsDf['idxLoop'])
                # print(t1 == t2)
            except:
                print('Loop structure Error :(')
        
        
        # These fields are to be filled by methods later on
        self.listIndent = []
        self.listJumpsD3 = np.zeros(Ncomp, dtype = float)
        # self.method_bestH0 = 'Dimitriadis' # default
        
        self.df_mainResults = pd.DataFrame({})
        self.df_stressRegions = pd.DataFrame({})
        self.df_stressGaussian = pd.DataFrame({})
        self.df_nPoints = pd.DataFrame({})
        self.df_log = pd.DataFrame({})
        
        # in dev
        self.df_strainGaussian = pd.DataFrame({})
        self.df_3parts = pd.DataFrame({}) #### TEST

        
    def getMaskForCompression(self, i, task = 'compression'):
        try:
            Npts = len(self.tsDf['idxAnalysis'].values)
            iStart = ufun.findFirst(np.abs(self.tsDf['idxAnalysis']), i+1)
            iStop = iStart + np.sum((np.abs(self.tsDf['idxAnalysis']) == i+1))
            
            if task == 'compression':
                mask = (self.tsDf['idxAnalysis'] == i+1).values
            elif task == 'precompression':
                mask = (self.tsDf['idxAnalysis'] == -(i+1)).values
            elif task == 'compression & precompression':
                mask = (np.abs(self.tsDf['idxAnalysis']) == i+1).values
            elif task == 'previous':
                i1 = max(0, iStart-(self.loop_ctSize))
                i2 = iStart
                mask = np.array([((i >= i1) and (i < i2)) for i in range(Npts)])
            elif task == 'following':
                i1 = iStop
                i2 = min(Npts, iStop+(self.loop_ctSize))
                mask = np.array([((i >= i1) and (i < i2)) for i in range(Npts)])
            elif task == 'surrounding':
                i1 = max(0,    iStart-(self.loop_ctSize//2))
                i2 = min(Npts, iStop +(self.loop_ctSize//2))
                mask = np.array([((i >= i1) and (i < i2)) for i in range(Npts)])
        
        except:
            Npts = len(self.tsDf['idxAnalysis'].values)
            iStart = ufun.findFirst(np.abs(self.tsDf['idxAnalysis']), i+1)
            iStop = iStart + np.sum((np.abs(self.tsDf['idxAnalysis']) == i+1))
            
            if task == 'compression':
                mask = (self.tsDf['idxAnalysis'] == i+1).values
            elif task == 'precompression':
                mask = (self.tsDf['idxAnalysis'] == -(i+1)).values
            elif task == 'compression & precompression':
                mask = (np.abs(self.tsDf['idxAnalysis']) == i+1).values
            elif task == 'previous':
                i2 = ufun.findFirst_V2(i+1, np.abs(self.tsDf['idxAnalysis']))
                if i2 == -1:
                    i2 = self.tsDf['idxAnalysis'].size
                i1 = Npts - ufun.findFirst_V2(i, np.abs(self.tsDf['idxAnalysis'][::-1]))
                if i == 0:
                    i1 = 0
                mask = np.array([((j >= i1) and (j < i2)) for j in range(Npts)])
            elif task == 'following':
                i2 = ufun.findFirst_V2(i+2, np.abs(self.tsDf['idxAnalysis']))
                if i2 == -1:
                    i2 = self.tsDf['idxAnalysis'].size
                i1 = Npts - ufun.findFirst_V2(i+1, np.abs(self.tsDf['idxAnalysis'][::-1]))
                if i1 > Npts:
                    i1 = 0
                mask = np.array([((j >= i1) and (j < i2)) for j in range(Npts)])
            elif task == 'surrounding':
                mask = ((np.abs(self.tsDf['idxLoop']) == i+1) & (np.abs(self.tsDf['idxAnalysis']) == 0)).values
                
        return(mask)
        
        
        
    def correctJumpForCompression(self, i):
        colToCorrect = ['dx', 'dy', 'dz']
        mask = self.getMaskForCompression(i, task = 'compression & precompression')
        iStart = ufun.findFirst(np.abs(self.tsDf['idxAnalysis']), i+1)
        for c in colToCorrect:
            #### CHANGE HERE: iStart+5 -> iStart+15
            jump = np.median(self.tsDf[c].values[iStart:iStart+5]) - np.median(self.tsDf[c].values[iStart-2:iStart])
            self.tsDf.loc[mask, c] -= jump
        
        #### D2
        newD2 = (self.tsDf.loc[mask, 'dx']**2 + self.tsDf.loc[mask, 'dy']**2)**0.5
        self.tsDf.loc[mask, 'D2'] = newD2
        #### D3
        newD3 = (newD2**2 + self.tsDf.loc[mask, 'dz']**2)**0.5
        jumpD3 = np.mean(newD3 - self.tsDf.loc[mask, 'D3'])
        D3corrected = True
        self.listJumpsD3[i] = jumpD3
        self.tsDf.loc[mask, 'D3'] = newD3
        
        
        
        
    
    def Pplot_Timeseries(self, plotSettings):
        
        fig, axes = plt.subplots(1, 1, figsize=(17/gs.cm_in, 7/gs.cm_in), layout='compressed')
        apm.setGraphicOptions(mode = 'print', 
                              palette = 'Set2', 
                              colorList = apm.cL_Set21)
        
        #### First Plot
        ax = axes
        # Distance axis
        color = gs.colorList40[30] # 'skyblue'# 'blue'
        ax.set_xlabel('Time (s)', labelpad=1)
        ax.set_ylabel('Thickness (nm)', color=color, labelpad=1)
        ax.tick_params(axis='x')#, labelsize=8)
        ax.tick_params(axis='y', labelcolor=color)#, labelsize=8)
        ax.scatter(self.tsDf['T'].values, self.tsDf['D3'].values-self.DIAMETER, 
                color = color, zorder = 5, s = 3)
        # ax.scatter(self.tsDf['T'].values[idx], self.tsDf['D3'].values[idx]-self.DIAMETER, 
        #         color = color, ls = '--', linewidth = 1, zorder = 1, s = 4)
        
        for ii in range(self.Ncomp): # (nLoops): #
            IC = self.listIndent[ii]
            compValid = IC.isValidForAnalysis
            if compValid:
                fitError = IC.dictFitFH_Chadwick['Full']['error']

                if (not fitError):                
                    ax.scatter(IC.Df['T'].values, IC.Df['D3'].values-self.DIAMETER, s = 3,
                            color = 'chartreuse', zorder = 6)
                    # if not IC.error_bestH0:
                    #     ax.plot(IC.Df['T'].values[0], IC.bestH0, 
                    #             color = '#b29600', marker = '*', markersize = 4, zorder = 3)  
                else:
                    ax.scatter(IC.Df['T'].values, IC.Df['D3'].values-self.DIAMETER, s = 3,
                            color = 'crimson', zorder = 6)
            else:
                ax.scatter(IC.Df['T'].values, IC.Df['D3'].values-self.DIAMETER, s = 3,
                        color = 'crimson', zorder = 6)
        (axm, axM) = ax.get_ylim()
        ax.set_ylim([min(-0,axm), axM])
        ax.set_xlim([0, min(185, ax.get_xlim()[-1])])
        ax.grid(axis='y', zorder=0)
        ax.tick_params(axis='y', labelcolor = color)

        
        # Force axis
        ax1bis = ax.twinx()
        color = 'firebrick'
        # ax1bis.set_ylabel('Force (pN)', color=color)
        ax1bis.set_ylabel('Force (nN)', color=color)
        # ax1bis.plot(self.tsDf['T'].values[idx], self. tsDf['F'].values[idx], color=color)
        ax1bis.plot(self.tsDf['T'].values, self. tsDf['F'].values/1e3, color=color, lw=1.5)
        ax1bis.tick_params(axis='y', labelcolor=color)
        ax1bis.set_yticks([0, 0.5, 1.0, 1.5])
        minh = np.min(self.tsDf['D3'].values - self.DIAMETER)
        ratio = min(1/abs(minh/axM), 5)
        (axmbis, axMbis) = ax1bis.get_ylim()
        # ax1bis.set_ylim([0, max(axMbis*ratio, 3*max(self.tsDf['F'].values))])
        ax1bis.set_ylim([0, max(axMbis*ratio, 3*max(self.tsDf['F'].values/1e3))])
        
        # all_axes = [axes[0], ax1bis, axes[1], ax2bis]
        # fig.tight_layout()
        return(fig, axes)
    
    
    def Pplot_Timeseries_V2(self, plotSettings):
        
        apm.setGraphicOptions(mode = 'print', 
                              palette = 'Set2', 
                              colorList = apm.cL_Set21)
        
        fig = plt.figure(figsize=(17/gs.cm_in, 6/gs.cm_in))
        spec = fig.add_gridspec(6, 1, hspace=0.15, top = 0.975, bottom=0.125, left = 0.065, right = 0.99)
        ax1 = fig.add_subplot(spec[:4])
        ax2 = fig.add_subplot(spec[4:])
        
        #### Distance Plot
        ax = ax1
        color = 'steelblue' # gs.colorList40[30] # 'skyblue'# 'blue'
        ax1.set_ylabel('Thickness (nm)', color=color, labelpad=1)
        ax.tick_params(axis='y', labelcolor=color)#, labelsize=8)
        ax.scatter(self.tsDf['T'].values, self.tsDf['D3'].values-self.DIAMETER, 
                   color = color, edgecolors = None, linewidths = 0,
                   zorder = 5, s = 2, alpha = 0.9)        

        (ax_ym, ax_yM) = ax.get_ylim()
        ax.set_ylim([min(-0,ax_ym), ax_yM])
        ax.set_ylim([0, 600])
        ax.tick_params(axis='y', labelcolor = color)
        ax.grid(axis='y', zorder=0)
        
        # Force plot
        ax = ax2
        color = 'firebrick'
        ax.set_ylabel('Force (nN)', color=color, labelpad=1)
        ax.plot(self.tsDf['T'].values, self. tsDf['F'].values/1e3, color=color, lw=1.5)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        ax.grid(axis='y', zorder=0)
        (ax_ym, ax_yM) = ax.get_ylim()
        # ax.set_ylim([0, 1.05*max(self.tsDf['F'].values/1e3)])
        ax.set_ylim([0, 1.1])
        
        #### Shades
        for i in range(1, len(self.listIndent)+1):
            df = self.tsDf[self.tsDf['idxLoop'] == (i)]
            # print(df.idxAnalysis)
            t1 = df['T'].values[ufun.findFirst(-i, df.idxAnalysis)]
            t2 = df['T'].values[ufun.findFirst(i, df.idxAnalysis)]
            t3 = df['T'].values[ufun.findLast(i, df.idxAnalysis)]
            t4 = df['T'].values[ufun.findLast(-i, df.idxAnalysis)]
            for ax in [ax1, ax2]:
                ax.axvspan(t1, t2, color='grey', alpha=0.15, zorder = 0, ec=None)
                ax.axvspan(t2, t3, color='grey', alpha=0.3, zorder = 0, ec=None)
        
        LM0 = mpatches.Rectangle((0, 0), 0, 0, facecolor='w', 
                                 edgecolor='k', linewidth=0.2,
                                 label='Constant field')
        LM1 = mpatches.Rectangle((0, 0), 0, 0, color='grey', alpha=0.15, linewidth=0,
                                   label='Force release')
        LM2 = mpatches.Rectangle((0, 0), 0, 0, color='grey', alpha=0.3, linewidth=0,
                                   label='Compression and relaxation')
        LegendHandles = [LM0, LM1, LM2]
        
        # ax1.axvspan(-100, -100, color='grey', alpha=0.15, zorder = 0, ec=None,
        #            label = 'Release of the force')
        # ax1.axvspan(-100, -100, color='grey', alpha=0.3, zorder = 0, ec=None,
        #            label = 'Compression and relaxation')
        
        #### shared formatting 
        (ax_xm, ax_xM) = ax.get_xlim()
        
        ax1.set_xlim([0, ax_xM])
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        ax1.legend(handles=LegendHandles, loc='upper right', 
                   handlelength = 1.5, handleheight = 1, 
                   framealpha=1, fontsize=5, labelspacing=0.25)

        ax2.set_xlim([0, ax_xM])
        ax2.set_xlabel('Time (s)', labelpad=1)
        ax2.legend().set_visible(False)
        
        fig.tight_layout()
        axes = [ax1, ax2]
        return(fig, axes)
    
    
    def Pplot_Timeseries_V2bis(self, plotSettings):
        
        apm.setGraphicOptions(mode = 'print', 
                              palette = 'Set2', 
                              colorList = apm.cL_Set21)
        
        fig = plt.figure(figsize=(11/gs.cm_in, 6/gs.cm_in), layout="constrained")
        # fig.tight_layout()
        spec = fig.add_gridspec(6, 1, hspace=0.15, top = 0.975, bottom=0.125, left = 0.065, right = 0.99)
        ax1 = fig.add_subplot(spec[:4])
        ax2 = fig.add_subplot(spec[4:])
        
        Ni = 3
        LI = self.listIndent[:Ni]
        tsDf = self.tsDf[self.tsDf['idxLoop'] <= Ni]
        
        # time_ticks = np.array([5.5, 10.167, 11.667, 13, 19.0] + [19.0*k for k in range(2, Ni+1)])
        # time_ticklabels = np.array([5.5, 10, 11.5, 13, 19.0] + [19.0*k for k in range(2, Ni+1)])
        # x_Vsep = np.array([19.0*k for k in range(1, Ni+1)])
        
        time_ticks = np.array([6.3, 10.167, 11.667, 13, 18.8, 38.2, 57.5])
        time_ticklabels = np.array([6.0, 10, 11.5, 13, 19, 38, 57])
        x_Vsep = np.array([18.8, 38.2, 57.5])
        
        #### Distance Plot
        ax = ax1
        color = 'steelblue' # gs.colorList40[30] # 'skyblue'# 'blue'
        ax1.set_ylabel('Thickness (nm)', color=color, labelpad=1)
        ax.tick_params(axis='y', labelcolor=color)#, labelsize=8)
        ax.scatter(tsDf['T'].values, tsDf['D3'].values-self.DIAMETER, 
                   color = color, edgecolors = None, linewidths = 0,
                   zorder = 5, s = 2, alpha = 0.9)
        for xv in x_Vsep:
            ax.axvline(xv, ls='-', lw=1, color='k', alpha=0.25)

        (ax_ym, ax_yM) = ax.get_ylim()
        ax.set_ylim([min(-0,ax_ym), ax_yM])
        ax.set_ylim([0, 600])
        ax.tick_params(axis='y', labelcolor = color)
        ax.grid(axis='y', zorder=0)
        
        # Force plot
        ax = ax2
        color = 'firebrick'
        ax.set_ylabel('Force (nN)', color=color, labelpad=1)
        ax.plot(tsDf['T'].values, tsDf['F'].values/1e3, color=color, lw=1.5)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        ax.grid(axis='y', zorder=0)
        (ax_ym, ax_yM) = ax.get_ylim()
        # ax.set_ylim([0, 1.05*max(self.tsDf['F'].values/1e3)])
        ax.set_ylim([0, 1.1])
        for xv in x_Vsep:
            ax.axvline(xv, ls='-', lw=1, color='k', alpha=0.25)
        
        #### Shades
        for i in range(1, Ni+1):
            df = tsDf[tsDf['idxLoop'] == (i)]
            # print(df.idxAnalysis)
            t1 = df['T'].values[ufun.findFirst(-i, df.idxAnalysis)+1]
            # t1bis = df['T'].values[ufun.findFirst(-i, df.idxAnalysis)]
            t2 = df['T'].values[ufun.findFirst(i, df.idxAnalysis)]
            t3 = df['T'].values[ufun.findLast(i, df.idxAnalysis)]
            t4 = df['T'].values[ufun.findLast(-i, df.idxAnalysis)]
            # for xv in [t1, t1bis]:
            #     ax.axvline(xv, ls='-', lw=0.5, color='k')
            for ax in [ax1, ax2]:
                ax.axvspan(t1, t2, color='grey', alpha=0.15, zorder = 0, ec=None)
                ax.axvspan(t2, t3, color='grey', alpha=0.3, zorder = 0, ec=None)
        
        LM0 = mpatches.Rectangle((0, 0), 0, 0, facecolor='w', 
                                 edgecolor='k', linewidth=0.2,
                                 label='Constant field')
        LM1 = mpatches.Rectangle((0, 0), 0, 0, color='grey', alpha=0.15, linewidth=0,
                                   label='Force release')
        LM2 = mpatches.Rectangle((0, 0), 0, 0, color='grey', alpha=0.3, linewidth=0,
                                   label='Compression\n & relaxation')
        LegendHandles = [LM0, LM1, LM2]
        
        # ax1.axvspan(-100, -100, color='grey', alpha=0.15, zorder = 0, ec=None,
        #            label = 'Release of the force')
        # ax1.axvspan(-100, -100, color='grey', alpha=0.3, zorder = 0, ec=None,
        #            label = 'Compression and relaxation')
        
        #### shared formatting 
        (ax_xm, ax_xM) = ax.get_xlim()
        
        ax1.plot([43, 45], [300, 300], color='deepskyblue')
        ax1.text(x=42, y=240, s=r'$H_{init}$', color='deepskyblue')
        # ax1.add_patch(plt.Rectangle((42, 225), 3.5, 65, fc="white",
        #                            zorder=2, alpha=0.75))
        ax1.plot([52, 54], [250, 250], color='red')
        ax1.text(x=53, y=190, s=r'$H_{final}$', color='red')
        ax1.add_patch(plt.Rectangle((53, 175), 4.0, 65, fc="white",
                                   zorder=2, alpha=0.75))
        
        ax1.set_xlim([0, ax_xM])
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        ax1.legend(handles=LegendHandles, loc='upper right', 
                   handlelength = 1.25, handleheight = 1, handletextpad=0.4,
                   framealpha=1, fontsize=5.5, labelspacing=0.2)

        ax2.set_xlim([0, ax_xM])
        ax2.set_xlabel('Time (s)', labelpad=1)
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_ticklabels)
        ax2.xaxis.set_tick_params(rotation=50, labelsize=6, pad=0.05)
        ax2.legend().set_visible(False)
        
        fig.get_layout_engine().set(h_pad = 0.015, 
                                    hspace=0, wspace=0)
        
        # fig.tight_layout()
        axes = [ax1, ax2]
        return(fig, axes)
    
    
    def Pplot_Timeseries_V3(self, plotSettings):
        
        apm.setGraphicOptions(mode = 'print', 
                              palette = 'Set2', 
                              colorList = apm.cL_Set21)
        
        LI = self.listIndent
        NI = len(LI)
        Np = 3
        
        fig = plt.figure(figsize=(7/gs.cm_in, 5/gs.cm_in), layout="constrained")
        spec = fig.add_gridspec(6, min(Np, NI), hspace=0.15, 
                                top = 0.975, bottom=0.125, 
                                left = 0.065, right = 0.99)
        axes1, axes2 = [], []
        b1, b2 = [1000, 0], [1.5, 0]
        
        for i in range(1, min(Np+1, NI+1)):
            ax1 = fig.add_subplot(spec[:4, i-1])
            ax2 = fig.add_subplot(spec[4:, i-1])
            axes1.append(ax1)
            axes2.append(ax2)
            
            #### Bounds
            # ii = i + 3
            ii = i + 0
            df = self.tsDf[self.tsDf['idxLoop'] == ii]
            i1 = ufun.findFirst(ii, df.idxAnalysis)
            i2 = ufun.findLast(ii, df.idxAnalysis)
            Ti = df['T'].values[i1:i2]
            Hi = df['D3'].values[i1:i2]-self.DIAMETER
            Fi = df['F'].values[i1:i2]/1e3
            
            i_Hmin = np.argmin(Hi)
            t_Hmin = Ti[i_Hmin]
            i_Fmax = np.argmax(Fi)
            t_Fmax = Ti[i_Fmax]
            Dt = t_Fmax - t_Hmin
            
            #### Distance Plot
            ax = ax1
            color = 'steelblue' # gs.colorList40[30] # 'skyblue'# 'blue'
            # ax.scatter(Ti, Hi, color = color, edgecolors = None, 
            #            linewidths = 0, zorder = 5, s = 2, alpha = 0.9)
            ax.plot(Ti, Hi, color=color, lw=0.75) 
            ax.axvline(t_Hmin, ls='--', dashes=[5, 1], lw=0.5, color = 'steelblue', alpha=0.7)
            ax.axvline(t_Fmax, ls='--', dashes=[5, 1], lw=0.5, color = 'firebrick', alpha=0.7)
            
            ax.grid(axis='y', zorder=0)
            ax.set_title(f'$\\delta T$ = {Dt:.2f} s', fontsize=6, pad=3)
            
            b = ax.get_ylim()
            if b[0]<b1[0]:
                b1[0]=b[0]
            if b[1]>b1[1]:
                b1[1]=b[1]
            
            if i == 1:
                ax.set_ylabel('Thickness (nm)', color=color, labelpad=1)
                ax.tick_params(axis='y', labelcolor = color)
            else:
                ax.set_yticklabels([])
                ax.set_ylabel('')
                
            
            
            #### Force plot
            ax = ax2
            color = 'firebrick'
            
            ax.plot(Ti, Fi, color=color, lw=0.75)
            ax.axvline(t_Hmin, ls='--', dashes=[5, 1], lw=0.5, color = 'steelblue', alpha=0.7)
            ax.axvline(t_Fmax, ls='--', dashes=[5, 1], lw=0.5, color = 'firebrick', alpha=0.7)
            
            ax.grid(axis='y', zorder=0)
            
            b = ax.get_ylim()
            if b[0]<b2[0]:
                b2[0]=b[0]
            if b[1]>b2[1]:
                b2[1]=b[1]
            
            if i == 1:
                ax.set_ylabel('Force (nN)', color=color, labelpad=1)
                ax.tick_params(axis='y', labelcolor=color)
                ax.set_yticks([0, 0.5, 1.0, 1.5])
            else:
                ax.set_yticklabels([])
                ax.set_ylabel('')
                    
            #### shared formatting 
            # (ax_xm, ax_xM) = ax.get_xlim()
            # ax1.set_xlim([0, ax_xM])
            ax1.set_xticks([])
            ax1.set_xticklabels([])
    
            # ax2.set_xlim([0, ax_xM])
            ax2.set_xlabel('Time (s)', labelpad=1)
            ax2.legend().set_visible(False)
            
        # Set common boundaries
        for i in range(1, min(Np+1, NI+1)):
            ax1 = axes1[i-1]
            ax2 = axes2[i-1]
            ax1.set_ylim(b1)
            ax2.set_ylim(b2)
        
        # fig.tight_layout()
        axes = np.array([axes1, axes2])
        return(fig, axes)
    
    
    
    def Pplot_FH500_V3(self, plotSettings):
        
        apm.setGraphicOptions(mode = 'print', 
                              palette = 'Set2', 
                              colorList = apm.cL_Set21)

        fig, ax = plt.subplots(1, 1, figsize = (8/gs.cm_in, 5/gs.cm_in))
        ax.set_xlabel('h (nm)')
        ax.set_ylabel('F (pN)')
        ax.grid(axis='y')
        
        col_sur = 'navy'
        col_precomp = 'royalblue'
        ax.plot([], [], ls='', marker='.', ms=2,
                color=col_sur, zorder=2, label = 'Constant field 5mT')
        ax.plot([], [], ls='', marker='.', ms=2,
                color=col_precomp, zorder=2, label = 'Initial relaxation')

        Np = min(5, len(self.listIndent))
        
        for i in range(Np):
            ax = ax
            IC = self.listIndent[i]
            IC.Pplot_FH500_V3(fig, ax, i, Np, plotSettings)
            
            M_sur = self.getMaskForCompression(i, task = 'surrounding')
            H_sur = self.tsDf.D3.values[M_sur] - self.DIAMETER
            F_sur = self.tsDf.F.values[M_sur]
            ax.plot(H_sur, F_sur, ls='', marker='.', ms=2,
                    color=col_sur, zorder=2)
            
            
            M_precomp = self.getMaskForCompression(i, task = 'precompression')
            H_precomp = self.tsDf.D3.values[M_precomp] - self.DIAMETER
            F_precomp = self.tsDf.F.values[M_precomp]
            ax.plot(H_precomp, F_precomp, ls='', marker='.', ms=2,
                    color=col_precomp, zorder=2)

        
        ax.legend(fontsize=4, handlelength=1)
        fig.tight_layout()
        
        axes = [ax]
        return(fig, axes)
    
    
    
    def Pplot_FH500_V2(self, plotSettings):
        apm.setGraphicOptions(mode = 'print', 
                              palette = 'Set2', 
                              colorList = apm.cL_Set21)
        numIndent = [1]
        nColsSubplot = 1
        nRowsSubplot = len(numIndent)
        fig, ax = plt.subplots(nRowsSubplot, nColsSubplot,
                               figsize = (6/gs.cm_in, 5/gs.cm_in))
        axes=[ax]

        Np = min(5, len(self.listIndent))

        for i, n in enumerate(numIndent):
            ax = axes[i]
            IC = self.listIndent[n]
            IC.Pplot_FH500_V2(fig, ax, plotSettings)
        
        fig.tight_layout()
        return(fig, axes)
    
    
    
    
    def Pplot_FH500(self, plotSettings):
        apm.setGraphicOptions(mode = 'print', 
                              palette = 'Set2', 
                              colorList = apm.cL_Set21)
        nColsSubplot = 1
        nRowsSubplot = 5
        fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
                                 figsize = (8/gs.cm_in, 25/gs.cm_in))
        # figTitle = 'Thickness-Force of indentations\n'
        # if plotH0:
        #     figTitle += 'with H0 detection (' + self.method_bestH0 + ') ; ' 
        # if plotFit:
        #     figTitle += 'with fit (Chadwick)'
        # fig.suptitle(figTitle)
        Np = min(5, len(self.listIndent))
        
        for i in range(Np):
            ax = axes[i]
            IC = self.listIndent[i]
            IC.Pplot_FH500(fig, ax, plotSettings)
        
        fig.tight_layout()
        return(fig, axes)
    





    def plot_Timeseries(self, plotSettings):
        # gs.set_manuscript_options_jv()
        fig, ax = plt.subplots(1,1,
                               # figsize = (5,4))
                               # figsize=(17/gs.cm_in,3))
                                figsize=(np.max(self.tsDf['T'])*(1/8),4))
                               # figsize=(np.max(self.tsDf['T'])*(5/7), 4))
        # fig.suptitle(self.cellID)
        
        # nLoops = 3
        # idx = self.tsDf[self.tsDf['idxLoop'] <= nLoops].index.values
        
        # Distance axis
        color = gs.colorList40[30] # 'skyblue'# 'blue'
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Thickness (nm)', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.scatter(self.tsDf['T'].values, self.tsDf['D3'].values-self.DIAMETER, 
                color = color, ls = '--', linewidth = 1, zorder = 1, s = 4)
        ax.grid(axis='y', which='major')

        
        for ii in range(self.Ncomp):
            IC = self.listIndent[ii]
            
            compValid = IC.isValidForAnalysis
            
            if compValid:
                fitError = IC.dictFitFH_Chadwick['Full']['error']

                if (not fitError):
                    ax.scatter(IC.Df['T'].values[:], IC.Df['D3'].values[:]-self.DIAMETER, s = 4,
                            color = 'chartreuse', linestyle = '-', linewidth = 1.25, zorder = 3)
                    
                    # if not IC.error_bestH0:
                    #     ax.plot(IC.Df['T'].values[0], IC.bestH0, 
                    #             color = '#b29600', marker = '*', markersize = 4, zorder = 3)
                        
                else:
                    ax.scatter(IC.Df['T'].values, IC.Df['D3'].values-self.DIAMETER, s = 4,
                            color = 'crimson', linestyle = '-', linewidth = 1.25, zorder = 3)
                
            else:
                ax.scatter(IC.Df['T'].values, IC.Df['D3'].values-self.DIAMETER, s = 4,
                        color = 'crimson', linestyle = '-', linewidth = 1.25, zorder = 3)
                
            
        
        (axm, axM) = ax.get_ylim()
        ax.set_ylim([min(0,axm), axM])
        if (max(self.tsDf['D3'].values-self.DIAMETER) > 200):
            ax.set_yticks(np.arange(0, max(self.tsDf['D3'].values-self.DIAMETER), 100))
        
        # Force axis
        ax.tick_params(axis='y', labelcolor = color)
        axbis = ax.twinx()
        color = 'firebrick'
        # axbis.set_ylabel('Force (pN)', color=color)
        axbis.set_ylabel('Force (nN)', color=color)
        # axbis.plot(self.tsDf['T'].values[idx], self. tsDf['F'].values[idx], color=color)
        axbis.plot(self.tsDf['T'].values, self. tsDf['F'].values/1e3, color=color, lw=2.5)
        axbis.tick_params(axis='y', labelcolor=color, labelsize = 11)
        axbis.set_yticks([0, 0.5, 1.0, 1.5])
        minh = np.min(self.tsDf['D3'].values - self.DIAMETER)
        ratio = min(1/abs(minh/axM), 5)
        (axmbis, axMbis) = axbis.get_ylim()
        
        try:
            axbis.set_ylim([0, max(axMbis*ratio, 3*max(self.tsDf['F'].values/1e3))])
        except:
            pass
        
        axes = [ax, axbis]
        fig.tight_layout()
        return(fig, axes)
        
    
    
    
    def plot_Timeseries_V0(self, plotSettings):
        #Plots with lines, can't distinguish between real points and small pauses caused my labview b/w loops
        fig, ax = plt.subplots(1,1,
                               # figsize = (5,4))
                               figsize=(np.max(self.tsDf['T'])*(1/7),4))
        fig.suptitle(self.cellID)
        
        # Distance axis
        color = gs.colorList40[30] # 'skyblue'# 'blue'
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Thickness (nm)', color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.plot(self.tsDf['T'].values, self.tsDf['D3'].values-self.DIAMETER, 
                color = color, ls = '--', linewidth = 1, zorder = 1)
        
        for ii in range(self.Ncomp):
            IC = self.listIndent[ii]
            
            compValid = IC.isValidForAnalysis
            
            if compValid:
                fitError = IC.dictFitFH_Chadwick['Full']['error']

                if (not fitError):                
                    ax.plot(IC.Df['T'].values, IC.Df['D3'].values-self.DIAMETER, 
                            color = 'chartreuse', linestyle = '-', linewidth = 1.25, zorder = 3)
                    
                    if not IC.error_bestH0:
                        ax.plot(IC.Df['T'].values[0], IC.bestH0, 
                                color = gs.colorList40[30], marker = 'o', markersize = 2, zorder = 3)
                        
                else:
                    ax.plot(IC.Df['T'].values, IC.Df['D3'].values-self.DIAMETER, 
                            color = 'crimson', linestyle = '-', linewidth = 1.25, zorder = 3)
                
            else:
                ax.plot(IC.Df['T'].values, IC.Df['D3'].values-self.DIAMETER, 
                        color = 'crimson', linestyle = '-', linewidth = 1.25, zorder = 3)
                
            
        
        (axm, axM) = ax.get_ylim()
        ax.set_ylim([min(0,axm), axM])
        if (max(self.tsDf['D3'].values-self.DIAMETER) > 200):
            ax.set_yticks(np.arange(0, max(self.tsDf['D3'].values-self.DIAMETER), 100))
        
        # Force axis
        ax.tick_params(axis='y', labelcolor = color)
        axbis = ax.twinx()
        color = 'firebrick'
        axbis.set_ylabel('Force (pN)', color=color)
        axbis.plot(self.tsDf['T'].values,self. tsDf['F'].values, color=color)
        axbis.tick_params(axis='y', labelcolor=color, labelsize = 8)
        axbis.set_yticks([0,500,1000,1500])
        minh = np.min(self.tsDf['D3'].values-self.DIAMETER)
        ratio = min(1/abs(minh/axM), 5)
        (axmbis, axMbis) = axbis.get_ylim()
        axbis.set_ylim([0, max(axMbis*ratio, 3*max(self.tsDf['F'].values))])
        
        axes = [ax, axbis]
        fig.tight_layout()
        return(fig, axes)
    
    def plot_FH_VWC(self, plotSettings, plotH0 = True, plotFit = True):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
                                 # figsize = (3, 4))
                                figsize = (4*nColsSubplot, 4*nRowsSubplot))
        figTitle = 'Thickness-Force of indentations\n'
        if plotH0:
            figTitle += 'with H0 detection (' + self.method_bestH0 + ') ; ' 
        if plotFit:
            figTitle += 'with fit (Van Wyk-Chadwick)'
        
        fig.suptitle(figTitle)
        
        for i in range(self.Ncomp):
            colSp = (i) % nColsSubplot
            rowSp = (i) // nColsSubplot
            if nRowsSubplot == 1:
                ax = axes[colSp]
            elif nRowsSubplot >= 1:
                ax = axes[rowSp,colSp]
                
            IC = self.listIndent[i]
            if IC.isValidForAnalysis:
                IC.plot_FH_VWC(fig, ax, plotSettings, plotH0 = plotH0, plotFit = plotFit)
            
        fig.tight_layout()
        return(fig, axes)
    
    
    def plot_FH_Dimitriadis(self, plotSettings, plotH0 = True, plotFit = True):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
                                 # figsize = (3, 4))
                                figsize = (4*nColsSubplot, 4*nRowsSubplot))
        figTitle = 'Thickness-Force of indentations\n'
        if plotH0:
            figTitle += 'with H0 detection (' + self.method_bestH0 + ') ; ' 
        if plotFit:
            figTitle += 'with fit (Dimitriadis)'
        
        fig.suptitle(figTitle)
        
        for i in range(self.Ncomp):
            colSp = (i) % nColsSubplot
            rowSp = (i) // nColsSubplot
            if nRowsSubplot == 1:
                ax = axes[colSp]
            elif nRowsSubplot >= 1:
                ax = axes[rowSp,colSp]
                
            IC = self.listIndent[i]
            IC.plot_FH_Dimitriadis(fig, ax, plotSettings, plotH0 = plotH0, plotFit = plotFit)
            
        fig.tight_layout()
        return(fig, axes)
    


    def plot_FH_ChadAndDimi(self, plotSettings, plotH0 = True, plotFit = True):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
                                 # figsize = (3, 4))
                                figsize = (4*nColsSubplot, 4*nRowsSubplot))
        figTitle = 'Thickness-Force of indentations\n'
        if plotH0:
            figTitle += 'with H0 detection (' + self.method_bestH0 + ') ; ' 
        if plotFit:
            figTitle += 'with fit (Chad & Dimi)'
        
        fig.suptitle(figTitle)
        
        for i in range(self.Ncomp):
            colSp = (i) % nColsSubplot
            rowSp = (i) // nColsSubplot
            if nRowsSubplot == 1:
                ax = axes[colSp]
            elif nRowsSubplot >= 1:
                ax = axes[rowSp,colSp]
                
            IC = self.listIndent[i]
            IC.plot_FH_ChadAndDimi(fig, ax, plotSettings, plotH0 = plotH0, plotFit = plotFit)
            
        fig.tight_layout()
        return(fig, axes)
    
    
    def plot_FH(self, plotSettings, plotH0 = True, plotFit = True):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
                                 # figsize = (3, 4))
                                figsize = (4*nColsSubplot, 4*nRowsSubplot))
        figTitle = 'Thickness-Force of indentations\n'
        if plotH0:
            figTitle += 'with H0 detection (' + self.method_bestH0 + ') ; ' 
        if plotFit:
            figTitle += 'with fit (Chadwick)'
        
        fig.suptitle(figTitle)
        
        for i in range(self.Ncomp):
            colSp = (i) % nColsSubplot
            rowSp = (i) // nColsSubplot
            if nRowsSubplot == 1:
                ax = axes[colSp]
            elif nRowsSubplot >= 1:
                ax = axes[rowSp,colSp]
                
            IC = self.listIndent[i]
            IC.plot_FH(fig, ax, plotSettings, plotH0 = plotH0, plotFit = plotFit)
            
        fig.tight_layout()
        return(fig, axes)
    
    
    
    def plot_SS(self, plotSettings, plotFit = True, fitType = 'stressRegion'):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
                               figsize = (3*nColsSubplot, 3*nRowsSubplot))
        
        figTitle = 'Strain-Stress of compressions'
        if plotFit:
            figTitle += '\nfit type: ' + fitType
            if fitType in ['stressRegion', 'stressGaussian']:
                step = plotSettings['plotStressCenters'][1] - plotSettings['plotStressCenters'][0]
                figTitle += ' - {:.0f}_{:.0f}'.format(step, 
                                                      plotSettings['plotStressHW'])
            
            if fitType == 'nPoints':
                figTitle += (' - ' + plotSettings['plotPoints'])
            
            # NEW: Log
            if fitType == 'Log':
                figTitle += (' - ' + plotSettings['plotLog'])
            
            # NEW: Strain
            if fitType == 'strainGaussian':
                step = plotSettings['plotStrainCenters'][1] - plotSettings['plotStrainCenters'][0]
                figTitle += ' - {:.3e}_{:.3e}'.format(step, 
                                                      plotSettings['plotStrainHW'])
                
            # NEW: 3parts
            if fitType == '3parts':
                figTitle += ' - 3 parts'
                
        fig.suptitle(figTitle)
        
        for i in range(self.Ncomp):
            colSp = (i) % nColsSubplot
            rowSp = (i) // nColsSubplot
            if nRowsSubplot == 1:
                ax = axes[colSp]
            elif nRowsSubplot >= 1:
                ax = axes[rowSp,colSp]
                
            IC = self.listIndent[i]
            if IC.isValidForAnalysis:
                IC.plot_SS(fig, ax, plotSettings, plotFit = plotFit, fitType = fitType)
            

        axes = ufun.setCommonBounds_V2(axes, mode = 'firstLine', 
                                       xb = [0, 'auto'], yb = [0, 'auto'],
                                       Xspace = [0, 1], Yspace = [0, 5000])

        # axes = ufun.setCommonBounds(axes, xb = [0, 'auto'], yb = [0, 'auto'])
            
        fig.tight_layout()
        return(fig, axes)
    
    
    def plot_KS(self, plotSettings, fitType = 'stressRegion'):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, axes = plt.subplots(nRowsSubplot, nColsSubplot,
                               figsize = (3*nColsSubplot, 3*nRowsSubplot))
        figTitle = 'Tangeantial modulus'
        figTitle += '\nfit type: ' + fitType
        if fitType in ['stressRegion', 'stressGaussian']:
            step = plotSettings['plotStressCenters'][1] - plotSettings['plotStressCenters'][0]
            figTitle += ' - {:.0f}_{:.0f}'.format(step, 
                                                  plotSettings['plotStressHW'])
        if fitType == 'nPoints':
            figTitle += (' - ' + plotSettings['plotPoints'])
            
        if fitType == 'Log':
            figTitle += (' - ' + plotSettings['plotLog'])
        
        if fitType in ['strainGaussian']:
            step = plotSettings['plotStrainCenters'][1] - plotSettings['plotStrainCenters'][0]
            figTitle += ' - {:.3e}_{:.3e}'.format(step, 
                                                  plotSettings['plotStrainHW'])
            
        fig.suptitle(figTitle)
        
        for i in range(self.Ncomp):
            colSp = (i) % nColsSubplot
            rowSp = (i) // nColsSubplot
            if nRowsSubplot == 1:
                ax = axes[colSp]
            elif nRowsSubplot >= 1:
                ax = axes[rowSp,colSp]
                
            IC = self.listIndent[i]
            if IC.isValidForAnalysis:
                IC.plot_KS(fig, ax, plotSettings, fitType = fitType)
            ax.grid(axis = 'y')
        
        axes = ufun.setCommonBounds_V2(axes, mode = 'allLines', 
                                       xb = [0, 'auto'], yb = [0, 'auto'],
                                       Xspace = [0, 5000], Yspace = [0, 60])
        # axes = ufun.setCommonBounds(axes, xb = [0, 'auto'], yb = [0, 'auto'])
            
        fig.tight_layout()
        return(fig, axes)
    
    
    def plot_and_save(self, plotSettings, dpi = 150, figSubDir = 'MecaAnalysis_allCells'):
        """
        Parameters
        ----------
        plotSettings : TYPE
            DESCRIPTION.
        dpi : TYPE, optional
            DESCRIPTION. The default is 150.
        figSubDir : TYPE, optional
            DESCRIPTION. The default is 'MecaAnalysis_allCells'.

        Returns
        -------
        None.

        """
        # See the definition of plotSettings below !
        suf = plotSettings['subfolder_suffix']
        if len(suf) > 0:
            suf = '_' + suf
            figSubDir = figSubDir + suf
            
        # 0.
        if plotSettings['Plots_Papier']:
            figDir_Papier = 'C:/Users/josep/Desktop/Seafile/PapierDensité/FiguresMain'
            figSubDir_Papier = 'F1'
            
            # ------
            #### hF(t) plots
            # try:
            # name = self.cellID + '_F1C_hF(t)'
            # fig, ax = self.Pplot_Timeseries(plotSettings)
            # ufun.archiveFig(fig, name = name, dpi = 150, ext = '.pdf', 
            #                 figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # ufun.archiveFig(fig, name = name, dpi = 500, ext = '.png', 
            #                 figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # except:
            #     pass
            
            # # try:
            # name = self.cellID + '_F1C_hF(t)_V2'
            # fig, ax = self.Pplot_Timeseries_V2(plotSettings)
            # ufun.archiveFig(fig, name = name, dpi = 150, ext = '.pdf', 
            #                 figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # ufun.archiveFig(fig, name = name, dpi = 500, ext = '.png', 
            #                 figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # # except:
            # #     pass
        
            # try:
            name = 'F1_D_' + self.cellID
            fig, ax = self.Pplot_Timeseries_V2bis(plotSettings)
            ufun.archiveFig(fig, name = name, dpi = 300, ext = '.pdf', 
                            figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            ufun.archiveFig(fig, name = name, dpi = 500, ext = '.png', 
                            figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # except:
            #     pass
            
            # # try:
            # name = self.cellID + '_SF1D_hF(t)_V3'
            # fig, ax = self.Pplot_Timeseries_V3(plotSettings)
            # ufun.archiveFig(fig, name = name, dpi = 150, ext = '.pdf', 
            #                 figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # ufun.archiveFig(fig, name = name, dpi = 500, ext = '.png', 
            #                 figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # # except:
            # #     pass
            
            
            # ------
            #### F(h) plots
            # try:
            # name = self.cellID + '_F1D_F(h)_E500'
            # fig, ax = self.Pplot_FH500(plotSettings)
            # ufun.archiveFig(fig, name = name, dpi = 150, ext = '.pdf', 
            #                 figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # ufun.archiveFig(fig, name = name, dpi = 500, ext = '.png', 
            #                 figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # except:
            #     pass
        
            # try:
            name = 'F1_G_1-1' + self.cellID
            fig, ax = self.Pplot_FH500_V2(plotSettings)
            ufun.archiveFig(fig, name = name, dpi = 300, ext = '.pdf', 
                            figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            ufun.archiveFig(fig, name = name, dpi = 500, ext = '.png', 
                            figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # except:
            #     pass
        
            # # try:
            # name = self.cellID + '_F1D_F(h)_E500_V3'
            # fig, ax = self.Pplot_FH500_V3(plotSettings)
            # ufun.archiveFig(fig, name = name, dpi = 150, ext = '.pdf', 
            #                 figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # ufun.archiveFig(fig, name = name, dpi = 500, ext = '.png', 
            #                 figDir = figDir_Papier, figSubDir = figSubDir_Papier)
            # # except:
            # #     pass
        
        
        
        # 1.
        if plotSettings['FH(t)']:
            # try:
            name = self.cellID + '_01_h(t)'
            fig, ax = self.plot_Timeseries(plotSettings)
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass
        
        if plotSettings['F(H)_Dimitriadis']:
            
            # try:
                
            name = self.cellID + '_Dimitriadis_F(h)'
            fig, ax = self.plot_FH_Dimitriadis(plotSettings)
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass 
        
        if plotSettings['F(H)_ChadAndDimi']:
            
            # try:
                
            name = self.cellID + '_ChadAndDimi_F(h)'
            fig, ax = self.plot_FH_ChadAndDimi(plotSettings)
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass 
        
        if plotSettings['F(H)_VWC']:
            
            # try:
                
            name = self.cellID + '_VWC_F(h)'
            fig, ax = self.plot_FH_VWC(plotSettings, plotH0 = True, plotFit = True)
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass 
        
        # 2.
        if plotSettings['F(H)']:
            # try:
                name = self.cellID + '_02_F(h)'
                fig, ax = self.plot_FH(plotSettings, plotH0 = True, plotFit = True)
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass
            
        # 3.
        if plotSettings['S(e)_stressRegion']:
            try:
                name = self.cellID + '_03-1_S(e)_stressRegion'
                fig, ax = self.plot_SS(plotSettings, plotFit = True, fitType = 'stressRegion')
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            except:
                pass
        if plotSettings['K(S)_stressRegion']:
            try:
                name = self.cellID + '_03-2_K(S)_stressRegion'
                fig, ax = self.plot_KS(plotSettings, fitType = 'stressRegion')
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            except:
                pass
            
        # 4.
        if plotSettings['S(e)_stressGaussian']:
            # try:
            name = self.cellID + '_04-1_S(e)_stressGaussian'
            fig, ax = self.plot_SS(plotSettings, plotFit = True, fitType = 'stressGaussian')
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass
        if plotSettings['K(S)_stressGaussian']:
            # try:
            name = self.cellID + '_04-2_K(S)_stressGaussian'
            fig, ax = self.plot_KS(plotSettings, fitType = 'stressGaussian')
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass
            
        # 5.
        if plotSettings['S(e)_nPoints']:
            # try:
            name = self.cellID + '_05-1_S(e)_nPoints'
            fig, ax = self.plot_SS(plotSettings, plotFit = True, fitType = 'nPoints')
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass
        if plotSettings['K(S)_nPoints']:
            # try:
            name = self.cellID + '_05-2_K(S)_nPoints'
            fig, ax = self.plot_KS(plotSettings, fitType = 'nPoints')
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass
            

        # 6.
        if plotSettings['S(e)_Log']:
            try:
                name = self.cellID + '_06-1_S(e)_Log'
                fig, ax = self.plot_SS(plotSettings, plotFit = True, fitType = 'Log')
                ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            except:
                pass
        if plotSettings['K(S)_Log']:
            # try:
            name = self.cellID + '_06-2_K(S)_Log'
            fig, ax = self.plot_KS(plotSettings, fitType = 'Log')
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass     
            
        
        # 7.
        if plotSettings['S(e)_strainGaussian']:
            # try:
            name = self.cellID + '_07-1_S(e)_strainGaussian'
            fig, axes = self.plot_SS(plotSettings, plotFit = True, fitType = 'strainGaussian')
            
            for i in range(len(axes.flatten())):
                try:
                    ax = axes.flatten()[i]
                    bestH0 = self.listIndent[i].bestH0
                    new_tick_locations = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
                    def tick_function(eps, h0):
                        H = h0 - 3*h0*eps
                        return(["%.1f" % h for h in H])
                    ax2 = ax.twiny()
                    ax2.set_xlim(ax.get_xlim())
                    ax2.set_xticks(new_tick_locations)
                    ax2.set_xticklabels(tick_function(new_tick_locations, bestH0), fontdict={'fontsize': 6})
                    # ax2.set_xlabel(r"h (nm)")
                    ax2.grid()
                except:
                    pass
                
            plt.tight_layout()
            
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass
        
        if plotSettings['K(S)_strainGaussian']:
            # try:
            name = self.cellID + '_07-2_K(S)_strainGaussian'
            fig, axes = self.plot_KS(plotSettings, fitType = 'strainGaussian')
            
            for i in range(len(axes.flatten())):
                try:
                    ax = axes.flatten()[i]
                    bestH0 = self.listIndent[i].bestH0
                    new_tick_locations = np.array([0.05, 0.1, 0.15, 0.2, 0.25])
                    def tick_function(eps, h0):
                        H = h0 - 3*h0*eps
                        return(["%.1f" % h for h in H])
                    ax2 = ax.twiny()
                    ax2.set_xlim(ax.get_xlim())
                    ax2.set_xticks(new_tick_locations)
                    ax2.set_xticklabels(tick_function(new_tick_locations, bestH0), fontdict={'fontsize': 6})
                    # ax2.set_xlabel(r"h (nm)")
                    ax2.grid()
                except:
                    pass
            
            plt.tight_layout()
            
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass
        
        
        #### TEST
        if plotSettings['S(e)_3parts']:
            # try:
            name = self.cellID + '_TEST_S(e)_3parts'
            fig, ax = self.plot_SS(plotSettings, plotFit = True, fitType = '3parts')
            ufun.archiveFig(fig, name = name, figSubDir = figSubDir, dpi = dpi)
            # except:
            #     pass
        
        
        
            
    def exportTimeseriesWithStressStrain(self):
        """
        

        Returns
        -------
        None.

        """
        Nrows = self.tsDf.shape[0]
        ts_H0 = np.full(Nrows, np.nan)
        ts_stress = np.full(Nrows, np.nan)
        ts_strain = np.full(Nrows, np.nan)
        ts_contactRadius = np.full(Nrows, np.nan)
        ts_chadwickRatio = np.full(Nrows, np.nan)
        for IC in self.listIndent:
            iStart = IC.i_tsDf + IC.jStart
            iStop = IC.i_tsDf + IC.jMax+1
            if not IC.error_bestH0:
                ts_H0[iStart:iStop].fill(IC.bestH0)
                ts_stress[iStart:iStop] = IC.stressCompr
                ts_strain[iStart:iStop] = IC.strainCompr
                ts_contactRadius[iStart:iStop] = IC.contactRadius
                ts_chadwickRatio[iStart:iStop] = IC.ChadwickRatio
                
        self.tsDf['H0'] = ts_H0
        self.tsDf['Stress'] = ts_stress
        self.tsDf['Strain'] = ts_strain
        saveName = self.fileName[:-4] + '_stress-strain.csv'
        savePath = os.path.join(cp.DirDataTimeseriesStressStrain, saveName)
        self.tsDf.to_csv(savePath, sep=';', index = False)
        if cp.CloudSaving != '':
            cloudSavePath = os.path.join(cp.DirCloudTimeseriesStressStrain, saveName)
            self.tsDf.to_csv(cloudSavePath, sep=';', index = False)
    
    
    def make_mainResults(self, fitSettings):
        """
        

        Parameters
        ----------
        fitSettings : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # fitSettings = {'doChadwickFit' : True, 'doDimitriadisFit' : False,
        #     'doStressRegionFits' : True, 'doStressGaussianFits' : True, 'doNPointsFits' : True
        #     }
        
        #### Important setting : dictColumnsMeca
        dictColumnsMeca = {'date':'',
                           'cellName':'',
                           'cellID':'',
                           'cellCode':'',
                           'manipID':'',
                           'compNum':np.nan,
                           'compDuration':'',
                           'totalDuration':np.nan,
                           'compStartTime':np.nan,
                           'compAbsStartTime':np.nan,
                           'compStartTimeThisDay':np.nan,
                           'initialThickness':np.nan,
                           'minThickness':np.nan,
                           'maxIndent':np.nan,
                           'Dh_BeforeAfter':np.nan,
                           'Dh_Precomp':np.nan,
                           'Df_Precomp':np.nan,
                           'E_Precomp':np.nan,
                           'previousThickness':np.nan,
                           'surroundingThickness':np.nan,
                           'surroundingDx':np.nan,
                           'surroundingDy':np.nan,
                           'surroundingDz':np.nan,
                           'validatedThickness':False, 
                           'jumpD3':np.nan,
                           'peakDelay':np.nan,
                           'minForce':np.nan, 
                           'maxForce':np.nan, 
                           'ctFieldForce':np.nan,
                           'minStress':np.nan, 
                           'maxStress':np.nan, 
                           'minStrain':np.nan, 
                           'maxStrain':np.nan,
                           'ctFieldThickness':np.nan,
                           'ctFieldFluctuAmpli':np.nan,
                           'ctFieldMinThickness':np.nan,
                           'ctFieldMaxThickness':np.nan,
                           'ctFieldVarThickness':np.nan,
                           'ctFieldDX':np.nan,
                           'ctFieldDY':np.nan,
                           'ctFieldDZ':np.nan,
                           'bestH0':np.nan,
                           'error_bestH0':True,
                           'method_bestH0':'',
                           'dimi_Hmin':np.nan
                           }
        
        if fitSettings['doVWCFit']:
            for m in fitSettings['VWCFitMethods']:
                m2 = 'vwc_' + m
                d = {'error_'+ m2 : True,
                     'nbPts_'+ m2 : np.nan, 
                     'K_'+ m2 : np.nan, 
                     'ciwK_'+ m2 : np.nan, 
                     'Y_'+ m2 : np.nan, 
                     'ciwY_'+ m2 : np.nan, 
                     'H0_'+ m2 : np.nan, 
                     'ciwH0_' + m2 : np.nan,
                     'R2_'+ m2 : np.nan,
                     'Chi2_'+ m2 : np.nan,
                     'valid_'+ m2 : False,
                     'issue_' + m2 : '',
                     }
                dictColumnsMeca = {**dictColumnsMeca, **d}
                
        if fitSettings['doChadwickFit']:
            for m in fitSettings['ChadwickFitMethods']:
                d = {'error_'+ m : True,
                     'nbPts_'+ m : np.nan, 
                     'E_'+ m : np.nan, 
                     'ciwE_'+ m : np.nan, 
                     'H0_'+ m : np.nan, 
                     'R2_'+ m : np.nan,
                     'Chi2_'+ m : np.nan,
                     'valid_'+ m: False,
                     'issue_' + m: '',
                     }
                dictColumnsMeca = {**dictColumnsMeca, **d}
            
        if fitSettings['doDimitriadisFit']:
            for m in fitSettings['DimitriadisFitMethods']:
                m2 = 'Dimi_' + m
                d = {'error_'+ m2 : True,
                     'nbPts_'+ m2 : np.nan, 
                     'E_'+ m2 : np.nan, 
                     'ciwE_'+ m2 : np.nan, 
                     'H0_'+ m2 : np.nan, 
                     'R2_'+ m2 : np.nan,
                     'Chi2_'+ m2 : np.nan,
                     'valid_'+ m2: False,
                     'issue_' + m2: '',
                     }
                dictColumnsMeca = {**dictColumnsMeca, **d}   
        
        
        N = self.Ncomp
        
        results = {}
        for k in dictColumnsMeca.keys():
            results[k] = [dictColumnsMeca[k] for i in range(N)]
        
        # currentCellID = self.cellID
        totalDuration = np.max(self.tsDf['T'].values)
        ctFieldH = (self.tsDf.loc[self.tsDf['idxAnalysis'] == 0, 'D3'].values - self.DIAMETER)
        ctFieldThickness   = np.median(ctFieldH)
        ctFieldMinThickness = np.min(ctFieldH)
        ctFieldMaxThickness = np.max(ctFieldH)
        ctFieldVarThickness = np.var(ctFieldH)
        ctFieldF   = (self.tsDf.loc[self.tsDf['idxAnalysis'] == 0, 'F'].values)
        ctFieldForce   = np.median(ctFieldF)
        ctFieldFluctuAmpli = np.percentile(ctFieldH, 90) - np.percentile(ctFieldH,10)
        ctFieldDX = np.median(self.tsDf.loc[self.tsDf['idxAnalysis'] == 0, 'dx'].values)
        ctFieldDY = np.median(self.tsDf.loc[self.tsDf['idxAnalysis'] == 0, 'dy'].values)
        ctFieldDZ = np.median(self.tsDf.loc[self.tsDf['idxAnalysis'] == 0, 'dz'].values)
            
        for i in range(N):
            
            IC = self.listIndent[i]
            if IC.isValidForAnalysis:
            
                # Identifiers
                results['date'][i] = ufun.findInfosInFileName(self.cellID, 'date')
                results['manipID'][i] = ufun.findInfosInFileName(self.cellID, 'manipID')
                results['cellName'][i] = ufun.findInfosInFileName(self.cellID, 'cellName')
                results['cellID'][i] = self.cellID
                results['cellCode'][i] = '_'.join(self.cellID.split('_')[-2:])
                results['compNum'][i] = i+1
                
                # Time-related
                date_T0 = self.expDf.at[self.expDf.index.values[0], 'date_T0']
                results['compDuration'][i] = self.expDf.at[self.expDf.index.values[0], 'compression duration']
                results['totalDuration'][i] = totalDuration
                results['compStartTime'][i] = IC.rawDf['T'].values[0]
                results['compAbsStartTime'][i] = IC.rawDf['Tabs'].values[0]
                results['compStartTimeThisDay'][i] = IC.rawDf['Tabs'].values[0] - date_T0
                
                
                # Thickness-related ( = D3-DIAMETER)
                previousMask = self.getMaskForCompression(i, task = 'previous')
                surroundingMask = self.getMaskForCompression(i, task = 'surrounding')
                followingMask = self.getMaskForCompression(i, task = 'following')
                precompression = self.getMaskForCompression(i, task = 'precompression')
                H_before = self.tsDf.D3.values[previousMask] - self.DIAMETER
                H_after = self.tsDf.D3.values[followingMask] - self.DIAMETER
                H_surrounding = self.tsDf.D3.values[surroundingMask] - self.DIAMETER
                H_precomp = self.tsDf.D3.values[precompression] - self.DIAMETER
                F_precomp = self.tsDf.F.values[precompression]
                
                previousThickness = np.median(H_before)
                surroundingThickness = np.median(H_surrounding)
                surroundingDx = np.median(self.tsDf.dx.values[surroundingMask])
                surroundingDy = np.median(self.tsDf.dy.values[surroundingMask])
                surroundingDz = np.median(self.tsDf.dz.values[surroundingMask])
                
                # New stuff
                Dh_BeforeAfter = np.median(H_after[:3]) - np.median(H_before[-3:])
                Dh_Precomp = np.percentile(H_precomp, 97) - np.percentile(H_precomp, 3)
                Df_Precomp = np.percentile(F_precomp, 97) - np.percentile(F_precomp, 3)
                H0_500 = IC.dictFitFH_Chadwick['f_<_500']['H0']
                delta1, delta2 = H0_500 - np.percentile(H_precomp, 3), H0_500 - np.percentile(H_precomp, 97)
                E_Precomp = 1e6 * (3*H0_500*Df_Precomp)/(np.pi*0.5*self.DIAMETER*(Dh_Precomp)*(delta1+delta2)) 
                # E ~ 3*H0*F_max / pi*R*(H0-h_min)²
                # print('\n')
                # print(f"H0 = {IC.dictFitFH_Chadwick['f_<_500']['H0']:.1f}")
                # print('H_min | H_p03 | H_p97 | H_max')
                # print(f'{np.min(H_precomp):.1f} | {np.percentile(H_precomp, 3):.1f} | {np.percentile(H_precomp, 97):.1f} | {np.max(H_precomp):.1f}')
                # print('F_min | F_p03 | F_p97 | F_max')
                # print(f'{np.min(F_precomp):.1f} | {np.percentile(F_precomp, 3):.1f} | {np.percentile(F_precomp, 97):.1f} | {np.max(F_precomp):.1f}')
                # print('E_500 | E_Precomp')
                # print(f"{IC.dictFitFH_Chadwick['f_<_500']['E']:.0f} | {E_Precomp:.0f}")
                
                # Attribution
                results['Dh_BeforeAfter'][i] = Dh_BeforeAfter
                results['Dh_Precomp'][i] = Dh_Precomp
                results['Df_Precomp'][i] = Df_Precomp
                results['E_Precomp'][i] = E_Precomp
                results['previousThickness'][i] = previousThickness
                results['surroundingThickness'][i] = surroundingThickness
                results['surroundingDx'][i] = surroundingDx
                results['surroundingDy'][i] = surroundingDy
                results['surroundingDz'][i] = surroundingDz
                results['ctFieldDX'][i] = ctFieldDX
                results['ctFieldDY'][i] = ctFieldDY
                results['ctFieldDZ'][i] = ctFieldDZ
                results['ctFieldThickness'][i] = ctFieldThickness
                results['ctFieldMinThickness'][i] = ctFieldMinThickness
                results['ctFieldMaxThickness'][i] = ctFieldMaxThickness
                results['ctFieldVarThickness'][i] = ctFieldVarThickness
                results['ctFieldFluctuAmpli'][i] = ctFieldFluctuAmpli
                results['jumpD3'][i] = self.listJumpsD3[i]
                results['peakDelay'][i] = IC.peakDelay
                
                results['initialThickness'][i] = np.mean(IC.hCompr[0:3])
                results['minThickness'][i] = np.min(IC.hCompr)
                results['maxIndent'][i] = results['initialThickness'][i] - results['minThickness'][i]
    
                results['validatedThickness'][i] = np.min([results['initialThickness'][i],results['minThickness'][i],
                                              results['previousThickness'][i],results['surroundingThickness'][i],
                                              results['ctFieldThickness'][i]]) > 0
                
                # Force-related
                results['minForce'][i] = np.min(IC.fCompr)
                results['maxForce'][i] = np.max(IC.fCompr)
                results['ctFieldForce'][i] = ctFieldForce
                
                # Best H0 related
                results['bestH0'][i] = IC.bestH0
                results['method_bestH0'][i] = IC.method_bestH0
                results['error_bestH0'][i] = IC.error_bestH0
                
                # Dimitriadis related
                results['dimi_Hmin'][i] = IC.dimi_Hmin
                
                # Strain-stress-related
                results['minStress'][i] = np.min(IC.stressCompr)
                results['maxStress'][i] = np.max(IC.stressCompr)
                results['minStrain'][i] = np.min(IC.strainCompr)
                results['maxStrain'][i] = np.max(IC.strainCompr)
                
                # Whole curve fits related
                
                if fitSettings['doVWCFit'] and IC.isValidForAnalysis:
                    for m in fitSettings['VWCFitMethods']:
                        try:
                            m2 = 'vwc_' + m
                            results['error_'+ m2][i] = IC.dictFitFH_VWC[m]['error']
                            results['nbPts_'+ m2][i] = IC.dictFitFH_VWC[m]['nbPts']
                            results['K_'+ m2][i] = IC.dictFitFH_VWC[m]['K']
                            results['ciwK_'+ m2][i] = IC.dictFitFH_VWC[m]['ciwK']
                            results['Y_'+ m2][i] = IC.dictFitFH_VWC[m]['Y']
                            results['ciwY_'+ m2][i] = IC.dictFitFH_VWC[m]['ciwY']
                            results['H0_'+ m2][i] = IC.dictFitFH_VWC[m]['H0']
                            results['ciwH0_'+ m2][i] = IC.dictFitFH_VWC[m]['ciwH0']
                            results['R2_'+ m2][i] = IC.dictFitFH_VWC[m]['R2']
                            results['Chi2_'+ m2][i] = IC.dictFitFH_VWC[m]['Chi2']
                            results['valid_'+ m2][i] = IC.dictFitFH_VWC[m]['valid']
                            results['issue_'+ m2][i] = IC.dictFitFH_VWC[m]['issue']
                        except:
                            print("VWC Error")
                            # print(IC.dictFitFH_VWC)
                            
                if fitSettings['doChadwickFit'] and IC.isValidForAnalysis:
                    for m in fitSettings['ChadwickFitMethods']:
                        try:
                            results['error_'+ m][i] = IC.dictFitFH_Chadwick[m]['error']
                            results['nbPts_'+ m][i] = IC.dictFitFH_Chadwick[m]['nbPts']
                            results['E_'+ m][i] = IC.dictFitFH_Chadwick[m]['E']
                            results['ciwE_'+ m][i] = IC.dictFitFH_Chadwick[m]['ciwE']
                            results['H0_'+ m][i] = IC.dictFitFH_Chadwick[m]['H0']
                            results['R2_'+ m][i] = IC.dictFitFH_Chadwick[m]['R2']
                            results['Chi2_'+ m][i] = IC.dictFitFH_Chadwick[m]['Chi2']
                            results['valid_'+ m][i] = IC.dictFitFH_Chadwick[m]['valid']
                            results['issue_'+ m][i] = IC.dictFitFH_Chadwick[m]['issue']
                        except:
                            print("Chadwick Error")
                            # print(IC.dictFitFH_Chadwick)
                            
                if fitSettings['doDimitriadisFit'] and IC.isValidForAnalysis:
                    for m in fitSettings['DimitriadisFitMethods']:
                        # try:
                        m2 = 'Dimi_' + m
                        results['error_'+ m2][i] = IC.dictFitFH_Dimitriadis[m]['error']
                        results['nbPts_'+ m2][i] = IC.dictFitFH_Dimitriadis[m]['nbPts']
                        results['E_'+ m2][i] = IC.dictFitFH_Dimitriadis[m]['E']
                        results['ciwE_'+ m2][i] = IC.dictFitFH_Dimitriadis[m]['ciwE']
                        results['H0_'+ m2][i] = IC.dictFitFH_Dimitriadis[m]['H0']
                        results['R2_'+ m2][i] = IC.dictFitFH_Dimitriadis[m]['R2']
                        results['Chi2_'+ m2][i] = IC.dictFitFH_Dimitriadis[m]['Chi2']
                        results['valid_'+ m2][i] = IC.dictFitFH_Dimitriadis[m]['valid']
                        results['issue_'+ m2][i] = IC.dictFitFH_Dimitriadis[m]['issue']
                        # except:
                        #     print("Dimitriadis Error")
                        #     print(IC.dictFitFH_Dimitriadis)
        
        df_mainResults = pd.DataFrame(results)
        self.df_mainResults = df_mainResults
    
    
    
    def make_localFitsResults(self, fitSettings):
        """
        

        Parameters
        ----------
        fitSettings : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # fitSettings = {'doChadwickFit' : True, 'doDimitriadisFit' : False,
        #     'doStressRegionFits' : True, 'doStressGaussianFits' : True, 'doNPointsFits' : True
        #     }

        if fitSettings['doStressRegionFits']:
            df = pd.concat([IC.df_stressRegions for IC in self.listIndent], axis = 0)
            df.reset_index(drop=True, inplace=True)
            self.df_stressRegions = df
            
        if fitSettings['doStressGaussianFits']:
            df = pd.concat([IC.df_stressGaussian for IC in self.listIndent], axis = 0)
            df.reset_index(drop=True, inplace=True)
            self.df_stressGaussian = df
            
        if fitSettings['doNPointsFits']:
            df = pd.concat([IC.df_nPoints for IC in self.listIndent], axis = 0)
            df.reset_index(drop=True, inplace=True)
            self.df_nPoints = df
        
        if fitSettings['doLogFits']:
            df = pd.concat([IC.df_log for IC in self.listIndent], axis = 0)
            df.reset_index(drop=True, inplace=True)
            self.df_log = df
            
        # NEW
        if fitSettings['doStrainGaussianFits']:
            df = pd.concat([IC.df_strainGaussian for IC in self.listIndent], axis = 0)
            df.reset_index(drop=True, inplace=True)
            self.df_strainGaussian = df
            
        #### TEST
        if fitSettings['do3partsFits']:
            df = pd.concat([IC.df_3parts for IC in self.listIndent], axis = 0)
            df.reset_index(drop=True, inplace=True)
            self.df_3parts = df
    
        
    def getH0Df(self):
        L = []
        for IC in self.listIndent:
            if IC.isValidForAnalysis:
                L.append(IC.getH0Df())
        df = pd.concat(L, axis = 0)
        return(df)
    
    
    #### METHODS IN DEVELOPMENT
    
    def plot_KS_smooth(self):
        nColsSubplot = 5
        nRowsSubplot = ((self.Ncomp-1) // nColsSubplot) + 1
        fig, ax = plt.subplots(1,1, figsize = (7,5))
        figTitle = 'Tangential modulus'
        fig.suptitle(figTitle)
        
        listArraysStrain = []
        listArraysStress = []
        listArraysK = []
        for i in range(self.Ncomp):
            IC = self.listIndent[i]
            if IC.computed_SSK_filteredDer:
                listArraysStress.append(IC.SSK_filteredDer[:, 0])
                listArraysStrain.append(IC.SSK_filteredDer[:, 1])
                listArraysK.append(IC.SSK_filteredDer[:, 2])
        
        stress = np.concatenate(listArraysStress)
        strain = np.concatenate(listArraysStrain)
        K = np.concatenate(listArraysK)
        
        mat_ssk = np.array([stress, strain, K]).T

        mat_ssk_stressSorted = mat_ssk[mat_ssk[:, 0].argsort()]

        stress_sorted = mat_ssk_stressSorted[:, 0]
        strain_sorted = mat_ssk_stressSorted[:, 1]
        K_sorted = mat_ssk_stressSorted[:, 2]
        
        it = 1
        frac = 0.2
        # delta = np.max(stress_sorted) * 0.0075
        SK_smoothed = sm.nonparametric.lowess(exog=stress_sorted, endog=K_sorted, 
                                              frac=frac, it=it)

        ax.plot(stress_sorted, K_sorted, c = 'g', ls = '', marker = 'o', markersize = 2)
        ax.plot(SK_smoothed[:, 0], SK_smoothed[:, 1], c = 'r', ls = '-')
            
        fig.tight_layout()
        return(fig, ax)
    
    


class IndentCompression:
    """
    This class deals with all that is done on a single compression.
    """
    
    def __init__(self, CC, indentDf, thisExpDf, i_indent, i_tsDf):
        
        self.rawDf = indentDf
        self.thisExpDf = thisExpDf
        self.i_indent = i_indent
        self.i_tsDf = i_tsDf
        
        self.cellID = CC.cellID
        self.DIAMETER = CC.DIAMETER
        self.EXPTYPE = CC.EXPTYPE
        self.normalField = CC.normalField
        self.minCompField = CC.minCompField
        self.maxCompField = CC.maxCompField
        self.nUplet = CC.nUplet
        
        try:
            self.loopStruct = CC.loopStruct
            self.loop_totalSize = CC.loop_totalSize
            self.loop_rampSize = CC.loop_rampSize
            self.loop_ctSize = CC.loop_ctSize
        except:
            pass
        
        # These fields are to be modified or filled by methods later on
        
        # validateForAnalysis()
        self.isValidForAnalysis = False

        # refineStartStop()
        self.isRefined = False
        self.jMax = np.argmax(self.rawDf.B)
        self.rawT0 = self.rawDf['T'].values[0]
        self.jStart = 0
        self.jStop = len(self.rawDf.D3.values)
        self.hCompr = (self.rawDf.D3.values[:self.jMax+1] - self.DIAMETER)
        self.hRelax = (self.rawDf.D3.values[self.jMax+1:] - self.DIAMETER)
        self.fCompr = (self.rawDf.F.values[:self.jMax+1])
        self.fRelax = (self.rawDf.F.values[self.jMax+1:])
        self.TCompr = (self.rawDf['T'].values[:self.jMax+1])
        self.TRelax = (self.rawDf['T'].values[self.jMax+1:])
        self.BCompr = (self.rawDf.B.values[:self.jMax+1])
        self.BRelax = (self.rawDf.B.values[self.jMax+1:])
        
        # New
        i_minD = np.argmin(self.rawDf.D3.values)
        i_maxF = np.argmax(self.rawDf.F.values)
        self.peakDelay = self.rawDf['T'].values[i_maxF] - self.rawDf['T'].values[i_minD]

        
        self.Df = self.rawDf
        
        # computeH0()
        self.dictH0 = {}
        
        # setBestH0()
        self.bestH0 = np.nan
        self.method_bestH0 = ''
        self.zone_bestH0 = ''
        self.error_bestH0 = True
        
        # find_dimi_range()
        self.DimitriadisRatio = np.zeros_like(self.hCompr)*np.nan
        self.dimi_Hmin = np.nan
        
        # computeStressStrain()
        self.deltaCompr = np.zeros_like(self.hCompr)*np.nan
        self.stressCompr = np.zeros_like(self.hCompr)*np.nan
        self.strainCompr = np.zeros_like(self.hCompr)*np.nan
        self.contactRadius = np.zeros_like(self.hCompr)*np.nan
        self.ChadwickRatio = np.zeros_like(self.hCompr)*np.nan
        
        # fitFH_Chadwick() & fitFH_Dimitriadis()
        self.dictFitFH_Chadwick = {}
        self.dictFitFH_Dimitriadis = {}
        self.dictFitFH_VWC = {}
        
        # Test of new chad fit
        # self.dictFitFH_Chadwick_fixedH0 = {}
        
        # fitSS_stressRegion() & fitSS_stressGaussian() & fitSS_nPoints()
        self.dictFitsSS_stressRegions = {}
        self.dictFitsSS_stressGaussian = {}
        self.dictFitsSS_nPoints = {}
        self.dictFitsSS_log = {}
        
        # dictFits_To_DataFrame()
        self.df_stressRegions = pd.DataFrame({})
        self.df_stressGaussian = pd.DataFrame({})
        self.df_nPoints = pd.DataFrame({})
        self.df_log = pd.DataFrame({})
        
        # NEW ! with strain
        self.dictFitsSS_strainGaussian = {} # fitSS_strainGaussian()
        self.df_strainGaussian = pd.DataFrame({}) # dictFits_To_DataFrame()
        
        #### TEST
        self.dictFitsSS_3parts = {} 
        self.df_3parts = pd.DataFrame({})
        self.computed_SSK_filteredDer = False

    
        
    def validateForAnalysis(self):
        """
        

        Returns
        -------
        None.

        """
       
        listB = self.rawDf.B.values
        
        # Test to check if most of the compression have not been deleted due to bad image quality 
        highBvalues = (listB > (self.maxCompField +self.minCompField)/2)
        N_highBvalues = np.sum(highBvalues)
        testHighVal = (N_highBvalues > 16)
        

        # Test to check if the range of B field is large enough
        minB, maxB = min(listB), max(listB)
        testRangeB = ((maxB-minB) > 0.7*(self.maxCompField - self.minCompField))
        thresholdB = (maxB-minB)/50
        thresholdDeltaB = (maxB-minB)/400

        # Is the curve ok to analyse ?
        isValidForAnalysis = testHighVal and testRangeB # Some criteria can be added here
        self.isValidForAnalysis = isValidForAnalysis
        
        return(isValidForAnalysis)
    
    
    
    def refineStartStop(self):
        """
        

        Returns
        -------
        None.

        """
        listB = self.rawDf.B.values
        NbPtsRaw = len(listB)
        
        # Correct for bugs in the B data
        for k in range(1,len(listB)):
            B = listB[k]
            if B > 1.25*self.maxCompField:
                listB[k] = listB[k-1]

        offsetStart, offsetStop = 0, 0
        minB, maxB = min(listB), max(listB)
        thresholdB = 0.5
        # thresholdDeltaB = (maxB-minB)/400 # NEW CONDITION for the beginning of the compression : 
        # remove the first points where the steps in B are very very small
        
        k = 0
        while (listB[k] < minB+thresholdB):
            offsetStart += 1
            k += 1
        
        # k = 0
        # while (listB[k] < minB+thresholdB) or (listB[k+1]-listB[k] < thresholdDeltaB):
        #     offsetStart += int((listB[k] < minB+thresholdB) or ((listB[k+1]-listB[k]) < thresholdDeltaB))
        #     k += 1

        # k = 0
        # while (listB[-1-k] < minB+thresholdB):
        #     offsetStop += int(listB[-1-k] < minB+thresholdB)
        #     k += 1

        jMax = np.argmax(self.rawDf.B) # End of compression, beginning of relaxation

        hCompr_raw = (self.rawDf.D3.values[:jMax+1] - self.DIAMETER)

        # Refinement of the compression delimitation.
        # Remove the 1-2 points at the begining where there is just the viscous relaxation of the cortex
        # because of the initial decrease of B and the cortex thickness increases.
        
        NptsCompr = len(hCompr_raw)
        k = offsetStart
        while (k<(NptsCompr//2)) and (hCompr_raw[k] < np.max(hCompr_raw[k+1:min(k+10, NptsCompr)])):
            k += 1
        offsetStart = k
        
        jStart = offsetStart # Beginning of compression
        jStop = self.rawDf.shape[0] - offsetStop # End of relaxation
        
        # Better compressions arrays
        self.hCompr = (self.rawDf.D3.values[jStart:jMax+1] - self.DIAMETER)
        self.hRelax = (self.rawDf.D3.values[jMax+1:jStop] - self.DIAMETER)
        self.fCompr = (self.rawDf.F.values[jStart:jMax+1])
        self.fRelax = (self.rawDf.F.values[jMax+1:jStop])
        self.BCompr = (self.rawDf.B.values[jStart:jMax+1])
        self.BRelax = (self.rawDf.B.values[jMax+1:jStop])
        self.TCompr = (self.rawDf['T'].values[jStart:jMax+1])
        self.TRelax = (self.rawDf['T'].values[jMax+1:jStop])
        
        self.jMax = jMax
        self.jStart = jStart
        self.jStop = jStop

        mask = np.array([((j >= jStart) and (j < jStop)) for j in range(NbPtsRaw)])
        
        self.Df = self.rawDf.loc[mask]
        self.isRefined = True
        
        
    def computeH0(self, method = 'Dimitriadis', zone = '%f_20'):
        """
        Alias for self.computeH0_main().
        See below.
        
        Outcome
        -------
        Calls self.computeH0_main(method, zone)
        """
        self.computeH0_main(method, zone)
        
        
    def computeH0_main(self, method, zone):
        """
        Compute the H0 for one or several methods and zones.
        Calls self.computeH0_sub(method, zone) for the computation itself.

        Parameters
        ----------
        method : string or list of strings
            Must be among:
            'Chadwick', 'Dimitriadis' or 'NaiveMax'.
        zone : string or list of strings
            Must be among:
                - Number of points-based: 'pts_15', 'pts_30', 
                - Percents of Force-based: '%f_10', '%f_20', '%f_30', '%f_40'
                - Percents of Thickness-based: '%h_10', '%h_20', '%h_30', '%h_40'

        Outcome
        -------
        Calls self.computeH0_sub(method, zone)

        """
        
        listAllMethods = ['Chadwick', 'Dimitriadis', 'NaiveMax', 'VWC']
        listAllZones = ['pts_15', 'pts_30', 
                        '%f_10', '%f_20', '%f_30', '%f_40', 
                        '%h_10', '%h_20', '%h_30', '%h_40']
        

        if method == 'all':
            method = listAllMethods
            
        if zone == 'all':
            zone = listAllZones
        
        method, zone = ufun.toList(method), ufun.toList(zone)
        
        for m in method:
            for z in zone:
                self.computeH0_sub(m, z)


                    
                
    def computeH0_sub(self, method, zone, returnDict = False):
        """
        Compute the H0 for ONE method and ONE zone.
        Calls fitChadwick_hf() or fitDimitriadis_hf() according to the method required.

        Parameters
        ----------
        method : string
            Must be among:
            'Chadwick', 'Dimitriadis' or 'NaiveMax'.
        zone : string
            Must be among:
                - Number of points-based: 'pts_15', 'pts_30', 
                - Percents of Force-based: '%f_10', '%f_20', '%f_30', '%f_40'
                - Percents of Thickness-based: '%h_10', '%h_20', '%h_30', '%h_40'

        Outcome
        -------
        Add new data in self.dictH0

        """
        
        [zoneType, zoneVal] = zone.split('_')
        
        if zoneType == 'pts':
            i1 = 0
            i2 = int(zoneVal)
            mask = np.array([((i >= i1) and (i < i2)) for i in range(len(self.fCompr))])    
        elif zoneType == '%f':
            pseudoForce = self.fCompr - np.min(self.fCompr)
            thresh = (int(zoneVal)/100.) * np.max(pseudoForce)
            mask = (pseudoForce < thresh)
        elif zoneType == '%Fs15':
            pseudoForce = self.fCompr - np.min(self.fCompr)
            thresh1 = (15./100.) * np.max(pseudoForce)
            thresh2 = (int(zoneVal)/100.) * np.max(pseudoForce)
            mask = (pseudoForce > thresh1) & (pseudoForce < thresh2)
        elif zoneType == '%h':
            pseudoDelta = np.max(self.hCompr) - self.hCompr
            thresh = (int(zoneVal)/100.) * np.max(pseudoDelta)
            mask = (pseudoDelta < thresh)
        elif zoneType == 'ratio':
            z1, z2 = zoneVal.split('-')
            thresh1 = float(z1)
            thresh2 = float(z2)
            mask = (self.ChadwickRatio > thresh1) & (self.ChadwickRatio < thresh2)
        elif zoneType == 'lesser':
            pseudoForce = self.fCompr - np.min(self.fCompr)
            thresh1 = np.min(pseudoForce)
            thresh2 = float(zoneVal)
            mask = (pseudoForce > thresh1) & (pseudoForce < thresh2)
        else:
            mask = np.ones_like(self.hCompr, dtype = bool)
            
        dH0 = {}
        
        if method == 'VWC':
            h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
            params, covM, error = fitVWC_hf(h, f, D)
            K, Y, H0 = params[0], params[1], params[2]
            dH0['H0_' + method + '_' + zone] = H0
            dH0['K_' + method + '_' + zone] = K
            dH0['Y_' + method + '_' + zone] = Y
            dH0['error_' + method + '_' + zone] = error
            dH0['nbPts_' + method + '_' + zone] = np.sum(mask)
            dH0['fArray_' + method + '_' + zone] = VWC(h, K, Y, H0)
            dH0['hArray_' + method + '_' + zone] = h
        
        if method == 'Chadwick':
            h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
            params, covM, error = fitChadwick_hf(h, f, D)
            H0, E = params[1], params[0]
            dH0['H0_' + method + '_' + zone] = H0
            dH0['E_' + method + '_' + zone] = E
            dH0['error_' + method + '_' + zone] = error
            dH0['nbPts_' + method + '_' + zone] = np.sum(mask)
            dH0['fArray_' + method + '_' + zone] = f
            dH0['hArray_' + method + '_' + zone] = inversedChadwickModel(f, E, H0/1000, D/1000)*1000
            
        elif method == 'Dimitriadis':
            h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
            params, covM, error = fitDimitriadis_hf(h, f, D)
            H0, E = params[1], params[0]
            dH0['H0_' + method + '_' + zone] = H0
            dH0['E_' + method + '_' + zone] = E
            dH0['error_' + method + '_' + zone] = error
            dH0['nbPts_' + method + '_' + zone] = np.sum(mask)
            dH0['fArray_' + method + '_' + zone] = dimitriadisModel(h/1000, E, H0/1000, D/1000)
            dH0['hArray_' + method + '_' + zone] = h
            
        elif method == 'NaiveMax': # This method ignore masks
            H0 = np.max(self.hCompr[:])
            dH0['H0_' + method] = H0
            dH0['E_' + method] = np.nan
            dH0['error_' + method] = False
            dH0['fArray_' + method] = np.array([])
            dH0['hArray_' + method] = np.array([])
        
        if not returnDict:
            self.dictH0.update(dH0)
        
        else:
            return(dH0)
        
        
            
    def setBestH0(self, method, zone):
        """
        Assign a value for H0 as the bestH0.
        The value must have been computed with self.computeH0() (and therefore stored in self.dictH0).
        The value must be described by its method & zone.

        Parameters
        ----------
        method : string
            Must be among:
            'Chadwick', 'Dimitriadis' or 'NaiveMax'.
        zone : string
            Must be among:
                - Number of points-based: 'pts_15', 'pts_30', 
                - Percents of Force-based: '%f_10', '%f_20', '%f_30', '%f_40'
                - Percents of Thickness-based: '%h_10', '%h_20', '%h_30', '%h_40'

        Outcome
        -------
        Assign the parameters: 
            - self.bestH0
            - self.error_bestH0
            - self.method_bestH0
            - self.zone_bestH0

        """
        
        self.bestH0 = self.dictH0['H0_' + method + '_' + zone]
        self.error_bestH0 = self.dictH0['error_' + method + '_' + zone]
        self.method_bestH0 = method
        self.zone_bestH0 = zone
        
        
    def getH0Df(self):
        """
        

        Returns
        -------
        None.

        """
        
        d = {'cellID':[],
             'compNum':[],
             'method':[],
             'zone':[],
             'H0':[],
             'nbPts':[],
             'error':[],
             }
        
        for k in self.dictH0.keys():
            if k.startswith('H0'):
                infos = k.split('_') # k = method_zoneType_zoneVal
                if infos[1] in ['Chadwick', 'Dimitriadis', 'VWC']:
                    cellID  = self.cellID
                    compNum = self.i_indent + 1
                    method = infos[1]
                    zone = infos[2] + '_' + infos[3]
                    H0 = self.dictH0[k]
                    nbPts = int(self.dictH0['nbPts_' + method + '_' + zone])
                    error = bool(self.dictH0['error_' + method + '_' + zone])
                    d['cellID'].append(cellID)
                    d['compNum'].append(compNum)
                    d['method'].append(method)
                    d['zone'].append(zone)
                    d['H0'].append(H0)
                    d['nbPts'].append(nbPts)
                    d['error'].append(error)
                    
        for k in d.keys():
            d[k] = np.array(d[k])
                    
        df = pd.DataFrame(d)
        return(df)
    
    
    
        
    
    
    
    def computeStressStrain(self, method = 'Chadwick', H0 = 'best'):
        """
        

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'Chadwick'.

        Returns
        -------
        None.

        """
        
        if H0 == 'best':
            H0 = self.bestH0
            error = self.error_bestH0
            
        else:
            # H0 is the number given in argument
            error = False
            
        if not error:
            deltaCompr = (H0 - self.hCompr)
            
            if method == 'Chadwick':
                # pN and µm
                stressCompr = self.fCompr / (np.pi * (self.DIAMETER/2000) * deltaCompr/1000)
                strainCompr = (deltaCompr/1000) / (3*(H0/1000))
                
            self.deltaCompr = deltaCompr
            self.stressCompr = stressCompr
            self.strainCompr = strainCompr
            
        
    def computeContactRadius(self, method = 'Chadwick', H0 = 'best'):
        """
        

        Parameters
        ----------
        method : TYPE, optional
            DESCRIPTION. The default is 'Chadwick'.

        Returns
        -------
        None.

        """
        if H0 == 'best':
            H0 = self.bestH0
            error = self.error_bestH0
            
        else:
            # H0 is the number given in argument
            error = False
            
        
        if not error:
            hCompr = self.hCompr / 1000 # µm
            deltaCompr = (H0 - self.hCompr) / 1000 # µm
            R = self.DIAMETER / 2000 # µm
            if method == 'Chadwick':
                S = R*deltaCompr
            elif method == 'Dimitriadis':
                S = R*deltaCompr/2
            
            S[S < 0] = 0
            
            self.contactRadius = np.sqrt(S)
            self.ChadwickRatio = self.contactRadius / (H0/1000)
            
            
        
    def find_dimi_range(self):
        H0 = self.bestH0
        error = self.error_bestH0
        
        valid = False
        dimi_Hmin = H0
        mask = np.zeros_like(self.hCompr).astype(bool)
        
        if not error:
            hCompr = self.hCompr / 1000 # µm
            deltaCompr = (H0 - self.hCompr) / 1000 # µm
            R = self.DIAMETER / 2000 # µm
            S = R*deltaCompr/2
            S[S < 0] = 0
            contactRadius = np.sqrt(S)
            DimitriadisRatio = contactRadius / (H0/1000)
            self.DimitriadisRatio = DimitriadisRatio
            mask = (DimitriadisRatio < 0.75)
            
            if np.sum(mask) >= 5:
                valid = True
                dimi_Hmin = np.min(self.hCompr[mask])
                self.dimi_Hmin = dimi_Hmin
                
        return(valid, mask, dimi_Hmin)
        
            
            
    def convergeToH0(self, method, zone, max_it = 10, stop_crit = 0.01):
        #### Finish and TEST IT !!!!!
        listDicts = []
        d0 = self.computeH0_sub(method, zone, returnDict = True)
        listDicts.append(d0)
        H0_current = self.bestH0
        H0_new = d0['H0']
        gap = np.abs((H0_current-H0_new)/H0_current)
        it = 1
        while it < max_it and gap > stop_crit:
            self.computeStressStrain(method='Chadwick', H0 = H0_new)
            self.computeContactRadius(method='Chadwick', H0 = H0_new)
            di = self.computeH0_sub(method, zone, returnDict = True)
            listDicts.append(di)
            H0_current = H0_new
            H0_new = di['H0']
            gap = np.abs((H0_current-H0_new)/H0_current)
            it = it + 1
        print(listDicts)
        return(listDicts[-1])
    
    def fitFH_VWC(self, fitValidationSettings, method = 'Full', mask = []):
        """
        

        Parameters
        ----------
        fitValidationSettings : TYPE
            DESCRIPTION.
        mask : TYPE, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        None.

        """
        
        if len(mask) == 0:
            mask = np.ones_like(self.hCompr, dtype = bool)
        h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
        params, ses, error = fitVWC_hf(h, f, D)
        
        K, Y, H0 = params
        # x = np.linspace(np.min(h), np.max(h), len(h))
        
        #### Test here
        [h_sort, f_sort] = ufun.sortMatrixByCol(np.array([h, f]).T, col=0, direction = -1).T
       
        fPredict = VWC(h_sort, K/1e6, Y/1e6, H0)
        kPredict = VWC(h_sort, K/1e6, 0, H0)
        ePredict = VWC(h_sort, 0, Y/1e6, H0)
        y, yPredict = f_sort, fPredict
        #### err_Chi2 for distance (nm)
        err_chi2 = err_chi2_H
        dictFit = makeDictFit_CVW_hf(params, ses, error, 
                                 h_sort, y, yPredict, kPredict, ePredict, # h_sort instead of h
                                 err_chi2, fitValidationSettings)
        
        self.dictFitFH_VWC[method] = dictFit
        
    
    def fitFH_Chadwick(self, fitValidationSettings, method = 'Full', mask = []):
        """
        

        Parameters
        ----------
        fitValidationSettings : TYPE
            DESCRIPTION.
        mask : TYPE, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        None.
 
        """
        if len(mask) == 0:
            mask = np.ones_like(self.hCompr, dtype = bool)
        h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
        params, ses, error = fitChadwick_hf(h, f, D)
        
        E, H0 = params
        hPredict = inversedChadwickModel(f, E, H0/1000, self.DIAMETER/1000)*1000
        x = f
        y, yPredict = h, hPredict
        #### err_Chi2 for distance (nm)
        err_chi2 = err_chi2_H
        dictFit = makeDictFit_hf(params, ses, error, 
                                 x, y, yPredict, 
                                 err_chi2, fitValidationSettings)
        

        self.dictFitFH_Chadwick[method] = dictFit
        

                
    def fitFH_Dimitriadis(self, fitValidationSettings, method = 'Full', mask = []):
        """
        

        Parameters
        ----------
        fitValidationSettings : TYPE
            DESCRIPTION.
        mask : TYPE, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        None.

        """
        if len(mask) == 0:
            mask = np.ones_like(self.hCompr, dtype = bool)
        h, f, D = self.hCompr[mask], self.fCompr[mask], self.DIAMETER
        params, ses, error = fitDimitriadis_hf(h, f, D)
        E, H0 = params
        fPredict = dimitriadisModel(h/1000, E, H0/1000, self.DIAMETER/1000, v = 0)
        x = h
        y, yPredict = f, fPredict
        #### err_Chi2 for force (pN)
        err_chi2 = err_chi2_F
        dictFit = makeDictFit_hf(params, ses, error, 
                                   x, y, yPredict, 
                                   err_chi2, fitValidationSettings)
        
        self.dictFitFH_Dimitriadis[method] = dictFit
                
    
    def fitSS_stressRegion(self, center, halfWidth, fitValidationSettings):
        """
        

        Parameters
        ----------
        center : TYPE
            DESCRIPTION.
        halfWidth : TYPE
            DESCRIPTION.
        fitValidationSettings : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        id_range = str(center) + '_' + str(halfWidth)
        stress, strain = self.stressCompr, self.strainCompr
        
        lowS, highS = center - halfWidth, center + halfWidth
        mask = ((stress > lowS) & (stress < highS))
        
        params, ses, error = fitLinear_ss(stress, strain, weights = mask)
        
        K, strain0 = params
        strainPredict = inversedConstitutiveRelation(stress[mask], K, strain0)
        
        x = stress[mask]
        y, yPredict = strain[mask], strainPredict
        #### err_Chi2 for strain
        err_chi2 = 0.01
        
        dictFit = makeDictFit_ss(params, ses, error, 
                                 center, halfWidth, x, y, yPredict, 
                                 err_chi2, fitValidationSettings)
        self.dictFitsSS_stressRegions[id_range] = dictFit  
                
    
    def fitSS_stressGaussian(self, center, halfWidth, fitValidationSettings):
        """
        

        Parameters
        ----------
        center : TYPE
            DESCRIPTION.
        halfWidth : TYPE
            DESCRIPTION.
        fitValidationSettings : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        id_range = str(center) + '_' + str(halfWidth)
        stress, strain = self.stressCompr, self.strainCompr
        
        X = stress.flatten(order='C')
        weights = np.exp( -((X - center) ** 2) / halfWidth ** 2)
        
        params, ses, error = fitLinear_ss(stress, strain, weights = weights)
        
        K, strain0 = params
        lowS, highS = center - halfWidth, center + halfWidth
        mask = ((stress > lowS) & (stress < highS))
        strainPredict = inversedConstitutiveRelation(stress[mask], K, strain0)
        
        x = stress[mask]
        y, yPredict = strain[mask], strainPredict
        #### err_Chi2 for strain
        err_chi2 = err_chi2_Strain
        
        dictFit = makeDictFit_ss(params, ses, error, 
                                 center, halfWidth, x, y, yPredict, 
                                 err_chi2, fitValidationSettings)
        self.dictFitsSS_stressGaussian[id_range] = dictFit
    
    
    def fitSS_nPoints(self, mask, fitValidationSettings):
        """
        

        Parameters
        ----------
        mask : TYPE
            DESCRIPTION.
        fitValidationSettings : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        iStart = ufun.findFirst(1, mask)
        iStop = iStart + np.sum(mask)
        id_range = str(iStart) + '_' +str(iStop)
        stress, strain = self.stressCompr, self.strainCompr
        params, ses, error = fitLinear_ss(stress, strain, weights = mask)
        
        K, strain0 = params
        strainPredict = inversedConstitutiveRelation(stress[mask], K, strain0)
        
        x = stress[mask]
        y, yPredict = strain[mask], strainPredict
        #### err_Chi2 for strain
        err_chi2 = err_chi2_Strain
        center = np.median(x)
        halfWidth = (np.max(x) - np.min(x))/2
        
        dictFit = makeDictFit_ss(params, ses, error, 
                                 center, halfWidth, x, y, yPredict, 
                                 err_chi2, fitValidationSettings)
        self.dictFitsSS_nPoints[id_range] = dictFit
        
    
    def fitSS_Log(self, mask, fitValidationSettings):
        """
        

        Parameters
        ----------
        mask : TYPE
            DESCRIPTION.
        fitValidationSettings : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        iStart = ufun.findFirst(1, mask)
        iStop = iStart + np.sum(mask)
        id_range = str(iStart) + '_' +str(iStop)
        stress, strain = self.stressCompr, self.strainCompr
        params, ses, error = fitLinear_ss(stress, strain, weights = mask)
        
        K, strain0 = params
        strainPredict = inversedConstitutiveRelation(stress[mask], K, strain0)
        
        x = stress[mask]
        y, yPredict = strain[mask], strainPredict
        #### err_Chi2 for strain
        err_chi2 = err_chi2_Strain
        center = np.median(x)
        halfWidth = (np.max(x) - np.min(x))/2
        
        dictFit = makeDictFit_ss(params, ses, error, 
                                 center, halfWidth, x, y, yPredict, 
                                 err_chi2, fitValidationSettings)
        self.dictFitsSS_log[id_range] = dictFit
    
    
    def dictFits_To_DataFrame(self, fitSettings):
        """
        

        Parameters
        ----------
        fitSettings : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if fitSettings['doStressRegionFits']:
            df = nestedDict_to_DataFrame(self.dictFitsSS_stressRegions)
            nRows = df.shape[0]
            df.insert(0, 'compNum', np.ones(nRows) * (self.i_indent+1))
            df.insert(0, 'cellID', [self.cellID for i in range(nRows)])
            self.df_stressRegions = df
            
        if fitSettings['doStressGaussianFits']:
            df = nestedDict_to_DataFrame(self.dictFitsSS_stressGaussian)
            nRows = df.shape[0]
            df.insert(0, 'compNum', np.ones(nRows) * (self.i_indent+1))
            df.insert(0, 'cellID', [self.cellID for i in range(nRows)])
            self.df_stressGaussian = df
            
        if fitSettings['doNPointsFits']:
            df = nestedDict_to_DataFrame(self.dictFitsSS_nPoints)
            nRows = df.shape[0]
            df.insert(0, 'compNum', np.ones(nRows) * (self.i_indent+1))
            df.insert(0, 'cellID', [self.cellID for i in range(nRows)])
            self.df_nPoints = df
        
        if fitSettings['doLogFits']:
            df = nestedDict_to_DataFrame(self.dictFitsSS_log)
            nRows = df.shape[0]
            df.insert(0, 'compNum', np.ones(nRows) * (self.i_indent+1))
            df.insert(0, 'cellID', [self.cellID for i in range(nRows)])
            self.df_log = df
            
        # NEW !
        if fitSettings['doStrainGaussianFits']:
            df = nestedDict_to_DataFrame(self.dictFitsSS_strainGaussian)
            nRows = df.shape[0]
            df.insert(0, 'compNum', np.ones(nRows) * (self.i_indent+1))
            df.insert(0, 'cellID', [self.cellID for i in range(nRows)])
            self.df_strainGaussian = df
            
        #### TEST !
        if fitSettings['do3partsFits']:
            df = nestedDict_to_DataFrame(self.dictFitsSS_3parts)
            nRows = df.shape[0]
            df.insert(0, 'compNum', np.ones(nRows) * (self.i_indent+1))
            df.insert(0, 'cellID', [self.cellID for i in range(nRows)])
            self.df_3parts = df
            
            
    def Pplot_FH500_V3(self, fig, ax, i, N, plotSettings):
        cm_in = 2.52
        apm.setGraphicOptions(mode = 'print', 
                              palette = 'Set2', 
                              colorList = apm.cL_Set21)
        
        
        clist = sns.color_palette("husl", N)
        col_comp = clist[i]
        col_relax = apm.lightenColor(clist[i], 1.2)
        
        #### Plot 1
        ax = ax
        if self.isValidForAnalysis:
            ax.plot(self.hCompr, self.fCompr, ls='', marker='.', ms=2, 
                    color=col_comp, zorder=4, label = f'Comp no. {i+1:.0f}')
            ax.plot(self.hRelax[:], self.fRelax[:], ls='', marker='.', ms=2, 
                    color=col_relax, zorder=3)
            # ax.plot([], [], ls='-', color='w', label = ' ', zorder=2)

        #### Style
        for ax in [ax]:
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
            for item in ax.get_xticklabels() + ax.get_yticklabels():
                item.set_fontsize(6)


    def Pplot_FH500_V2(self, fig, ax, plotSettings):
        cm_in = 2.52
        apm.setGraphicOptions(mode = 'print', 
                              palette = 'Set2', 
                              colorList = apm.cL_Set21)
        
        color_base = 'gray'
        color_relax = 'palegreen'
        color_Chad = 'deepskyblue'
        color_Chad400 = 'darkorange'
        # color_H0   = 'mediumseagreen'
        
        #### Plot 1
        ax = ax
        if self.isValidForAnalysis:
            ax.plot(self.hCompr, self.fCompr, ls='', marker='.', ms=4, 
                    color=color_base, label = 'Compression', zorder=3)
            # ax.plot(self.hRelax[:], self.fRelax[:], ls='', marker='.', ms=4, 
            #         color=color_relax, label = 'Relaxation', zorder=2)
            # ax.plot([], [], ls='-', color='w', label = ' ', zorder=2)
            titleText = self.cellID + '_c' + str(self.i_indent + 1)
            legendText = ''
            ax.set_xlabel('h (nm)')
            ax.set_ylabel('F (pN)')
            ax.grid(axis='y')
    
            method = 'Full'
            dictFit = self.dictFitFH_Chadwick[method]
            fitError = dictFit['error']
            
            
                
            if not fitError:
                H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                fFit = dictFit['x']
                hPredict = dictFit['yPredict']
                
                legendText = r'$\bf{Fit\ full\ curve}$'
                legendText += '\n$H_0$ = '     + f'{H0:.0f} nm'
                legendText += '\n$E$ = ' + f'{E/1000:.2f} kPa'
                legendText += '\n$R^2$ = '     + f'{R2:.3f}'
                # ax.plot(hPredict, fFit, ls='-', color = color_Chad, linewidth = 0.75, 
                #         label = legendText, zorder = 5)
                ax.plot(hPredict, fFit, ls='-.', color = color_Chad, 
                        linewidth = 0.75, zorder = 5)
                
            else:
                titleText += '\nFIT ERROR'
                
                
            method = 'f_<_500'
            # dictFit = self.dictFitFH_Chadwick[method]
            dictFit = self.dictFitFH_Chadwick[method]
            fitError = dictFit['error']
                
            if not fitError:
                H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                fFit = dictFit['x']
                hPredict = dictFit['yPredict']
                
                F_max = np.max(fFit)
                F_plot = np.linspace(0, F_max, 100)
                H_plot = inversedChadwickModel(F_plot, E, H0/1000, self.DIAMETER/1000)*1000
                
                legendText = r'$\bf{Fit\ F < 500 pN}$'
                legendText += '\n$H_{500}$ = '     + f'{H0:.0f} nm'
                legendText += '\n$E_{500}$ = ' + f'{E/1000:.2f} kPa'
                legendText += '\n$R^2$ = '     + f'{R2:.3f}'
                # ax.plot(H_plot, F_plot, ls='-', color = color_Chad400, 
                #         linewidth = 1.25, label = legendText, zorder = 6)
                ax.plot(H_plot, F_plot, ls='-', color = apm.lightenColor(color_Chad400, 1.0),
                        linewidth = 1.25, zorder = 6)
    
                legendText += '$H_{500}$ = ' + f'{H0:.0f} nm' #+ '\n' + str_m_z
                ax.plot([H0], [0], ls = '', marker = '+', 
                        color = apm.lightenColor(color_Chad400, 0.8), 
                        markersize = 5, zorder = 8, mew=1.5)
            
            else:
                titleText += '\nFIT ERROR'


            #### Text instead of legend
            x_lim = ax.get_xlim()
            X0, dX = x_lim[0], x_lim[1] - x_lim[0]
            y_lim = ax.get_ylim()
            Y0, dY = y_lim[0], y_lim[1] - y_lim[0]
            fs = 7.5
            
            # rect = plt.Rectangle((X0+0.58*dX, Y0+0.64*dY), 0.38*dX, 0.34*dY,
            #          facecolor="w", alpha=0.9, zorder=7)
            # ax.add_patch(rect)
            
            # ax.text(X0+0.60*dX, Y0+0.9*dY, 'Compression', fontsize = fs,
            #         color = color_base, zorder=8) #, backgroundcolor = 'w')
            # ax.text(X0+0.60*dX, Y0+0.78*dY, 'Fit full curve', fontsize = fs, 
            #         fontstyle = 'italic', color = color_Chad, zorder=8) #, backgroundcolor = 'w')
            # ax.text(X0+0.60*dX, Y0+0.66*dY, 'Fit F < 500 pN', fontsize = fs, 
            #         fontweight = 'bold', color = color_Chad400, zorder=8) #, backgroundcolor = 'w')
            
            rect = plt.Rectangle((X0+0.43*dX, Y0+0.64*dY), 0.55*dX, 0.34*dY,
                     facecolor="w", alpha=0.9, zorder=7)
            ax.add_patch(rect)
            
            ax.text(X0+0.45*dX, Y0+0.9*dY, 'Compression', fontsize = fs,
                    color = color_base, zorder=8) #, backgroundcolor = 'w')
            ax.text(X0+0.45*dX, Y0+0.78*dY, 'Fit full curve', fontsize = fs, 
                    fontstyle = 'italic', color = color_Chad, zorder=8) #, backgroundcolor = 'w')
            ax.text(X0+0.45*dX, Y0+0.66*dY, 'Fit F < 500 pN', fontsize = fs, 
                    fontweight = 'bold', color = color_Chad400, zorder=8) #, backgroundcolor = 'w')
        

        #### Style
        for ax in [ax]:
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
            for item in ax.get_xticklabels() + ax.get_yticklabels():
                item.set_fontsize(6)










    def Pplot_FH500(self, fig, ax, plotSettings):
        cm_in = 2.52
        apm.setGraphicOptions(mode = 'print', 
                              palette = 'Set2', 
                              colorList = apm.cL_Set21)
        
        color_base = 'lightblue'
        color_relax = 'palegreen'
        # color_VW   = 'red'
        color_Chad = 'deepskyblue'
        # color_H0   = 'mediumseagreen'
        # color_total= 'indigo'
        color_Chad400 = 'darkorange'
        
        #### Plot 1
        ax = ax
        if self.isValidForAnalysis:
            ax.plot(self.hCompr, self.fCompr, ls='', marker='.', ms=4, color=color_base, label = 'Compression', zorder=3)
            ax.plot(self.hRelax[:], self.fRelax[:], ls='', marker='.', ms=4, color=color_relax, label = 'Relaxation', zorder=2)
            # ax.plot([], [], ls='-', color='w', label = ' ', zorder=2)
            titleText = self.cellID + '_c' + str(self.i_indent + 1)
            legendText = ''
            ax.set_xlabel('h (nm)')
            ax.set_ylabel('F (pN)')
    
            method = 'Full'
            # dictFit = self.dictFitFH_Chadwick[method]
            dictFit = self.dictFitFH_Chadwick[method]
            fitError = dictFit['error']
                
            if not fitError:
                H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                fFit = dictFit['x']
                hPredict = dictFit['yPredict']
                
                legendText = r'$\bf{Fit\ full\ curve}$'
                legendText += '\n$H_0$ = '     + f'{H0:.0f} nm'
                legendText += '\n$E$ = ' + f'{E/1000:.2f} kPa'
                legendText += '\n$R^2$ = '     + f'{R2:.3f}'
                ax.plot(hPredict, fFit, ls='-', color = color_Chad, linewidth = 1.0, 
                        label = legendText, zorder = 5)
                
            else:
                titleText += '\nFIT ERROR'
                
                
            method = 'f_<_500'
            # dictFit = self.dictFitFH_Chadwick[method]
            dictFit = self.dictFitFH_Chadwick[method]
            fitError = dictFit['error']
                
            if not fitError:
                H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                fFit = dictFit['x']
                hPredict = dictFit['yPredict']
                
                F_max = np.max(fFit)
                F_plot = np.linspace(0, F_max, 100)
                H_plot = inversedChadwickModel(F_plot, E, H0/1000, self.DIAMETER/1000)*1000
                
                legendText = r'$\bf{Fit\ F < 500 pN}$'
                legendText += '\n$H_{500}$ = '     + f'{H0:.0f} nm'
                legendText += '\n$E_{500}$ = ' + f'{E/1000:.2f} kPa'
                legendText += '\n$R^2$ = '     + f'{R2:.3f}'
                # ax.plot(hPredict, fFit, ls='--', color = color_Chad400, 
                #         linewidth = 1.5, label = legendText, zorder = 6)
                ax.plot(H_plot, F_plot, ls='-', color = color_Chad400, 
                        linewidth = 1.0, label = legendText, zorder = 6)
    
                legendText += '$H_{500}$ = ' + f'{H0:.0f} nm' #+ '\n' + str_m_z
                ax.plot([H0], [0], ls = '', marker = '+', color = apm.lightenColor(color_Chad400, 0.8), 
                        markersize = 5, zorder = 8, mew=1.5)
            
            else:
                titleText += '\nFIT ERROR'
                
                    
            # bestH0 = self.bestH0
            # method = self.method_bestH0
            # zone = self.zone_bestH0
            # str_m_z = method + '_' + zone
            # E_bestH0 = self.dictH0['E_' + method + '_' + zone]
            
            # if (not self.error_bestH0) and (method not in ['NaiveMax']):
            #     max_h = np.max(self.hCompr)
            #     high_h = np.linspace(max_h, bestH0, 20)
            #     if self.method_bestH0 == 'Dimitriadis':
            #         low_f = dimitriadisModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
            #     elif self.method_bestH0 == 'Chadwick':
            #         # chadwickModel(h, E, H0, DIAMETER)
            #         low_f = chadwickModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
            #     else:
            #         low_f = np.ones_like(high_h) * bestH0
                
            #     legendText = r'$\bf{Fit\ F < 15\%\ of\ max}$'
            #     legendText += '\nPrecise $H_0$ = ' + f'{bestH0:.0f} nm' #+ '\n' + str_m_z
            #     plot_startH = np.concatenate((self.dictH0['hArray_' + str_m_z][::-1], high_h))
            #     plot_startF = np.concatenate((self.dictH0['fArray_' + str_m_z][::-1], low_f))

            #     ax.plot([bestH0], [0], ls = '', marker = 'o', color = color_H0, markersize = 5, 
            #             label = legendText, zorder = 4)
            #     ax.plot(plot_startH, plot_startF, ls = '--', color = color_H0, linewidth = 1.5, zorder = 2)

            # ax = ufun.setAllTextFontSize(ax, size = 9)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1.01), 
                      handlelength=1, fontsize = 6)
            # ax.title.set_text(titleText)
            ax.grid(axis='y')
            
        #### Style
        for ax in [ax]:
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
            for item in ax.get_xticklabels() + ax.get_yticklabels():
                item.set_fontsize(6)


    
    
    def plot_FH_VWC(self, fig, ax, plotSettings, plotH0 = True, plotFit = True):
        """
        Parameters
        ----------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        plotSettings : TYPE
            DESCRIPTION.
        plotH0 : TYPE, optional
            DESCRIPTION. The default is True.
        plotFit : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        
        if self.isValidForAnalysis and plotSettings['F(H)_VWC']:
            ax.plot(self.hCompr, self.fCompr,'b-', linewidth = 1.5)
            # ax.plot(self.hRelax, self.fRelax,'r-', linewidth = 0.8)
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            legendText = ''
            ax.set_xlabel('h (nm)')
            ax.set_ylabel('f (pN)')
    
            if plotFit:
                
                method = 'Full'
                dictFit = self.dictFitFH_VWC[method]
                fitError = dictFit['error']
                    
                if not fitError:
                    K, Y, H0 = dictFit['K'], dictFit['Y'], dictFit['H0']
                    R2, Chi2 =  dictFit['R2'], dictFit['Chi2']
                    hFit = dictFit['x']
                    fPredict = dictFit['yPredict']
                    kPredict = dictFit['kPredict']
                    ePredict = dictFit['ePredict']
                    Eeff = Y + K * (0.8**-4)
                    
                    legendTextE = 'VWC Fit ;H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, Eeff, R2, Chi2)
                    legendTextK = 'Van Wyk ;K = {:.2e}Pa'.format(K)
                    legendTextY = 'Chadwick ;Y = {:.2e}Pa'.format(Y)
                    
                    
                    ax.plot(hFit, (fPredict),'k--', linewidth = 1.5, label = legendTextE, zorder = 2)
                    ax.plot(hFit, (kPredict),'--', linewidth =1.5, label = legendTextK, zorder = 2, color = '#75305b') # '#75305b') #'#940000')
                    ax.plot(hFit, (ePredict),'--', linewidth = 1.5, label = legendTextY, zorder = 2, color = '#62a07c') #'#62a07c') #'#237e76')
                    
                # else:
                #     titleText += '\nFIT ERROR'
                    
                # method = 'f_<_400'
                # # dictFit = self.dictFitFH_Chadwick[method]
                # dictFit = self.dictFitFH_Chadwick[method]
                # fitError = dictFit['error']
                    
                # if not fitError:
                #     H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                #     fFit = dictFit['x']
                #     hPredict = dictFit['yPredict']
                    
                #     legendText = 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                #     ax.plot(hPredict, fFit,'g--', linewidth = 0.8, 
                #             label = legendText, zorder = 2)
                # # else:
                # #     titleText += '\nFIT ERROR'
                
                # method = 'f_in_400_800'
                # # dictFit = self.dictFitFH_Chadwick[method]
                # dictFit = self.dictFitFH_Chadwick[method]
                # fitError = dictFit['error']
                    
                # if not fitError:
                #     H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                #     fFit = dictFit['x']
                #     hPredict = dictFit['yPredict']
                    
                #     legendText = 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                #     ax.plot(hPredict, fFit, ls='--', color = 'darkorange', linewidth = 0.8, 
                #             label = legendText, zorder = 2)
                # else:
                #     titleText += '\nFIT ERROR'
                    
            if plotH0 and self.method_bestH0 == 'VWC':
                bestH0 = self.bestH0
                method = self.method_bestH0
                zone = self.zone_bestH0
                str_m_z = method + '_' + zone
                K_bestH0 = self.dictH0['K_' + method + '_' + zone]
                Y_bestH0 = self.dictH0['Y_' + method + '_' + zone]
                
                if (not self.error_bestH0) and (method not in ['NaiveMax']):
                    max_h = np.max(self.hCompr)
                    high_h = np.linspace(max_h, bestH0, 20)
                    # if self.method_bestH0 == 'Dimitriadis':
                    #     low_f = dimitriadisModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
                    # elif self.method_bestH0 == 'Chadwick':
                    #     # chadwickModel(h, E, H0, DIAMETER)
                    #     low_f = chadwickModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
                    if self.method_bestH0 == 'VWC':
                        low_f = VWC(high_h/1000, K_bestH0, Y_bestH0, bestH0/1000)
                    else:
                        low_f = np.ones_like(high_h) * bestH0
                    
                    legendText = 'bestH0 = {:.2f}nm'.format(bestH0) + '\n' + str_m_z
                    # plot_startH = np.concatenate((self.dictH0['hArray_' + str_m_z][::-1], high_h))
                    # plot_startF = np.concatenate((self.dictH0['fArray_' + str_m_z][::-1], low_f))

                    ax.plot([bestH0], [0], ls = '', marker = '*', color = '#b29600', markersize = 5, 
                            label = legendText)
                    # ax.plot(plot_startH, plot_startF, ls = '--', color = 'skyblue', linewidth = 1.2, zorder = 4)

                    
                # if 'H0_Chadwick_' + 'ratio_2-2.5' in self.dictH0.keys():
                #     H0_ratio = self.dictH0['H0_Chadwick_ratio_2-2.5']
                #     E_ratio = self.dictH0['E_Chadwick_ratio_2-2.5']
                #     str_m_z = 'Chadwick_ratio_2-2.5'
                #     max_h = np.max(self.hCompr)
                #     high_h = np.linspace(max_h, H0_ratio, 20)
                #     low_f = chadwickModel(high_h/1000, E_ratio, H0_ratio/1000, self.DIAMETER/1000)

                #     # legendText = 'bestH0 = {:.2f}nm'.format(bestH0) + '\n' + str_m_z
                #     plot_startH = np.concatenate((self.dictH0['hArray_' + str_m_z][::-1], high_h))
                #     plot_startF = np.concatenate((self.dictH0['fArray_' + str_m_z][::-1], low_f))

                #     ax.plot([H0_ratio], [0], ls = '', marker = 'o', color = 'darkslateblue', markersize = 5, zorder = 3)
                #             # label = legendText)
                #     ax.plot(plot_startH, plot_startF, ls = '--', color = 'darkslateblue', linewidth = 1.2, zorder = 3)
                    

                ax.legend(loc = 'upper right', prop={'size': 6})
                ax.title.set_text(titleText)

                
                
            ax = ufun.setAllTextFontSize(ax, size = 9)
            ax.legend(loc = 'upper right', prop={'size': 6})
            ax.title.set_text(titleText)
            
                    
            # if plotSettings['Plot_Ratio'] and (not self.error_bestH0):
            #     ax_r = ax.twinx()
            #     ax_r.plot(self.hCompr, self.ChadwickRatio, color='gold', marker='o', markersize=1, lw=0, zorder = 1)
            #     ax_r.set_ylabel('a/h0')
            #     ax_r = ufun.setAllTextFontSize(ax_r, size = 9)
            #     ax_r.axhline(1, ls='--', lw=0.5, color = 'skyblue')
            #     ax_r.axhline(2, ls='--', lw=0.5, color = 'orange')
            #     ax_r.set_ylim([0,10])
            
    def plot_FH_ChadAndDimi(self, fig, ax, plotSettings, plotH0 = True, plotFit = True):
        """
        

        Parameters
        ----------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        plotSettings : TYPE
            DESCRIPTION.
        plotH0 : TYPE, optional
            DESCRIPTION. The default is True.
        plotFit : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        if self.isValidForAnalysis:
            ax.plot(self.hCompr, self.fCompr,'b-', linewidth = 0.8)
            ax.plot(self.hRelax, self.fRelax,'r-', linewidth = 0.8)
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            legendText = ''
            ax.set_xlabel('h (nm)')
            ax.set_ylabel('f (pN)')
    
            if plotFit:
                
                #### Dimitriadis
                try:
                    method = 'Full'
                    # dictFit = self.dictFitFH_Dimitriadis[method]
                    dictFit = self.dictFitFH_Dimitriadis[method]
                    fitError = dictFit['error']
                        
                    if not fitError:
                        H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                        fPredict = dictFit['yPredict']
                        hFit = dictFit['x']
                        
                        legendText = 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                        ax.plot(hFit, fPredict, ls=':', color = 'darkgreen', linewidth = 0.8, 
                                label = legendText, zorder = 2)
                    # else:
                    #     titleText += '\nFIT ERROR'
                except:
                    pass
                
                try:
                    method = 'Valid'
                    # dictFit = self.dictFitFH_Dimitriadis[method]
                    dictFit = self.dictFitFH_Dimitriadis[method]
                    fitError = dictFit['error']
                        
                    if not fitError:
                        H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                        fPredict = dictFit['yPredict']
                        hFit = dictFit['x']
                        
                        legendText = 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                        ax.plot(hFit, fPredict, ls=':', color = 'yellowgreen', linewidth = 0.8, 
                                label = legendText, zorder = 2)
                    # else:
                    #     titleText += '\nFIT ERROR'
                except:
                    pass
                
                #### Chadwick
                # try:
                #     method = 'Full'
                #     # dictFit = self.dictFitFH_Chadwick[method]
                #     dictFit = self.dictFitFH_Chadwick[method]
                #     fitError = dictFit['error']
                        
                #     if not fitError:
                #         H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                #         fFit = dictFit['x']
                #         hPredict = dictFit['yPredict']
                        
                #         legendText = 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                #         ax.plot(hPredict, fFit, ls='--', color = 'darkred', linewidth = 0.8, 
                #                 label = legendText, zorder = 2)
                #     # else:
                #     #     titleText += '\nFIT ERROR'
                # except:
                #     pass
                    
                try:
                    method = 'f_<_500'
                    # dictFit = self.dictFitFH_Chadwick[method]
                    dictFit = self.dictFitFH_Chadwick[method]
                    fitError = dictFit['error']
                        
                    if not fitError:
                        H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                        fFit = dictFit['x']
                        hPredict = dictFit['yPredict']
                        
                        legendText = 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                        ax.plot(hPredict, fFit, ls='--', color = 'darkorange', linewidth = 0.8, 
                                label = legendText, zorder = 2)
                except:
                    pass
 
                
                    
            if plotH0:
                bestH0 = self.bestH0
                method = self.method_bestH0
                zone = self.zone_bestH0
                str_m_z = method + '_' + zone
                E_bestH0 = self.dictH0['E_' + method + '_' + zone]
                
                if (not self.error_bestH0) and (method not in ['NaiveMax']):
                    max_h = np.max(self.hCompr)
                    high_h = np.linspace(max_h, bestH0, 20)
                    if self.method_bestH0 == 'Dimitriadis':
                        low_f = dimitriadisModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
                    elif self.method_bestH0 == 'Chadwick':
                        # chadwickModel(h, E, H0, DIAMETER)
                        low_f = chadwickModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
                    else:
                        low_f = np.ones_like(high_h) * bestH0
                    
                    legendText = 'bestH0 = {:.2f}nm'.format(bestH0) + '\n' + str_m_z
                    plot_startH = np.concatenate((self.dictH0['hArray_' + str_m_z][::-1], high_h))
                    plot_startF = np.concatenate((self.dictH0['fArray_' + str_m_z][::-1], low_f))

                    ax.plot([bestH0], [0], ls = '', marker = 'o', color = 'skyblue', markersize = 5, 
                            label = legendText)
                    ax.plot(plot_startH, plot_startF, ls = '--', color = 'skyblue', linewidth = 1.2, zorder = 4)

                    
                # if 'H0_Dimitriadis_' + 'ratio_2-2.5' in self.dictH0.keys():
                #     H0_ratio = self.dictH0['H0_Dimitriadis_ratio_2-2.5']
                #     E_ratio = self.dictH0['E_Dimitriadis_ratio_2-2.5']
                #     str_m_z = 'Dimitriadis_ratio_2-2.5'
                #     max_h = np.max(self.hCompr)
                #     high_h = np.linspace(max_h, H0_ratio, 20)
                #     low_f = DimitriadisModel(high_h/1000, E_ratio, H0_ratio/1000, self.DIAMETER/1000)

                #     # legendText = 'bestH0 = {:.2f}nm'.format(bestH0) + '\n' + str_m_z
                #     plot_startH = np.concatenate((self.dictH0['hArray_' + str_m_z][::-1], high_h))
                #     plot_startF = np.concatenate((self.dictH0['fArray_' + str_m_z][::-1], low_f))

                #     ax.plot([H0_ratio], [0], ls = '', marker = 'o', color = 'darkslateblue', markersize = 5, zorder = 3)
                #             # label = legendText)
                #     ax.plot(plot_startH, plot_startF, ls = '--', color = 'darkslateblue', linewidth = 1.2, zorder = 3)
                    

                ax.legend(loc = 'upper right', prop={'size': 6})
                ax.title.set_text(titleText)

                
                
            ax = ufun.setAllTextFontSize(ax, size = 9)
            ax.legend(loc = 'upper right', prop={'size': 6})
            ax.title.set_text(titleText)
            
                    
            # if plotSettings['Plot_Ratio'] and (not self.error_bestH0):
            #     ax_r = ax.twinx()
            #     ax_r.plot(self.hCompr, self.ChadwickRatio, color='gold', marker='o', markersize=1, lw=0, zorder = 1)
            #     ax_r.set_ylabel('a/h0')
            #     ax_r = ufun.setAllTextFontSize(ax_r, size = 9)
            #     ax_r.axhline(1, ls='--', lw=0.5, color = 'skyblue')
            #     ax_r.axhline(2, ls='--', lw=0.5, color = 'orange')
            #     ax_r.set_ylim([0,10])
    
    def plot_FH_Dimitriadis(self, fig, ax, plotSettings, plotH0 = True, plotFit = True):
        """
        

        Parameters
        ----------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        plotSettings : TYPE
            DESCRIPTION.
        plotH0 : TYPE, optional
            DESCRIPTION. The default is True.
        plotFit : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        if self.isValidForAnalysis:
            ax.plot(self.hCompr, self.fCompr,'b-', linewidth = 0.8)
            ax.plot(self.hRelax, self.fRelax,'r-', linewidth = 0.8)
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            legendText = ''
            ax.set_xlabel('h (nm)')
            ax.set_ylabel('f (pN)')
    
            if plotFit:
                method = 'Valid'
                # dictFit = self.dictFitFH_Dimitriadis[method]
                dictFit = self.dictFitFH_Dimitriadis[method]
                fitError = dictFit['error']
                    
                if not fitError:
                    H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                    fPredict = dictFit['yPredict']
                    hFit = dictFit['x']
                    
                    legendText = 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                    ax.plot(hFit, fPredict, ls='--', color = 'black', linewidth = 0.8, 
                            label = legendText, zorder = 2)
                # else:
                #     titleText += '\nFIT ERROR'
                    
                    
            if plotH0:
                bestH0 = self.bestH0
                method = self.method_bestH0
                zone = self.zone_bestH0
                str_m_z = method + '_' + zone
                E_bestH0 = self.dictH0['E_' + method + '_' + zone]
                
                if (not self.error_bestH0) and (method not in ['NaiveMax']):
                    max_h = np.max(self.hCompr)
                    high_h = np.linspace(max_h, bestH0, 20)
                    if self.method_bestH0 == 'Dimitriadis':
                        low_f = dimitriadisModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
                    elif self.method_bestH0 == 'Chadwick':
                        # chadwickModel(h, E, H0, DIAMETER)
                        low_f = chadwickModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
                    else:
                        low_f = np.ones_like(high_h) * bestH0
                    
                    legendText = 'bestH0 = {:.2f}nm'.format(bestH0) + '\n' + str_m_z
                    plot_startH = np.concatenate((self.dictH0['hArray_' + str_m_z][::-1], high_h))
                    plot_startF = np.concatenate((self.dictH0['fArray_' + str_m_z][::-1], low_f))

                    ax.plot([bestH0], [0], ls = '', marker = 'o', color = 'skyblue', markersize = 5, 
                            label = legendText)
                    ax.plot(plot_startH, plot_startF, ls = '--', color = 'skyblue', linewidth = 1.2, zorder = 4)

                    
                # if 'H0_Dimitriadis_' + 'ratio_2-2.5' in self.dictH0.keys():
                #     H0_ratio = self.dictH0['H0_Dimitriadis_ratio_2-2.5']
                #     E_ratio = self.dictH0['E_Dimitriadis_ratio_2-2.5']
                #     str_m_z = 'Dimitriadis_ratio_2-2.5'
                #     max_h = np.max(self.hCompr)
                #     high_h = np.linspace(max_h, H0_ratio, 20)
                #     low_f = DimitriadisModel(high_h/1000, E_ratio, H0_ratio/1000, self.DIAMETER/1000)

                #     # legendText = 'bestH0 = {:.2f}nm'.format(bestH0) + '\n' + str_m_z
                #     plot_startH = np.concatenate((self.dictH0['hArray_' + str_m_z][::-1], high_h))
                #     plot_startF = np.concatenate((self.dictH0['fArray_' + str_m_z][::-1], low_f))

                #     ax.plot([H0_ratio], [0], ls = '', marker = 'o', color = 'darkslateblue', markersize = 5, zorder = 3)
                #             # label = legendText)
                #     ax.plot(plot_startH, plot_startF, ls = '--', color = 'darkslateblue', linewidth = 1.2, zorder = 3)
                    

                ax.legend(loc = 'upper right', prop={'size': 6})
                ax.title.set_text(titleText)

                
                
            ax = ufun.setAllTextFontSize(ax, size = 9)
            ax.legend(loc = 'upper right', prop={'size': 6})
            ax.title.set_text(titleText)
            
                    
            # if plotSettings['Plot_Ratio'] and (not self.error_bestH0):
            #     ax_r = ax.twinx()
            #     ax_r.plot(self.hCompr, self.ChadwickRatio, color='gold', marker='o', markersize=1, lw=0, zorder = 1)
            #     ax_r.set_ylabel('a/h0')
            #     ax_r = ufun.setAllTextFontSize(ax_r, size = 9)
            #     ax_r.axhline(1, ls='--', lw=0.5, color = 'skyblue')
            #     ax_r.axhline(2, ls='--', lw=0.5, color = 'orange')
            #     ax_r.set_ylim([0,10])
    
    
    
    
    def plot_FH(self, fig, ax, plotSettings, plotH0 = True, plotFit = True):
        """
        

        Parameters
        ----------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        plotSettings : TYPE
            DESCRIPTION.
        plotH0 : TYPE, optional
            DESCRIPTION. The default is True.
        plotFit : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        if self.isValidForAnalysis:
            ax.plot(self.hCompr, self.fCompr,'b-', linewidth = 0.8)
            ax.plot(self.hRelax, self.fRelax,'r-', linewidth = 0.8)
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            legendText = ''
            ax.set_xlabel('h (nm)')
            ax.set_ylabel('f (pN)')
    
            if plotFit:
                method = 'Full'
                # dictFit = self.dictFitFH_Chadwick[method]
                dictFit = self.dictFitFH_Chadwick[method]
                fitError = dictFit['error']
                    
                if not fitError:
                    H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                    fFit = dictFit['x']
                    hPredict = dictFit['yPredict']
                    
                    legendText = 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                    ax.plot(hPredict, fFit,'k--', linewidth = 0.8, 
                            label = legendText, zorder = 2)
                # else:
                #     titleText += '\nFIT ERROR'
                    
                method = 'f_<_500'
                # dictFit = self.dictFitFH_Chadwick[method]
                dictFit = self.dictFitFH_Chadwick[method]
                fitError = dictFit['error']
                    
                if not fitError:
                    H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                    fFit = dictFit['x']
                    hPredict = dictFit['yPredict']
                    
                    legendText = 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                    ax.plot(hPredict, fFit,'g--', linewidth = 0.8, 
                            label = legendText, zorder = 2)
                # else:
                #     titleText += '\nFIT ERROR'
                
                try:
                    method = 'f_in_400_800'
                    # dictFit = self.dictFitFH_Chadwick[method]
                    dictFit = self.dictFitFH_Chadwick[method]
                    fitError = dictFit['error']
                        
                    if not fitError:
                        H0, E, R2, Chi2 = dictFit['H0'], dictFit['E'], dictFit['R2'], dictFit['Chi2']
                        fFit = dictFit['x']
                        hPredict = dictFit['yPredict']
                        
                        legendText = 'H0 = {:.1f}nm\nE = {:.2e}Pa\nR2 = {:.3f}\nChi2 = {:.1f}'.format(H0, E, R2, Chi2)
                        ax.plot(hPredict, fFit, ls='--', color = 'darkorange', linewidth = 0.8, 
                                label = legendText, zorder = 2)
                    # else:
                    #     titleText += '\nFIT ERROR'
                except:
                    pass
                    
            if plotH0:
                bestH0 = self.bestH0
                method = self.method_bestH0
                zone = self.zone_bestH0
                str_m_z = method + '_' + zone
                E_bestH0 = self.dictH0['E_' + method + '_' + zone]
                
                if (not self.error_bestH0) and (method not in ['NaiveMax']):
                    max_h = np.max(self.hCompr)
                    high_h = np.linspace(max_h, bestH0, 20)
                    if self.method_bestH0 == 'Dimitriadis':
                        low_f = dimitriadisModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
                    elif self.method_bestH0 == 'Chadwick':
                        # chadwickModel(h, E, H0, DIAMETER)
                        low_f = chadwickModel(high_h/1000, E_bestH0, bestH0/1000, self.DIAMETER/1000)
                    else:
                        low_f = np.ones_like(high_h) * bestH0
                    
                    legendText = 'bestH0 = {:.2f}nm'.format(bestH0) + '\n' + str_m_z
                    plot_startH = np.concatenate((self.dictH0['hArray_' + str_m_z][::-1], high_h))
                    plot_startF = np.concatenate((self.dictH0['fArray_' + str_m_z][::-1], low_f))

                    ax.plot([bestH0], [0], ls = '', marker = 'o', color = 'skyblue', markersize = 5, 
                            label = legendText)
                    ax.plot(plot_startH, plot_startF, ls = '--', color = 'skyblue', linewidth = 1.2, zorder = 4)

                    
                # if 'H0_Chadwick_' + 'ratio_2-2.5' in self.dictH0.keys():
                #     H0_ratio = self.dictH0['H0_Chadwick_ratio_2-2.5']
                #     E_ratio = self.dictH0['E_Chadwick_ratio_2-2.5']
                #     str_m_z = 'Chadwick_ratio_2-2.5'
                #     max_h = np.max(self.hCompr)
                #     high_h = np.linspace(max_h, H0_ratio, 20)
                #     low_f = chadwickModel(high_h/1000, E_ratio, H0_ratio/1000, self.DIAMETER/1000)

                #     # legendText = 'bestH0 = {:.2f}nm'.format(bestH0) + '\n' + str_m_z
                #     plot_startH = np.concatenate((self.dictH0['hArray_' + str_m_z][::-1], high_h))
                #     plot_startF = np.concatenate((self.dictH0['fArray_' + str_m_z][::-1], low_f))

                #     ax.plot([H0_ratio], [0], ls = '', marker = 'o', color = 'darkslateblue', markersize = 5, zorder = 3)
                #             # label = legendText)
                #     ax.plot(plot_startH, plot_startF, ls = '--', color = 'darkslateblue', linewidth = 1.2, zorder = 3)
                    

                ax.legend(loc = 'upper right', prop={'size': 6})
                ax.title.set_text(titleText)

                
                
            ax = ufun.setAllTextFontSize(ax, size = 9)
            ax.legend(loc = 'upper right', prop={'size': 6})
            ax.title.set_text(titleText)
            
                    
            if plotSettings['Plot_Ratio'] and (not self.error_bestH0):
                ax_r = ax.twinx()
                ax_r.plot(self.hCompr, self.ChadwickRatio, color='gold', marker='o', markersize=1, lw=0, zorder = 1)
                ax_r.set_ylabel('a/h0')
                ax_r = ufun.setAllTextFontSize(ax_r, size = 9)
                ax_r.axhline(1, ls='--', lw=0.5, color = 'skyblue')
                ax_r.axhline(2, ls='--', lw=0.5, color = 'orange')
                ax_r.set_ylim([0,10])
            
        
    def plot_SS(self, fig, ax, plotSettings, plotFit = True, fitType = 'stressRegion'):
        """
        

        Parameters
        ----------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        plotSettings : TYPE
            DESCRIPTION.
        plotFit : TYPE, optional
            DESCRIPTION. The default is True.
        fitType : TYPE, optional
            DESCRIPTION. The default is 'stressRegion'.

        Returns
        -------
        None.

        """
        if self.isValidForAnalysis and not self.error_bestH0:
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            ax.title.set_text(titleText)
            ax.set_xlabel('Strain')
            ax.set_ylabel('Stress (Pa)')
            main_color = 'k'
            ls = '-'
            lw = 1.8
            
            ax.plot(self.strainCompr, self.stressCompr, 
                    color = main_color, marker = 'o', 
                    markersize = 2, ls = '', alpha = 0.8)
            
            if plotFit:
                if fitType == 'stressRegion':
                    # Read settings
                    HW = plotSettings['plotStressHW']
                    centers = plotSettings['plotStressCenters']
                    dictFit = self.dictFitsSS_stressRegions
                    id_ranges = [str(C) + '_' + str(HW) for C in centers]
                    # colorDict = {id_ranges[k]:gs.colorList30[k] for k in range(len(id_ranges))}
                    for k in range(len(id_ranges)):
                        idr = id_ranges[k]
                        try:
                            d = dictFit[idr]
                            if not d['error']:
                                if not d['valid']:
                                    ls = '--'
                                x = d['x']
                                y = d['yPredict']
                                color = gs.colorList30[k]
                                ax.plot(y, x, color = color, ls = ls, lw = lw)
                        except:
                            pass
                            
                if fitType == 'stressGaussian':
                    # Read settings
                    HW = plotSettings['plotStressHW']
                    centers = plotSettings['plotStressCenters']
                    dictFit = self.dictFitsSS_stressGaussian
                    id_ranges = [str(C) + '_' + str(HW) for C in centers]
                    # colorDict = {id_ranges[k]:gs.colorList30[k] for k in range(len(id_ranges))}
                    for k in range(len(id_ranges)):
                        idr = id_ranges[k]
                        try:
                            d = dictFit[idr]
                            if not d['error']:
                                if not d['valid']:
                                    ls = '--'
                                x = d['x']
                                y = d['yPredict']
                                color = gs.colorList30[k]
                                ax.plot(y, x, color = color, ls = ls, lw = lw)
                        except:
                            pass
                            
                if fitType == 'nPoints':
                    dictFit = self.dictFitsSS_nPoints
                    id_ranges = list(dictFit.keys())
                    # colorDict = {id_ranges[k]:gs.colorList30[k] for k in range(len(id_ranges))}
                    for k in range(len(id_ranges)):
                        idr = id_ranges[k]
                        d = dictFit[idr]
                        if not d['error']:
                            if not d['valid']:
                                ls = '--'
                            x = d['x']
                            y = d['yPredict']
                            color = gs.colorList30[k]
                            ax.plot(y, x, color = color, ls = ls, lw = lw)
                
                if fitType == 'Log':
                    dictFit = self.dictFitsSS_log
                    id_ranges = list(dictFit.keys())
                    # colorDict = {id_ranges[k]:gs.colorList30[k] for k in range(len(id_ranges))}
                    for k in range(len(id_ranges)):
                        idr = id_ranges[k]
                        d = dictFit[idr]
                        if not d['error']:
                            if not d['valid']:
                                ls = '--'
                            x = d['x']
                            y = d['yPredict']
                            color = gs.colorList30[k]
                            ax.plot(y, x, color = color, ls = ls, lw = lw)
                    
                            
                if fitType == 'strainGaussian':
                    # Read settings
                    HW = plotSettings['plotStrainHW']
                    centers = plotSettings['plotStrainCenters']
                    dictFit = self.dictFitsSS_strainGaussian
                    id_ranges = [str(C) + '_' + str(HW) for C in centers]
                    # colorDict = {id_ranges[k]:gs.colorList30[k] for k in range(len(id_ranges))}
                    for k in range(len(id_ranges)):
                        idr = id_ranges[k]
                        try:
                            d = dictFit[idr]
                            if not d['error']:
                                if not d['valid']:
                                    ls = '--'
                                x = d['x']
                                y = d['yPredict']
                                color = gs.colorList30[k]
                                ax.plot(x, y, color = color, ls = ls, lw = lw)
                                
                                # new_tick_locations = np.array([0.05, 0.1, 0.15])

                                # def tick_function(eps, h0):
                                #     H = h0 - 3*h0*eps
                                #     return(["%.1f" % h for h in H])
                                
                                # ax2 = ax.twiny()
                                # ax2.set_xlim(ax.get_xlim())
                                # ax2.set_xticks(new_tick_locations)
                                # ax2.set_xticklabels(tick_function(new_tick_locations, self.bestH0))
                                # ax2.set_xlabel(r"h (nm)")
                                # ax2.grid()
                        except:
                            pass
                
                #### TEST
                if fitType == '3parts':
                    dictFit = self.dictFitsSS_3parts
                    id_ranges = list(dictFit.keys())
                    # colorDict = {id_ranges[k]:gs.colorList30[k] for k in range(len(id_ranges))}
                    for k in range(len(id_ranges)):
                        idr = id_ranges[k]
                        d = dictFit[idr]
                        if not d['error']:
                            if not d['valid']:
                                ls = '--'
                            x = d['x']
                            y = d['yPredict']
                            color = gs.colorList30[k]
                            ax.plot(y, x, color = color, ls = ls, lw = lw)
                   
            ax = ufun.setAllTextFontSize(ax, size = 9)
            
            # NEW ! Jojo
            if plotSettings['Plot_Ratio'] and (not self.error_bestH0):
                ax_r = ax.twinx()
                ax_r.plot(self.strainCompr, self.ChadwickRatio, color='gold', marker='o', markersize=1, lw=0, zorder = 1)
                ax_r.set_ylabel('a/h0')
                ax_r = ufun.setAllTextFontSize(ax_r, size = 9)
                ax_r.axhline(1, ls='--', lw=0.5, color = 'skyblue')
                ax_r.axhline(2, ls='--', lw=0.5, color = 'orange')
                ax_r.set_ylim([0,10])
    
                
            
            
    
        
    def plot_KS(self, fig, ax, plotSettings, fitType = 'stressRegion'):
        """
        

        Parameters
        ----------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        plotSettings : TYPE
            DESCRIPTION.
        fitType : TYPE, optional
            DESCRIPTION. The default is 'stressRegion'.

        Returns
        -------
        None.

        """
        if self.isValidForAnalysis and not self.error_bestH0:
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            ax.title.set_text(titleText)
            ax.set_ylabel('K (kPa)')
            # ax.set_ylim([0, 16])
            
            if fitType == 'stressRegion':
                df = self.df_stressRegions
                ax.set_xlabel('Stress (Pa)')
                # ax.set_xlim([0, 1200])
                
                # Read settings
                HW = plotSettings['plotStressHW']
                centers = plotSettings['plotStressCenters']
                # N = len(centers)
                # colors = gs.colorList30[:N]
                
                fltr = (df['center_x'].apply(lambda x : x in centers)) & \
                       (df['halfWidth_x'] == HW)
                
                df_fltr = df[fltr]
                N = len(df_fltr['center_x'].values)
                colors = gs.colorList30[:N]
                N_col = len(colors)

                for k in range(N):
                    X = df_fltr['center_x'].values[k]
                    Y = df_fltr['K'].values[k]/1000
                    Yerr = df_fltr['ciwK'].values[k]/1000
                    if df_fltr['valid'].values[k]:
                        color = colors[k%N_col]
                        mec = 'k'
                    else:
                        # color = 'w'
                        # mec = colors[k%N_col]
                        color = colors[k%N_col]
                        mec = 'none'
                    if (not pd.isnull(Y)) and (Y > 0):
                        ax.errorbar(X, Y, yerr = Yerr, color = color, marker = 'o', 
                                    ms = 5, mec = mec, ecolor = color) 
            
            if fitType == 'stressGaussian':
                df = self.df_stressGaussian
                ax.set_xlabel('Stress (Pa)')
                # ax.set_xlim([0, 1200])
                
                # Read settings
                HW = plotSettings['plotStressHW']
                centers = plotSettings['plotStressCenters']
                # N = len(centers)
                # colors = gs.colorList30[:N]
                
                fltr = (df['center_x'].apply(lambda x : x in centers)) & \
                       (df['halfWidth_x'] == HW)
                
                df_fltr = df[fltr]
                N = len(df_fltr['center_x'].values)
                colors = gs.colorList30[:N]
                N_col = len(colors)
                
                # relativeError[k] = (Err/K_fit)
                # # mec = None
                
                for k in range(N):
                    X = df_fltr['center_x'].values[k]
                    Y = df_fltr['K'].values[k]/1000
                    Yerr = df_fltr['ciwK'].values[k]/1000
                    if df_fltr['valid'].values[k]:
                        color = colors[k%N_col]
                        mec = 'k'
                    else:
                        # color = 'w'
                        # mec = colors[k%N_col]
                        color = colors[k%N_col]
                        mec = 'none'
                        
                    if (not pd.isnull(Y)) and (Y > 0):
                        ax.errorbar(X, Y, yerr = Yerr, color = color, marker = 'o', 
                                    ms = 5, mec = mec, ecolor = color) 
                
            if fitType == 'nPoints':
                df = self.df_nPoints
                ax.set_xlabel('Stress (Pa)')
                # ax.set_xlim([0, 1200])
    
                N = df.shape[0]
                colors = gs.colorList30[:N]
                N_col = len(colors)

                for k in range(N):
                    X = df['center_x'].values[k]
                    Y = df['K'].values[k]/1000
                    Yerr = df['ciwK'].values[k]/1000
                    if df['valid'].values[k]:
                        color = colors[k%N_col]
                        mec = 'k'
                    else:
                        # color = 'w'
                        # mec = colors[k%N_col]
                        color = colors[k%N_col]
                        mec = 'none'
                        
                    if (not pd.isnull(Y)) and (Y > 0):
                        ax.errorbar(X, Y, yerr = Yerr, color = color, marker = 'o', 
                                    ms = 5, mec = mec, ecolor = color)
                        
                        
            if fitType == 'strainGaussian':
                df = self.df_strainGaussian
                ax.set_xlabel('Strain')
                ax.autoscale()
                # ax.set_xlim([0, 1200])
                
                # Read settings
                HW = plotSettings['plotStrainHW']
                centers = plotSettings['plotStrainCenters']
                # N = len(centers)
                # colors = gs.colorList30[:N]
                fltr = (df['center_x'].apply(lambda x : x in centers)) & \
                       (df['halfWidth_x'] == HW)
                
                df_fltr = df[fltr]
                N = len(df_fltr['center_x'].values)
                colors = gs.colorList30[:N]
                N_col = len(colors)
                
                # relativeError[k] = (Err/K_fit)
                # # mec = None
                
                for k in range(N):
                    X = df_fltr['center_x'].values[k]
                    Y = df_fltr['K'].values[k]/1000
                    Yerr = df_fltr['ciwK'].values[k]/1000
                    if df_fltr['valid'].values[k]:
                        color = colors[k%N_col]
                        mec = 'k'
                    else:
                        # color = 'w'
                        # mec = colors[k%N_col]
                        color = colors[k%N_col]
                        mec = 'none'
                        
                    if (not pd.isnull(Y)) and (Y > 0):
                        ax.errorbar(X, Y, yerr = Yerr, color = color, marker = 'o', 
                                    ms = 5, mec = mec, ecolor = color)
            
            if fitType == 'Log':
                df = self.df_log
                ax.set_xlabel('Stress (Pa)')
                ax.set_xlim([0, 1200])
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                # ax.set_ylim([0, 6])
                legendText = ''
                N = df.shape[0]
                colors = gs.colorList30[:N]
                XtoFit = []
                YtoFit = []
                YerrToFit = []
                colorMask = []
                for k in range(N):
                    X = df['center_x'].values[k]
                    Y = df['K'].values[k]/1000
                    Yerr = df['ciwK'].values[k]/1000

                    if df['valid'].values[k]:
                        color = colors[k]
                        mec = 'none'
                        XtoFit.append(X)
                        YtoFit.append(Y)
                        YerrToFit.append(Yerr)
                    else:
                        color = 'w'
                        mec = colors[k]
                        
                    if (not pd.isnull(Y)) and (Y > 0):
                        ax.errorbar(X, Y, yerr = Yerr, color = color, marker = 'o', 
                                    ms = 5, mec = mec, ecolor = colors[k])
                
                XtoFit, YtoFit, YerrToFit = np.asarray(XtoFit), np.asarray(YtoFit), np.asarray(YerrToFit)
                
                if len(XtoFit) > 4:
                    posValues = ((XtoFit > 0) & (YtoFit > 0))
                    XtoFit, YtoFit = XtoFit[posValues], YtoFit[posValues]
                    # print(YerrToFit)
     
                    #Unwighted linear regression
                    # params, results = ufun.fitLine(np.log(XtoFit), np.log(YtoFit)) # Y=a*X+b ; params[0] = b,  params[1] = a
                    
                    #Weighted linear regression
                    params, results = ufun.fitLineWeighted(np.log(XtoFit), np.log(YtoFit), 1/(YerrToFit))
                    k = np.exp(params[0])
                    a = params[1]
                    R2 = results.rsquared
                    pval = results.pvalues[1] # pvalue on the param 'a'
                    legendText += " Y = {:.1e} * X^{:.1f}".format(k, a)
                    legendText += " \n p-val = {:.3f}".format(pval)
                    legendText += " \n R2 = {:.2f}".format(R2)
                    # print("Y = {:.4e} * X^{:.4f}".format(k, a))
                    # print("p-value on the 'a' coefficient: {:.4e}".format(pval))
                    # print("R2 of the fit: {:.4f}".format(R2))
                    # fitY = k * X**a
                    # imin = np.argmin(X)
                    # imax = np.argmax(X)
                    # ax.plot([X[imin],X[imax]], [fitY[imin],fitY[imax]], '--', lw = '1', 
                    #         color = color, zorder = 4)
                    fitX = np.linspace(np.min(XtoFit), np.max(XtoFit), 100)
                    fitY = k * fitX**a
                    # ax.plot(fitX, fitY, '--', lw = '1', color = 'red', zorder = 4, label = legendText)
                    # ax.legend(loc = 'upper left', prop={'size': 6})
        
        
            for item in ([ax.title, ax.xaxis.label, \
                          ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(9)
    
    #### METHODS IN DEVELOPMENT                
                
    def fitSS_strainGaussian(self, center, halfWidth, fitValidationSettings):
        """
        

        Parameters
        ----------
        center : TYPE
            DESCRIPTION.
        halfWidth : TYPE
            DESCRIPTION.
        fitValidationSettings : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        id_range = str(center) + '_' + str(halfWidth)
        stress, strain = self.stressCompr, self.strainCompr
        
        X = strain.flatten(order='C')
        weights = np.exp( -((X - center) ** 2) / halfWidth ** 2)
        
        params, ses, error = fitLinear_ss_i(strain, stress, weights = weights)
        
        K, stress0 = params
        lowStrain, highStrain = center - halfWidth, center + halfWidth
        mask = ((strain > lowStrain) & (strain < highStrain))
        stressPredict = constitutiveRelation(strain[mask], K, stress0)
        
        x = strain[mask]
        y, yPredict = stress[mask], stressPredict
        #### err_Chi2 for stress
        err_chi2 = err_chi2_Stress
        
        dictFit = makeDictFit_ss(params, ses, error, 
                                 center, halfWidth, x, y, yPredict, 
                                 err_chi2, fitValidationSettings)
        self.dictFitsSS_strainGaussian[id_range] = dictFit
        
        
            
        
    def plot_KS_Xthickness(self, fig, ax, plotSettings, fitType = 'stressRegion'):
        """
        

        Parameters
        ----------
        fig : TYPE
            DESCRIPTION.
        ax : TYPE
            DESCRIPTION.
        plotSettings : TYPE
            DESCRIPTION.
        fitType : TYPE, optional
            DESCRIPTION. The default is 'stressRegion'.

        Returns
        -------
        None.

        """
        if self.isValidForAnalysis and not self.error_bestH0:
            titleText = self.cellID + '__c' + str(self.i_indent + 1)
            ax.title.set_text(titleText)
            ax.set_ylabel('K (kPa)')
            ax.set_ylim([0, 16])

            df = self.df_strainGaussian
            ax.set_xlabel('Strain')
            # ax.set_xlim([0, 1200])
            
            # Read settings
            HW = plotSettings['plotStrainHW']
            centers = plotSettings['plotStrainCenters']
            # N = len(centers)
            # colors = gs.colorList30[:N]
            
            fltr = (df['center_x'].apply(lambda x : x in centers)) & \
                   (df['halfWidth_x'] == HW)
            
            df_fltr = df[fltr]
            N = len(df_fltr['center_x'].values)
            colors = gs.colorList30[:N]
            N_col = len(colors)
            
            # relativeError[k] = (Err/K_fit)
            # # mec = None
            
            for k in range(N):
                X = df_fltr['center_x'].values[k]
                Y = df_fltr['K'].values[k]/1000
                Yerr = df_fltr['ciwK'].values[k]/1000
                if df_fltr['valid'].values[k]:
                    color = colors[k%N_col]
                    mec = 'none'
                else:
                    color = 'w'
                    mec = colors[k%N_col]
                if (not pd.isnull(Y)) and (Y > 0):
                    ax.errorbar(X, Y, yerr = Yerr, color = color, marker = 'o', 
                                ms = 5, mec = mec, ecolor = color) 
                
            
            
            for item in ([ax.title, ax.xaxis.label, \
                          ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(9)
                
        
    def fitFH_Chadwick_fixedH0(self, fitValidationSettings, method = 'Full', mask = []):
        """
        

        Parameters
        ----------
        fitValidationSettings : TYPE
            DESCRIPTION.
        mask : TYPE, optional
            DESCRIPTION. The default is [].

        Returns
        -------
        None.

        """
        if len(mask) == 0:
            mask = np.ones_like(self.hCompr, dtype = bool)
        h, f, D, H0 = self.hCompr[mask], self.fCompr[mask], self.DIAMETER, self.bestH0
        
        params, ses, error = fitChadwick_hf_fixedH0(h, f, D, H0)
        
        E = params[0]
        seE = ses[0]
        params = (E, H0)
        ses = (seE, 0)
        hPredict = inversedChadwickModel(f, E, H0/1000, self.DIAMETER/1000)*1000
        x = f
        y, yPredict = h, hPredict
        #### err_Chi2 for distance (nm)
        err_chi2 = err_chi2_H
        dictFit = makeDictFit_hf(params, ses, error, 
                                 x, y, yPredict, 
                                 err_chi2, fitValidationSettings)
        print(dictFit)
        

        self.dictFitFH_Chadwick_fixedH0[method] = dictFit
        

    
class BadIndentCompression:
    def __init__(self, CC, indentDf, thisExpDf, i_indent):
        self.rawDf = indentDf
        self.thisExpDf = thisExpDf
        self.i_indent = i_indent
        self.i_tsDf = pd.DataFrame({})
        
        self.rawT0 = 0
        
        self.cellID = CC.cellID
        self.DIAMETER = CC.DIAMETER
        self.EXPTYPE = CC.EXPTYPE
        self.normalField = CC.normalField
        self.minCompField = CC.minCompField
        self.maxCompField = CC.maxCompField
        self.nUplet = CC.nUplet
        try:
            self.loopStruct = CC.loopStruct
            self.loop_totalSize = CC.loop_totalSize
            self.loop_rampSize = CC.loop_rampSize
            self.loop_ctSize = CC.loop_ctSize
        except:
            pass
        
        # These fields are to be modified or filled by methods later on
        
        # validateForAnalysis()
        self.isValidForAnalysis = False
        
        # refineStartStop()
        self.isRefined = False
        self.jMax = 0
        self.jStart = 0
        self.jStop = 0
        self.hCompr = []
        self.hRelax = []
        self.fCompr = []
        self.fRelax = []
        self.TCompr = []
        self.TRelax = []
        self.BCompr = []
        self.BRelax = []

        
        self.Df = self.rawDf
        
        # computeH0()
        self.dictH0 = {}
        
        # setBestH0()
        self.bestH0 = np.nan
        self.method_bestH0 = ''
        self.zone_bestH0 = ''
        self.error_bestH0 = True
        
        # computeStressStrain()
        self.deltaCompr = np.zeros_like(self.hCompr)*np.nan
        self.stressCompr = np.zeros_like(self.hCompr)*np.nan
        self.strainCompr = np.zeros_like(self.hCompr)*np.nan
        self.contactRadius = np.zeros_like(self.hCompr)*np.nan
        self.ChadwickRatio = np.zeros_like(self.hCompr)*np.nan
        
        # fitFH_Chadwick() & fitFH_Dimitriadis()
        self.dictFitFH_Chadwick = {}
        self.dictFitFH_Dimitriadis = {}
        self.dictFitFH_VWC = {}
        
        # Test of new chad fit
        # self.dictFitFH_Chadwick_fixedH0 = {}
        
        # fitSS_stressRegion() & fitSS_stressGaussian() & fitSS_nPoints()
        self.dictFitsSS_stressRegions = {}
        self.dictFitsSS_stressGaussian = {}
        self.dictFitsSS_nPoints = {}
        self.dictFitsSS_log = {}
        
        # dictFits_To_DataFrame()
        self.df_stressRegions = pd.DataFrame({})
        self.df_stressGaussian = pd.DataFrame({})
        self.df_nPoints = pd.DataFrame({})
        self.df_log = pd.DataFrame({})
        
        # NEW ! with strain
        self.dictFitsSS_strainGaussian = {} # fitSS_strainGaussian()
        self.df_strainGaussian = pd.DataFrame({}) # dictFits_To_DataFrame()
        
        
        #### TEST
        self.dictFitsSS_3parts = {} 
        self.df_3parts = pd.DataFrame({})
        self.computed_SSK_filteredDer = False



    
    
    
# %%%% Default settings

#### HOW TO USE:

# See ufun.updateDefaultSettingsDict(settingsDict, defaultSettingsDict)
# And the 'Settings' flag in analyseTimeSeries_meca() 

# %%%%% For Fits
DEFAULT_stressCenters = [ii for ii in range(100, 1550, 50)]
DEFAULT_stressHalfWidths = [50, 75, 100]

DEFAULT_strainCenters = [ii/10000 for ii in range(125, 3750, 125)]
DEFAULT_strainHalfWidths = [0.0125, 0.025, 0.05]

DEFAULT_fitSettings = {# H0
                       'methods_H0':['Chadwick', 'Dimitriadis'],
                       'zones_H0':['%f_10', '%f_20'],
                       'method_bestH0':'Chadwick',
                       'zone_bestH0':'%f_10',
                       # Global fits
                       'doVWCFit' : True,
                       'VWCFitMethods' : ['Full'],
                       'doChadwickFit' : True,
                       'ChadwickFitMethods' : ['Full', 'f_<_400', 'f_in_400_800'],
                       'doDimitriadisFit' : False,
                       'DimitriadisFitMethods' : ['Full'],
                       # Local fits
                       'doStressRegionFits' : True,
                       'doStressGaussianFits' : True,
                       'centers_StressFits' : DEFAULT_stressCenters,
                       'halfWidths_StressFits' : DEFAULT_stressHalfWidths,
                       'doNPointsFits' : True,
                       'nbPtsFit' : 13,
                       'overlapFit' : 3,
                       # NEW - Numi
                       'doLogFits' : False,
                       'nbPtsFitLog' : 10,
                       'overlapFitLog' : 5,
                       # NEW - Jojo
                       'doStrainGaussianFits' : True,
                       'centers_StrainFits' : DEFAULT_strainCenters,
                       'halfWidths_StrainFits' : DEFAULT_strainHalfWidths,
                       # TEST - Jojo
                       'do3partsFits' : False,
                       }


# %%%%% For Validation

DEFAULT_crit_nbPts = 8 # sup or equal to
DEFAULT_crit_R2 = 0.6 # sup or equal to
DEFAULT_crit_Chi2 = 1 # inf or equal to
DEFAULT_str_crit = 'nbPts>{:.0f} - R2>{:.2f} - Chi2<{:.1f}'.format(DEFAULT_crit_nbPts, 
                                                                   DEFAULT_crit_R2, 
                                                                   DEFAULT_crit_Chi2)

DEFAULT_fitValidationSettings = {'crit_nbPts': DEFAULT_crit_nbPts, 
                                 'crit_R2': DEFAULT_crit_R2, 
                                 'crit_Chi2': DEFAULT_crit_Chi2,
                                 'str': DEFAULT_str_crit}

# %%%%% For Plots
DEFAULT_plot_stressCenters = [ii for ii in range(100, 1550, 50)]
DEFAULT_plot_stressHalfWidth = 75

DEFAULT_plot_strainCenters = [ii/10000 for ii in range(125, 3750, 125)]
DEFAULT_plot_strainHalfWidth = 0.0125

DEFAULT_plotSettings = {# ON/OFF switchs plot by plot
                        'FH(t)':True,
                        'F(H)':True,
                        'F(H)_Dimitriadis':True,
                        'F(H)_ChadAndDimi':True,
                        'F(H)_VWC':True,
                        'S(e)_stressRegion':True,
                        'K(S)_stressRegion':True,
                        'S(e)_stressGaussian':True,
                        'K(S)_stressGaussian':True,
                        'S(e)_nPoints':True,
                        'K(S)_nPoints':True,
                        'S(e)_Log':True, # NEW - Numi
                        'K(S)_Log':True, # NEW - Numi
                        'S(e)_strainGaussian':True, # NEW - Jojo
                        'K(S)_strainGaussian':True, # NEW - Jojo
                        'Plot_Ratio':True, # NEW
                        # Fits plotting parameters
                        # Stress
                        'plotStressCenters':DEFAULT_plot_stressCenters,
                        'plotStressHW':DEFAULT_plot_stressHalfWidth,
                        # Strain
                        'plotStrainCenters':DEFAULT_plot_strainCenters,
                        'plotStrainHW':DEFAULT_plot_strainHalfWidth,
                        # Points
                        'plotPoints':str(DEFAULT_fitSettings['nbPtsFit']) \
                                     + '_' + str(DEFAULT_fitSettings['overlapFit']),
                        'plotLog':str(DEFAULT_fitSettings['nbPtsFitLog']) \
                                     + '_' + str(DEFAULT_fitSettings['overlapFitLog']),
                        # TEST
                        'S(e)_3parts': False,
                        }
        
# %%%% Main 

        
def analyseTimeSeries_meca(f, tsDf, expDf, taskName = '', PLOT = False, SHOW = False, 
                           fitSettings = {}, fitValidationSettings = {}, plotSettings = {}):
    """
    

    Parameters
    ----------
    f : string
        Name of the analysed file.
    tsDf : pandas DataFrame
        DataFrame containing the Time Series data (T, B, F, D3, etc).
    expDf : pandas DataFrame
        DataFrame containing the experimental conditions.
    taskName : string, optional
        Used as folder name to save the plots. The default is ''.
    PLOT : boolean, optional
        Make the plots or not. The default is False.
    SHOW : boolean, optional
        Show the plots or not. The default is False.
    fitSettings : dictionary, optional
        Modify default fit settings. The default is {} (no modifications).
    fitValidationSettings : dictionary, optional
        Modify default fit validation settings. The default is {} (no modifications).
    plotSettings : dictionary, optional
        Modify default fit plots settings. The default is {} (no modifications).

    Returns
    -------
    res : dictionary
        Contains the many result pandas DataFrame computed.

    """
    top = time.time()
    
    print(gs.BLUE + f + gs.NORMAL, end = ' ... ')
    plt.ioff()
    warnings.filterwarnings('ignore')
    
    
    #### 0. Settings
    #### 0.1 Fits settings
    fitSettings = ufun.updateDefaultSettingsDict(fitSettings, 
                                                 DEFAULT_fitSettings)
    

    
    method_bestH0 = fitSettings['method_bestH0']
    zone_bestH0 = fitSettings['zone_bestH0']
    
    centers_StressFits = fitSettings['centers_StressFits']
    halfWidths_StressFits = fitSettings['halfWidths_StressFits']
    nbPtsFit = fitSettings['nbPtsFit']
    overlapFit = fitSettings['overlapFit']
    nbPtsFitLog = fitSettings['nbPtsFitLog']
    overlapFitLog = fitSettings['overlapFitLog']
    
    centers_StrainFits = fitSettings['centers_StrainFits']
    halfWidths_StrainFits = fitSettings['halfWidths_StrainFits']
    
    #### 0.2 Options for fits validation
    fitValidationSettings = ufun.updateDefaultSettingsDict(fitValidationSettings, 
                                                           DEFAULT_fitValidationSettings)
    
    str_crit = 'nbPts>{:.0f} - R2>{:.2f} - Chi2<{:.1f}'.format(fitValidationSettings['crit_nbPts'], 
                                                               fitValidationSettings['crit_R2'], 
                                                               fitValidationSettings['crit_Chi2'])
    fitValidationSettings['str'] = str_crit
    fitValidationSettings
    
    
    #### 0.3 Plots
    plotSettings = ufun.updateDefaultSettingsDict(plotSettings, 
                                                  DEFAULT_plotSettings)
    plotSettings['subfolder_suffix'] = taskName
    plotSettings['plotPoints'] = str(fitSettings['nbPtsFit']) \
                                 + '_' + str(fitSettings['overlapFit'])
    plotSettings['plotLog'] = str(fitSettings['nbPtsFitLog']) \
                                 + '_' + str(fitSettings['overlapFitLog'])


    
    #### 1. Import experimental infos
    tsDf.dx, tsDf.dy, tsDf.dz, tsDf.D2, tsDf.D3 = tsDf.dx*1000, tsDf.dy*1000, tsDf.dz*1000, tsDf.D2*1000, tsDf.D3*1000
    thisManipID = ufun.findInfosInFileName(f, 'manipID')
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]
    Ncomp = max(tsDf['idxAnalysis'])
    cellID = ufun.findInfosInFileName(f, 'cellID')
    
    #### 2. Create CellCompression object
    CC = CellCompression(cellID, tsDf, thisExpDf, f)
    CC.method_bestH0 = method_bestH0
    
    #### 2.1 Correct jumps
    for i in range(Ncomp):
        CC.correctJumpForCompression(i)

    idxAnalysisVals = tsDf['idxAnalysis'].unique()
    idxComps = [i-1 for i in range(1, Ncomp+1) if i in idxAnalysisVals]    

    #### 3. Start looping over indents
    for i in range(Ncomp):
            
        #### 3.2 Segment the i-th compression
        maskComp = CC.getMaskForCompression(i, task = 'compression')
        thisCompDf = tsDf.loc[maskComp,:]
        i_tsDf = ufun.findFirst(1, maskComp)
        
        try:
            #### 3.3 Create IndentCompression object
            IC = IndentCompression(CC, thisCompDf, thisExpDf, i, i_tsDf)
            CC.listIndent.append(IC)
            
            #### 3.4 State if i-th compression is valid for analysis
            doThisCompAnalysis = IC.validateForAnalysis()
            
        except:
            IC = BadIndentCompression(CC, thisCompDf, thisExpDf, i)
            CC.listIndent.append(IC)
            
            #### 3.4 State if i-th compression is valid for analysis
            doThisCompAnalysis = False


        if doThisCompAnalysis:
            
            #### 3.5 Inside i-th compression, delimit the compression and relaxation phases            
            IC.refineStartStop()
            
            #### 3.6 Find the best H0
            IC.computeH0(method = fitSettings['methods_H0'], zone = fitSettings['zones_H0'])
            IC.setBestH0(method = method_bestH0, zone = zone_bestH0)

            
            #### 3.7 Fit with Chadwick model of the force-thickness curve
            if fitSettings['doChadwickFit']:
                for m in fitSettings['ChadwickFitMethods']:
                    if m == 'Full':
                        IC.fitFH_Chadwick(fitValidationSettings, method = m)
                    else:
                        if m.startswith('f'):
                            # try:
                            mask = ufun.strToMask(IC.fCompr, m)
                            IC.fitFH_Chadwick(fitValidationSettings, method = m, mask = mask)
                            # except:
                            #     pass
            
            #### 3.8 Fit with Van Wyk model of the force-thickness curve
            if fitSettings['doVWCFit']:
                for m in fitSettings['VWCFitMethods']:
                    if m == 'Full':
                        IC.fitFH_VWC(fitValidationSettings, method = m)
                    else:
                        if m.startswith('f'):
                            # try:
                            mask = ufun.strToMask(IC.fCompr, m)
                            IC.fitFH_VWC(fitValidationSettings, method = m, mask = mask)
                            # except:
                            #     pass
                        
            #### 3.9 Fit with Dimitiradis model of the force-thickness curve            
            if fitSettings['doDimitriadisFit']:
                for m in fitSettings['DimitriadisFitMethods']:
                    if m == 'Full':
                        IC.fitFH_Dimitriadis(fitValidationSettings, method = m)
                    else:
                        if m.startswith('f'):
                            # try:
                            mask = ufun.strToMask(IC.fCompr, m)
                            IC.fitFH_Dimitriadis(fitValidationSettings, method = m, mask = mask)
                            # except:
                            #     pass
                        if m == 'Valid':
                            valid, mask, dimi_Hmin = IC.find_dimi_range()
                            if valid == False:
                                mask = (mask*0).astype(bool)
                            IC.fitFH_Dimitriadis(fitValidationSettings, method = m, mask = mask)
            

            #### 3.11 Compute stress and strain based on the best H0
            IC.computeStressStrain(method = 'Chadwick')
            
            #### 3.11.1 Compute the contact radius and the 'Chadwick Ratio' = a/h
            IC.computeContactRadius(method = 'Chadwick')
            

            #### 3.11.2 IN DEV : Re-Compute the best H0 
            # IC.computeH0(method = 'Chadwick', zone = 'ratio_2-2.5')
            # IC.computeH0(method = 'Chadwick', zone = 'ratio_2-3')

            #### 3.11.2 Re-Compute the best H0
            # try:
            #     IC.computeH0(method = 'Chadwick', zone = 'ratio_2-2.5')
            # except:
            #     pass
            # try:
            #     IC.computeH0(method = 'Chadwick', zone = 'ratio_2-3')
            # except:
            #     pass

            
            #### 3.12 Local fits of stress-strain curves
            
            #### 3.12.1 Local fits based on stress regions
            if fitSettings['doStressRegionFits']:
                for jj in range(len(halfWidths_StressFits)):
                    for ii in range(len(centers_StressFits)):
                        C, HW = centers_StressFits[ii], halfWidths_StressFits[jj]
                        validRange = ((C-HW) > 0)
                        if validRange:
                            IC.fitSS_stressRegion(C, HW, fitValidationSettings)
           
            #### 3.12.2 Local fits based on sliding gaussian weights based on stress values
            if fitSettings['doStressGaussianFits']:
                for jj in range(len(halfWidths_StressFits)):
                    for ii in range(len(centers_StressFits)):
                        C, HW = centers_StressFits[ii], halfWidths_StressFits[jj]
                        validRange = ((C-HW) > 0)
                        if validRange:
                            IC.fitSS_stressGaussian(C, HW, fitValidationSettings)
            
            #### 3.12.3 Local fits based on fixed number of points
            if fitSettings['doNPointsFits']:
                nbPtsTotal = len(IC.stressCompr)
                iStart = 0
                iStop = iStart + nbPtsFit
                
                while iStop < nbPtsTotal:
                    mask = np.array([((i >= iStart) and (i < iStop)) for i in range(nbPtsTotal)])
                    IC.fitSS_nPoints(mask, fitValidationSettings)
                    
                    iStart = iStop - overlapFit
                    iStop = iStart + nbPtsFit
            
            #### 3.12.3 Local fits based on fixed number of points
            if fitSettings['doLogFits']:
                nbPtsTotal = len(IC.stressCompr)
                iStart = 0
                iStop = iStart + nbPtsFitLog
                
                while iStop < nbPtsTotal:
                    mask = np.array([((i >= iStart) and (i < iStop)) for i in range(nbPtsTotal)])
                    IC.fitSS_Log(mask, fitValidationSettings)
                    
                    iStart = iStop - overlapFitLog
                    iStop = iStart + nbPtsFitLog
            
            #### 3.12.4 Convert all dictFits into DataFrame that can be concatenated and exported after.
            #### 3.12.4 NEW TEST Local fits based on sliding gaussian weights based on strain values
            if fitSettings['doStrainGaussianFits']:
                for jj in range(len(halfWidths_StrainFits)):
                    for ii in range(len(centers_StrainFits)):
                        C, HW = centers_StrainFits[ii], halfWidths_StrainFits[jj]
                        validRange = ((C-HW) > 0)
                        if validRange:
                            IC.fitSS_strainGaussian(C, HW, fitValidationSettings)
                            
            #### 3.12.4 Convert all dictFits into DataFrame that can be concatenated and exported after.
            #### 3.12.4 NEW TEST Local fits based on sliding gaussian weights based on strain values
            if fitSettings['do3partsFits']:
                IC.fitSS_3parts(fitValidationSettings)
                    
            #### 3.12.5 Convert all dictFits into DataFrame that can be concatenated and exported after.
            IC.dictFits_To_DataFrame(fitSettings)
            
            
            #### 3.13 IN DEVELOPMENT - Trying to get a smoothed representation of the stress-strain 
           
            # IC.fitSS_polynomial()
            # IC.fitSS_smooth()
    
    #### Plots
    
    # Tests
    # fig1, axes1 = CC.plot_Timeseries(plotSettings)
    # fig2, axes2 = CC.plot_FH(plotSettings)
    # fig31, axes31 = CC.plot_SS(plotSettings, fitType='stressRegion')
    # fig41, axes41 = CC.plot_KS(plotSettings, fitType='stressRegion')
    # fig32, axes32 = CC.plot_SS(plotSettings, fitType='stressGaussian')
    # fig42, axes42 = CC.plot_KS(plotSettings, fitType='stressGaussian')
    # fig33, axes33 = CC.plot_SS(plotSettings, fitType='nPoints')
    # fig43, axes43 = CC.plot_KS(plotSettings, fitType='nPoints')
    # fig5, axes5 = CC.plot_KS_smooth()
    # if SHOW:
    #     plt.show()
    # else:
    #     plt.close('all')
    
    # print(CC.listIndent[0].dictFitFH_Chadwick)
    if PLOT:        
        CC.plot_and_save(plotSettings, dpi = 150, figSubDir = 'MecaAnalysis_allCells')
        
        if SHOW:
            plt.show()
        else:
            plt.close('all')
    
    #### Results
    
    CC.exportTimeseriesWithStressStrain()
    
    CC.make_mainResults(fitSettings)
    CC.make_localFitsResults(fitSettings)
    df_H0 = CC.getH0Df()
    
    res = {'results_main' : CC.df_mainResults,
            'results_stressRegions' : CC.df_stressRegions,
            'results_stressGaussian' : CC.df_stressGaussian,
            'results_nPoints' : CC.df_nPoints,
            'results_Log' : CC.df_log,
            'results_H0' : df_H0,
            'results_strainGaussian' : CC.df_strainGaussian,
            'results_3parts' : CC.df_3parts,
            }
    
    print(gs.GREEN + 'T = {:.3f}'.format(time.time() - top) + gs.NORMAL)
    warnings.filterwarnings('default')
    
    print(gs.ORANGE + 'END' + gs.NORMAL)
    
    return(res)


def TimeSeries_to_CompList(f, tsDf, expDf, taskName = '', PLOT = False, SHOW = False, 
                           fitSettings = {}, fitValidationSettings = {}, plotSettings = {}):
    """
    

    Parameters
    ----------
    f : string
        Name of the analysed file.
    tsDf : pandas DataFrame
        DataFrame containing the Time Series data (T, B, F, D3, etc).
    expDf : pandas DataFrame
        DataFrame containing the experimental conditions.
    taskName : string, optional
        Used as folder name to save the plots. The default is ''.
    PLOT : boolean, optional
        Make the plots or not. The default is False.
    SHOW : boolean, optional
        Show the plots or not. The default is False.
    fitSettings : dictionary, optional
        Modify default fit settings. The default is {} (no modifications).
    fitValidationSettings : dictionary, optional
        Modify default fit validation settings. The default is {} (no modifications).
    plotSettings : dictionary, optional
        Modify default fit plots settings. The default is {} (no modifications).

    Returns
    -------
    res : dictionary
        Contains the many result pandas DataFrame computed.

    """
    # top = time.time()
    
    print(gs.BLUE + f + gs.NORMAL, end = ' ... ')
    plt.ioff()
    warnings.filterwarnings('ignore')
    
    
    #### 0. Settings
    #### 0.1 Fits settings
    fitSettings = ufun.updateDefaultSettingsDict(fitSettings, 
                                                 DEFAULT_fitSettings)
    

    
    method_bestH0 = fitSettings['method_bestH0']
    # zone_bestH0 = fitSettings['zone_bestH0']
    
    # centers_StressFits = fitSettings['centers_StressFits']
    # halfWidths_StressFits = fitSettings['halfWidths_StressFits']
    # nbPtsFit = fitSettings['nbPtsFit']
    # overlapFit = fitSettings['overlapFit']
    # nbPtsFitLog = fitSettings['nbPtsFitLog']
    # overlapFitLog = fitSettings['overlapFitLog']
    
    # centers_StrainFits = fitSettings['centers_StrainFits']
    # halfWidths_StrainFits = fitSettings['halfWidths_StrainFits']
    
    #### 0.2 Options for fits validation
    fitValidationSettings = ufun.updateDefaultSettingsDict(fitValidationSettings, 
                                                           DEFAULT_fitValidationSettings)
    
    str_crit = 'nbPts>{:.0f} - R2>{:.2f} - Chi2<{:.1f}'.format(fitValidationSettings['crit_nbPts'], 
                                                               fitValidationSettings['crit_R2'], 
                                                               fitValidationSettings['crit_Chi2'])
    fitValidationSettings['str'] = str_crit
    fitValidationSettings
    
    
    #### 0.3 Plots
    plotSettings = ufun.updateDefaultSettingsDict(plotSettings, 
                                                  DEFAULT_plotSettings)
    plotSettings['subfolder_suffix'] = taskName
    plotSettings['plotPoints'] = str(fitSettings['nbPtsFit']) \
                                 + '_' + str(fitSettings['overlapFit'])
    plotSettings['plotLog'] = str(fitSettings['nbPtsFitLog']) \
                                 + '_' + str(fitSettings['overlapFitLog'])


    
    #### 1. Import experimental infos
    tsDf.dx, tsDf.dy, tsDf.dz, tsDf.D2, tsDf.D3 = tsDf.dx*1000, tsDf.dy*1000, tsDf.dz*1000, tsDf.D2*1000, tsDf.D3*1000
    thisManipID = ufun.findInfosInFileName(f, 'manipID')
    thisExpDf = expDf.loc[expDf['manipID'] == thisManipID]
    Ncomp = max(tsDf['idxAnalysis'])
    cellID = ufun.findInfosInFileName(f, 'cellID')
    
    #### 2. Create CellCompression object
    CC = CellCompression(cellID, tsDf, thisExpDf, f)
    CC.method_bestH0 = method_bestH0
    
    #### 2.1 Correct jumps
    for i in range(Ncomp):
        CC.correctJumpForCompression(i)

    # idxAnalysisVals = tsDf['idxAnalysis'].unique()
    # idxComps = [i-1 for i in range(1, Ncomp+1) if i in idxAnalysisVals]    
    
    Id_comps, Comps = [], []
    
    #### 3. Start looping over indents
    for i in range(Ncomp):
            
        #### 3.2 Segment the i-th compression
        maskComp = CC.getMaskForCompression(i, task = 'compression')
        thisCompDf = tsDf.loc[maskComp,:]
        i_tsDf = ufun.findFirst(1, maskComp)
        
        try:
            #### 3.3 Create IndentCompression object
            IC = IndentCompression(CC, thisCompDf, thisExpDf, i, i_tsDf)
            CC.listIndent.append(IC)
            
            #### 3.4 State if i-th compression is valid for analysis
            doThisCompAnalysis = IC.validateForAnalysis()
            
        except:
            IC = BadIndentCompression(CC, thisCompDf, thisExpDf, i)
            CC.listIndent.append(IC)
            
            #### 3.4 State if i-th compression is valid for analysis
            doThisCompAnalysis = False

        if doThisCompAnalysis:
            #### 3.5 Inside i-th compression, delimit the compression and relaxation phases            
            IC.refineStartStop()            
            
            #### 3.7 Fit with Chadwick model of the force-thickness curve
            IC.fitFH_Chadwick(fitValidationSettings, method = 'Full')
            
            #### Export            
            if IC.dictFitFH_Chadwick['Full']['error'] == False:
                Id_comps.append([IC.cellID, IC.i_indent, IC.DIAMETER])
                Comps.append([IC.hCompr, IC.fCompr])
                
    # Id_comps, Comps = np.array(Id_comps), np.array(Comps)
    return(Id_comps, Comps)


# %%%% Simple wrapper
# Simple script to call just the analysis on 1 timeseries file

# path = cp.DirDataTimeseries
# # path = cp.DirCloudTimeseries
# ld = os.listdir(path)

# expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)

# f = '23-02-23_M1_P1_C7_L40_disc20um_PY.csv'

# stressCenters = [ii for ii in range(50, 1550, 25)]
# stressHalfWidths = [25]

# plot_stressCenters = stressCenters
# plot_stressHalfWidth = stressHalfWidths[0]

# fitSettings = {# H0
#                 'methods_H0':['Chadwick', 'Dimitriadis'],
#                 'zones_H0':['%f_5', '%f_10', '%f_20'],
#                 'method_bestH0':'Chadwick',
#                 'zone_bestH0':'%f_10',
#                 # Stress regions
#                 'doStressRegionFits' : False,
#                 'doStressGaussianFits' : True,
#                 'centers_StressFits' : stressCenters,
#                 'halfWidths_StressFits' : stressHalfWidths,
#                 # Nb point
#                 'doNPointsFits' : False,
#                 'nbPtsFit' : 17,
#                 'overlapFit' : 9
#                 }

# plotSettings = {# ON/OFF switchs plot by plot
#                 'FH(t)':True,
#                 'F(H)':True,
#                 'S(e)_stressRegion':False,
#                 'K(S)_stressRegion':False,
#                 'S(e)_stressGaussian':True,
#                 'K(S)_stressGaussian':True,
#                 'S(e)_nPoints':False,
#                 'K(S)_nPoints':False,
#                 'S(e)_strainGaussian':True, # NEW
#                 'K(S)_strainGaussian':True, # NEW
#                 # Stress
#                 'plotStressCenters':plot_stressCenters,
#                 'plotStressHW':plot_stressHalfWidth,
#                 # Strain
#                 # 'plotStrainCenters':DEFAULT_plot_strainCenters,
#                 'plotStrainHW':DEFAULT_plot_strainHalfWidth,
#                 }

# def simpleWrapper(f, expDf):
#     tsDf = getCellTimeSeriesData(f, fromCloud = False)
#     # res = tsDf
#     res = analyseTimeSeries_meca(f, tsDf, expDf, PLOT = True, SHOW = False,
#                                   fitSettings = fitSettings,
#                                   plotSettings = plotSettings)
#     return(res)
    
# res = simpleWrapper(f, expDf)


# %%%% Complex wrapper

def buildDf_meca(list_mecaFiles, task, expDf, PLOT=False, SHOW = False, **kwargs):
    """
    Subfunction of computeGlobalTable_meca
    Create the dictionnary that will be converted in a pandas table in the end.
    """
    #### 0. Unwrap the kwargs
    fitSettings, fitValidationSettings, plotSettings = {}, {}, {}
    if 'fitSettings' in kwargs:
        fitSettings = kwargs['fitSettings']
    if 'fitValidationSettings' in kwargs:
        fitValidationSettings = kwargs['fitValidationSettings']
    if 'plotSettings' in kwargs:
        plotSettings = kwargs['plotSettings']
    
    if 'fitsSubDir' in kwargs:
        fitsSubDir = kwargs['fitsSubDir']
    else:
        fitsSubDir = ''
    
    list_resultDf = []
    
    for f in list_mecaFiles:
        tS_DataFilePath = os.path.join(cp.DirDataTimeseries, f)
        current_tsDf = pd.read_csv(tS_DataFilePath, sep = ';')
        
        #### Main subfunction
        res_all = analyseTimeSeries_meca(f, current_tsDf, expDf, taskName = task, 
                                         PLOT = PLOT, SHOW = SHOW,
                                         fitSettings = fitSettings, 
                                         fitValidationSettings = fitValidationSettings, 
                                         plotSettings = plotSettings)
        
        for k in res_all.keys(): 
            # Go through all types of results in res_all: 
            #     'results_main', 'results_stressRegions', 'results_stressGaussian', 
            #     'results_nPoints', 'results_H0', 
            #     'results_strainGaussian', etc.
            
            df = res_all[k]
            if k == 'results_main':
                #### The main results data for a cell are grabbed here!
                res_main = df
                
            else: # Other result dataframe : H0 detection or local fit...
                
                if df.size > 0:
                    #### Single cell data are saved here!
                    cellID = ufun.findInfosInFileName(f, 'cellID')
                    fileName = cellID + '_' + k + '.csv' # Example: '22-05-03_M4_P1_C17_results_nPoints.csv'
                    # print(fileName)
                    if fitsSubDir == '':
                        df_path = os.path.join(cp.DirDataAnalysisFits, fileName)
                    else:
                        subdir_path = os.path.join(cp.DirDataAnalysisFits, fitsSubDir)
                        ufun.softMkdir(subdir_path)
                       
                        df_path = os.path.join(subdir_path, fileName)
                        
                        
                    df.to_csv(df_path, sep=';', index=False)
                    if cp.CloudSaving != '':
                        cloudpath = os.path.join(cp.DirCloudAnalysisFits, fileName)
                        df.to_csv(cloudpath, sep=';', index=False)
                    
        list_resultDf.append(res_main) # Append res_main to list for concatenation

    mecaDf = pd.concat(list_resultDf) # Concatenation at the end of the loop
    return(mecaDf)


def updateUiDf_meca(ui_fileSuffix, mecaDf):
    """
    

    Parameters
    ----------
    ui_fileSuffix : TYPE
        DESCRIPTION.
    mecaDf : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    listColumnsUI = ['date','cellName','cellID','manipID','compNum',
                     'UI_Valid','UI_Comments']
    
    listDates = mecaDf['date'].unique()
    
    for date in listDates:
        ui_fileName = str(date) + '_' + str(ui_fileSuffix)
        
        try:
            savePath = os.path.join(cp.DirDataAnalysisUMS, (ui_fileName + '.csv'))
            uiDf = pd.read_csv(savePath, sep=None, engine='python')
            fromScratch = False
            print(gs.GREEN + date + ' : imported existing UI table' + gs.NORMAL)
            
        except:
            print(gs.DARKGREEN + str(date) + ' : no existing UI table found' + gs.NORMAL)
            fromScratch = True
    
        new_uiDf = mecaDf[mecaDf['date'] == date][listColumnsUI[:5]]
        if not fromScratch:
            existingCellId = uiDf['cellID'].values
            new_uiDf = new_uiDf.loc[new_uiDf['cellID'].apply(lambda x : x not in existingCellId), :]
        
        nrows = new_uiDf.shape[0]
        new_uiDf['UI_Valid'] = np.ones(nrows, dtype = bool)
        new_uiDf['UI_Comments'] = np.array(['' for i in range(nrows)])
        
        if not fromScratch:
            new_uiDf = pd.concat([uiDf, new_uiDf], axis = 0, ignore_index=True)
            
        savePath = os.path.join(cp.DirDataAnalysisUMS, (ui_fileName + '.csv'))
        new_uiDf.sort_values(by=['cellID', 'compNum'], inplace = True)
        new_uiDf.to_csv(savePath, sep='\t', index = False)
        
        if cp.CloudSaving != '':
            CloudTimeSeriesFilePath = os.path.join(cp.DirCloudAnalysisUMS, (ui_fileName + '.csv'))
            new_uiDf.to_csv(CloudTimeSeriesFilePath, sep = '\t', index=False)



def computeGlobalTable_meca(mode = 'fromScratch', task = 'all', fileName = 'MecaData', 
                            save = False, PLOT = False, source = 'Python',
                            **kwargs):
    """
    Main function of the complex wrapper of the mechanics analysis.

    Parameters
    ----------
    * mode : 'fromScratch' or 'updateExisting', default is 'fromScratch'.
        - 'fromScratch' will analyse all the time series data files and construct 
          a new GlobalTable from them regardless of the existing GlobalTable.
        - 'updateExisting' will open the existing GlobalTable and determine which 
          of the time series data files are new ones, and will append 
          the existing GlobalTable with the data analysed from those new files.
          
    * task : 'all' (default) or a string containing a file prefix, or several prefixes separated by ' & '.
        - 'all' will include in the analysis all the files that correspond to the chosen 'mode'.
        - giving file prefixes will include in the analysis all the files that start with this prefix 
          and correspond to the chosen 'mode'. Examples: task = '22-05-03' or task = '22-05-03_M1 & 22-05-04_M2 & 22-05-05'.
          
    * fileName : string
        The name of the table that will be created. If mode='updateExisting', it is this table that will be updated.
    * save : bool
        Save the final table or not.
    * PLOT : bool
        Make and save the plots or not.
    * source : string
        'Matlab' or 'Python', default is 'Python'.
    * fitsSubDir : string, optionnal
        A subdirectory to save the local fit files.
    * **kwargs : setting dictionaries.

    Returns
    -------
    mecaDf : pd.DataFrame
        The table containing all the analysis results
        
    Note
    -------
    Running this function will result in many sub-tables (local fits) and/or plots
    to be created and saved without being explicitly returned. Check the directories:
    cp.DirDataAnalysis, cp.DirDataAnalysisFits, cp.DirDataFig, cp.DirDataAnalysisUMS, cp.DirDataTimeseriesStressStrain.

    """

    top = time.time()
    
    ui_fileSuffix = 'UserManualSelection_MecaData'
    
    #### 1. Initialization
    # 1.1 Get the experimental dataframe
    expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
    # expDf_JV = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = '_JV')
    # expDf = pd.concat([expDf, expDf_JV])
    
    
    # 1.2 Get the list of all meca files    
    suffixPython = '_PY'
    if source == 'Matlab':
        list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and (('R40' in f) or ('R80' in f) or ('L40' in f) or ('L50' in f)) and not (suffixPython in f))]
        
    elif source == 'Python':
        list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                      if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                      and (('R40' in f) or ('R80' in f) or ('L40' in f) or ('L50' in f) or ('repeats' in f) or ('mT' in f)) and (suffixPython in f))]
    
    #### 2. Get the existing table if necessary
    imported_mecaDf = False
    
    # 2.1
    if mode == 'fromScratch':
        pass

    # 2.2
    elif mode == 'updateExisting':
        try:
            imported_mecaDf = True
            savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
            existing_mecaDf = pd.read_csv(savePath, sep=';')
            
        except:
            print('No existing table found')
    # 2.3
    else:
        pass
        # existing_mecaDf = buildDf_meca([], 'fromScratch', expDf, PLOT=False)
    
    #### 3. Select the files to analyse from the list according to the task
    list_taskMecaFiles = []
    # print(list_mecaFiles)
    # 3.1
    if task == 'all':
        list_taskMecaFiles = list_mecaFiles
    # 3.2
    else:
        task_list = task.split(' & ')
        for f in list_mecaFiles:
            currentCellID = ufun.findInfosInFileName(f, 'cellID')
            for t in task_list:
                if t in currentCellID:
                    list_taskMecaFiles.append(f)
                    break
    # print(list_taskMecaFiles)
    list_selectedMecaFiles = []
    # 3.3
    if mode == 'fromScratch':
        list_selectedMecaFiles = list_taskMecaFiles
        
    # 3.4
    elif mode == 'updateExisting':
        for f in list_taskMecaFiles:
            currentCellID = ufun.findInfosInFileName(f, 'cellID')
            if currentCellID not in existing_mecaDf.cellID.values:
                list_selectedMecaFiles.append(f)
                
    # Exclude the ones that are not in expDf
    listExcluded = []
    listManips = expDf['manipID'].values
    for i in range(len(list_selectedMecaFiles)): 
        f = list_selectedMecaFiles[i]
        manipID = ufun.findInfosInFileName(f, 'manipID')
        if not manipID in listManips:
            listExcluded.append(f)
            
    for f in listExcluded:
        list_selectedMecaFiles.remove(f)
            
    if len(listExcluded) > 0:
        textExcluded = 'The following files were excluded from analysis\n'
        textExcluded += 'because no matching experimental data was found:'
        print(gs.ORANGE + textExcluded)
        for f in listExcluded:
            print(f)
        print(gs.NORMAL)
    
    # print(list_mecaFiles)
    # print(list_taskMecaFiles)
    # print(list_selectedMecaFiles)
                
    #### 4. Run the analysis on the files, by blocks of 10
    listMecaDf = []
    Nfiles = len(list_selectedMecaFiles)
    
    if imported_mecaDf:
        listMecaDf.append(existing_mecaDf)
    
    # print(list_selectedMecaFiles)
    if Nfiles > 0:
        
        for i in range(0, Nfiles, 10):
            i_start, i_stop = i, min(i+10, Nfiles)
            bloc_selectedMecaFiles = list_selectedMecaFiles[i_start:i_stop]
            
            #### Main subfunction
            new_mecaDf = buildDf_meca(bloc_selectedMecaFiles, task, expDf, PLOT, **kwargs)
            listMecaDf.append(new_mecaDf) 
            mecaDf = pd.concat(listMecaDf)
            if save:
                saveName = fileName + '.csv'
                savePath = os.path.join(cp.DirDataAnalysis, saveName)
                mecaDf.to_csv(savePath, sep=';', index = False)
                if cp.CloudSaving != '':
                    cloudSavePath = os.path.join(cp.DirCloudAnalysis, saveName)
                    mecaDf.to_csv(cloudSavePath, sep=';', index = False)
                textSave = 'Intermediate save {:.0f}/{:.0f} successful !'.format((i+10)//10, ((Nfiles-1)//10)+1)
                print(gs.CYAN + textSave + gs.NORMAL)
    else:
        mecaDf = pd.concat(listMecaDf)
        textInfo = 'Table already up to date!'
        print(gs.CYAN + textInfo + gs.NORMAL)
        save = False
         
            
    #### 5. Final save of the main results
    if save:
        saveName = fileName + '.csv'
        savePath = os.path.join(cp.DirDataAnalysis, saveName)
        mecaDf.to_csv(savePath, sep=';', index = False)
        if cp.CloudSaving != '':
            cloudSavePath = os.path.join(cp.DirCloudAnalysis, saveName)
            mecaDf.to_csv(cloudSavePath, sep=';', index = False)
        print(gs.BLUE + 'Final save successful !' + gs.NORMAL)
    
    updateUiDf_meca(ui_fileSuffix, mecaDf)
    
    duration = time.time() - top
    print(gs.DARKGREEN + 'Total time: {:.0f}s'.format(duration) + gs.NORMAL)
    
    return(mecaDf)
            

    
def getGlobalTable_meca(fileName):
    try:
        savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
        mecaDf = pd.read_csv(savePath, sep=';')
        print('Extracted a table with ' + str(mecaDf.shape[0]) + ' lines and ' + str(mecaDf.shape[1]) + ' columns.')
    except:
        print('No existing table found')
        
    for c in mecaDf.columns:
        if 'Unnamed' in c:
            mecaDf = mecaDf.drop([c], axis=1)
        # if 'K_CIW_' in c:    
        #     mecaDf[c].apply(lambda x : x.strip('][').split(', ')).apply(lambda x : [float(x[0]), float(x[1])])
    
    if 'ExpDay' in mecaDf.columns:
        dateExemple = mecaDf.loc[mecaDf.index[0],'ExpDay']
        if not ('manipID' in mecaDf.columns):
            mecaDf['manipID'] = mecaDf['ExpDay'] + '_' + mecaDf['CellID'].apply(lambda x: x.split('_')[0])
            
    elif 'date' in mecaDf.columns:
        dateExemple = mecaDf.loc[mecaDf.index[0],'date']
        if re.match(ufun.dateFormatExcel, dateExemple):
            print('bad date')
        
    if not ('manipID' in mecaDf.columns):
        mecaDf['manipID'] = mecaDf['date'] + '_' + mecaDf['cellName'].apply(lambda x: x.split('_')[0])

        
    return(mecaDf)



def getCompressions(task = 'all', **kwargs):
    """
    
    """

    top = time.time()
    
    # mode = 'fromScratch'
    # fileName = 'MecaData'
    # source = 'Python'
    # save = False
    # PLOT = False
    
    # ui_fileSuffix = 'UserManualSelection_MecaData'
    
    #### 1. Initialization
    # 1.1 Get the experimental dataframe
    expDf = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = cp.suffix)
    # expDf_JV = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = '_JV')
    # expDf = pd.concat([expDf, expDf_JV])
    
    
    # 1.2 Get the list of all meca files    
    suffixPython = '_PY'
    # if source == 'Matlab':
    #     list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
    #                   if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
    #                   and (('R40' in f) or ('R80' in f) or ('L40' in f) or ('L50' in f)) and not (suffixPython in f))]
        
    # elif source == 'Python':
    list_mecaFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                  if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv") \
                  and (('R40' in f) or ('R80' in f) or ('L40' in f) or ('L50' in f) or ('repeats' in f) or ('mT' in f)) and (suffixPython in f))]
    
    #### 2. Get the existing table if necessary
    # imported_mecaDf = False
    
    # # 2.1
    # if mode == 'fromScratch':
    #     pass

    # # 2.2
    # elif mode == 'updateExisting':
    #     try:
    #         imported_mecaDf = True
    #         savePath = os.path.join(cp.DirDataAnalysis, (fileName + '.csv'))
    #         existing_mecaDf = pd.read_csv(savePath, sep=';')
            
    #     except:
    #         print('No existing table found')
    # # 2.3
    # else:
    #     pass
    #     # existing_mecaDf = buildDf_meca([], 'fromScratch', expDf, PLOT=False)
    
    #### 3. Select the files to analyse from the list according to the task
    list_taskMecaFiles = []
    # print(list_mecaFiles)
    # 3.1
    if task == 'all':
        list_taskMecaFiles = list_mecaFiles
    # 3.2
    else:
        task_list = task.split(' & ')
        for f in list_mecaFiles:
            currentCellID = ufun.findInfosInFileName(f, 'cellID')
            for t in task_list:
                if t in currentCellID:
                    list_taskMecaFiles.append(f)
                    break
            
    # print(list_taskMecaFiles)
    # print(list_taskMecaFiles)
    list_selectedMecaFiles = list_taskMecaFiles
    # # 3.3
    # if mode == 'fromScratch':
    #     list_selectedMecaFiles = list_taskMecaFiles
        
    # # 3.4
    # elif mode == 'updateExisting':
    #     for f in list_taskMecaFiles:
    #         currentCellID = ufun.findInfosInFileName(f, 'cellID')
    #         if currentCellID not in existing_mecaDf.cellID.values:
    #             list_selectedMecaFiles.append(f)
                
    # Exclude the ones that are not in expDf
    listExcluded = []
    listManips = expDf['manipID'].values
    for i in range(len(list_selectedMecaFiles)): 
        f = list_selectedMecaFiles[i]
        manipID = ufun.findInfosInFileName(f, 'manipID')
        if not manipID in listManips:
            listExcluded.append(f)
            
    for f in listExcluded:
        list_selectedMecaFiles.remove(f)
            
    if len(listExcluded) > 0:
        textExcluded = 'The following files were excluded from analysis\n'
        textExcluded += 'because no matching experimental data was found:'
        print(gs.ORANGE + textExcluded)
        for f in listExcluded:
            print(f)
        print(gs.NORMAL)
    
    # print(list_mecaFiles)
    # print(list_taskMecaFiles)
    # print(list_selectedMecaFiles)
                
    
    #### 4. Run the analysis on the files, by blocks of 10
    # listMecaDf = []
    # Nfiles = len(list_selectedMecaFiles)
    
    #### STEP
    list_mecaFiles = list_selectedMecaFiles
    # mecaDf = buildDf_meca(list_selectedMecaFiles, task, expDf, PLOT, **kwargs)
    
    #### 0. Unwrap the kwargs
    fitSettings, fitValidationSettings, plotSettings = {}, {}, {}
    if 'fitSettings' in kwargs:
        fitSettings = kwargs['fitSettings']
    if 'fitValidationSettings' in kwargs:
        fitValidationSettings = kwargs['fitValidationSettings']
    if 'plotSettings' in kwargs:
        plotSettings = kwargs['plotSettings']
    
    if 'fitsSubDir' in kwargs:
        fitsSubDir = kwargs['fitsSubDir']
    else:
        fitsSubDir = ''
    
    list_Id_comps, list_Comps = [], []
    
    for f in list_mecaFiles:
        tS_DataFilePath = os.path.join(cp.DirDataTimeseries, f)
        current_tsDf = pd.read_csv(tS_DataFilePath, sep = ';')
        
        #### Main subfunction
        Id_comps_f, Comps_f = TimeSeries_to_CompList(f, current_tsDf, expDf, taskName = task, 
                                                     fitSettings = fitSettings, 
                                                     fitValidationSettings = fitValidationSettings, 
                                                     plotSettings = plotSettings)
        list_Id_comps += Id_comps_f
        list_Comps += Comps_f
        
    # Id_comps = np.concat(list_Id_comps)
    # Comps = np.concat(list_Comps)
    Id_comps = list_Id_comps
    Comps = list_Comps
        
    duration = time.time() - top
    print(gs.DARKGREEN + 'Total time: {:.0f}s'.format(duration) + gs.NORMAL)
    
    return(Id_comps, Comps)



# %% (3) General import functions
    
# %%% Main functions

def getAnalysisTable(fileName):
    """
    

    Parameters
    ----------
    fileName : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE

    """
    if not fileName[-4:] == '.csv':
        ext = '.csv'
    else:
        ext = ''
        
    try:
        path = os.path.join(cp.DirDataAnalysis, (fileName + ext))
        df = pd.read_csv(path, sep=r'[;,,]', engine='python')
        print(gs.CYAN + 'Analysis table has ' + str(df.shape[0]) + ' lines and ' + \
              str(df.shape[1]) + ' columns.' + gs.NORMAL)
    except:
        print(gs.BRIGHTRED + 'No analysis table found' + gs.NORMAL)
        return()
        
    for c in df.columns:
        if 'Unnamed' in c:
            df = df.drop([c], axis=1)
            
    if 'CellName' in df.columns and 'CellID' in df.columns:
        shortCellIdColumn = 'CellID'
    elif 'cellName' in df.columns and 'cellID' in df.columns:
        shortCellIdColumn = 'cellName'
    
    
    if 'ExpDay' in df.columns:
        dateColumn = 'ExpDay'
    elif 'date' in df.columns:
        dateColumn = 'date'
        
    # try:
    dateExemple = df.loc[df.index[0],dateColumn]
    df = ufun.correctExcelDatesInDf(df, dateColumn, dateExemple)
    # except:
    #     pass
        # print(gs.ORANGE + 'Problem in date correction' + gs.NORMAL)
        
    try:
        if not ('manipID' in df.columns):
            df['manipID'] = df[dateColumn] + '_' + df[shortCellIdColumn].apply(lambda x: x.split('_')[0])
    except:
        print(gs.ORANGE + 'Could not infer Manip Ids' + gs.NORMAL)
        
    return(df)



def getMergedTable(fileName, DirDataExp = cp.DirRepoExp, suffix = cp.suffix,
                   mergeExpDf = True, mergFluo = False, mergeUMS = True,
                   findSubstrates = True):
    """
    

    Parameters
    ----------
    fileName : TYPE
        DESCRIPTION.
    DirDataExp : TYPE, optional
        DESCRIPTION. The default is cp.DirRepoExp.
    suffix : TYPE, optional
        DESCRIPTION. The default is cp.suffix.
    mergeExpDf : TYPE, optional
        DESCRIPTION. The default is True.
    mergFluo : TYPE, optional
        DESCRIPTION. The default is False.
    mergeUMS : TYPE, optional
        DESCRIPTION. The default is True.
    findSubstrates : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    df : TYPE

    """
    
    df = getAnalysisTable(fileName)
    
    if mergeExpDf:
        expDf = ufun.getExperimentalConditions(DirDataExp, suffix = suffix)
        # expDf_JV = ufun.getExperimentalConditions(cp.DirRepoExp, suffix = '_JV')
        # expDf = pd.concat([expDf, expDf_JV])
        
        df = pd.merge(expDf, df, how="inner", on='manipID', suffixes=("_x", "_y"),
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     copy=True,indicator=False,validate=None,
        )
        
    df = ufun.removeColumnsDuplicate(df)

    
    if mergeUMS:
        if 'ExpDay' in df.columns:
            dateColumn = 'ExpDay'
        elif 'date' in df.columns:
            dateColumn = 'date'
        listDates = df[dateColumn].unique()
        
        listFiles_UMS = os.listdir(cp.DirDataAnalysisUMS)
        print(cp.DirDataAnalysisUMS)
        listFiles_UMS_matching = []
        # listPaths_UMS = [os.path.join(DirDataAnalysisUMS, f) for f in listFiles_UMS]
        
        for f in listFiles_UMS:
            for d in listDates:
                if d in f:
                    listFiles_UMS_matching.append(f)
            
        listPaths_UMS_matching = [os.path.join(cp.DirDataAnalysisUMS, f) for f in listFiles_UMS_matching]
        listDF_UMS_matching = [pd.read_csv(p, sep = '\t') for p in listPaths_UMS_matching]
        
        umsDf = pd.concat(listDF_UMS_matching)
        # f_filterCol = lambda x : x not in ['date', 'cellName', 'manipID']
        # umsCols = umsDf.columns[np.array([f_filterCol(c) for c in umsDf.columns])]   
        
        df = pd.merge(df, umsDf, how="left", on=['cellID', 'compNum'], suffixes=("_x", "_y"),
        #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
        #     copy=True,indicator=False,validate=None,[umsCols]
        )
        
    df = ufun.removeColumnsDuplicate(df)
    
    if findSubstrates and 'substrate' in df.columns:
        vals_substrate = df['substrate'].values
        if 'diverse fibronectin discs' in vals_substrate:
            try:
                cellIDs = df[df['substrate'] == 'diverse fibronectin discs']['cellID'].values
                listFiles = [f for f in os.listdir(cp.DirDataTimeseries) \
                              if (os.path.isfile(os.path.join(cp.DirDataTimeseries, f)) and f.endswith(".csv"))]
                for Id in cellIDs:
                    for f in listFiles:
                        if Id == ufun.findInfosInFileName(f, 'cellID'):
                            thisCellSubstrate = ufun.findInfosInFileName(f, 'substrate')
                            thisCellSubstrate = dictSubstrates[thisCellSubstrate]
                            if not thisCellSubstrate == '':
                                df.loc[df['cellID'] == Id, 'substrate'] = thisCellSubstrate
                print(gs.GREEN  + 'Automatic determination of substrate type SUCCEDED !' + gs.NORMAL)
                
            except:
                print(gs.RED  + 'Automatic determination of substrate type FAILED !' + gs.NORMAL)
        
    df = ufun.removeColumnsDuplicate(df)
    
    print(gs.CYAN + 'Merged table has ' + str(df.shape[0]) + ' lines and ' \
          + str(df.shape[1]) + ' columns.' + gs.NORMAL)
        
    return(df)



def getMatchingFits(mecaDf, fitsSubDir = '', fitType = 'stressGaussian', output = 'df',
                    filter_fitID = None):
    """
    Browse the fits tables in 'cp.DirDataAnalysisFits'
    and return all those that corresponds to cells inside 'mecaDf'.
    
    Parameters
    ----------
    mecaDf : pandas.DataFrame
        A table returned by getMergedTable
    
    fitType = 'stressRegion' | 'stressGaussian' | 'nPoints'
        The type of fit wanted
    
    output = 'list' | 'dict' | 'df'
        The type of output returned
            'list' : list of dataframes
            'dict' : dict of {'cellId:dataframe'} 
            'df' : one concatenated dataframe.
    
    filter_fitID : string, optionnal
        Filter the imported data according to the 'id' column of the tables.
        
    NB: Other filter_### inputs should be added in this function if needed.
      
    Returns
    -------
    res : list, dict or pandas.DataFrame
        The required data in the specified format.

    Examples
    --------
    >>> mecaDf = getMergedTable('Global_MecaData_Py')
    >>> fitsDf = getMatchingFits(mecaDf, fitType = 'nPoints', output = 'df',
                        filter_fitID = None)
    >>> # Return all fits obtained with a specified number of points, in one df.
    >>> fitsDf = getMatchingFits(mecaDf, fitType = 'gaussianStress', output = 'df',
                        filter_fitID = '200_75')
    >>> # Return only fits in the region 200+/-75 Pa obtained with 
    >>> # gaussian stress window and in one df.
    >>> fitsDf = getMatchingFits(mecaDf, fitType = 'regionStress', output = 'df',
                        filter_fitID = '_75')
    >>> # Return only fits in regions of half width 75 Pa, obtained with 
    >>> # discrete stress window, and in one df.
    """
    if fitsSubDir != '':
        src_path = os.path.join(cp.DirDataAnalysisFits, fitsSubDir)
    else:
        src_path = cp.DirDataAnalysisFits
        
    listCellIDs = mecaDf['cellID'].unique()
    listFitResults = os.listdir(src_path)
    
    if filter_fitID != None:
        if filter_fitID.startswith('_'):
            filter_fitID = r'[\.\d]{2,6}' + filter_fitID # r'\d{2,4}'
        if filter_fitID.endswith('_'):
            filter_fitID = filter_fitID + r'[\.\d]{2,6}' # r'\d{2,4}'
    
    if output == 'df' or output == 'list':
        L = []
        for f in listFitResults:
            if fitType in f:
                cellID = ufun.findInfosInFileName(f, 'cellID')
                if cellID in listCellIDs:
                    f_path = os.path.join(src_path, f)
                    df = pd.read_csv(f_path, sep=';')
                    if filter_fitID != None:
                        fltr = df['id'].apply(lambda x : re.match(filter_fitID, x) != None)
                        df = df[fltr]
                    L.append(df)
                    
        if output == 'list':
            res = L
        elif output == 'df':
            res = pd.concat(L, ignore_index=True)
                    
    
    elif output == 'dict':
        res = {}
        for f in listFitResults:
            if fitType in f:
                cellID = ufun.findInfosInFileName(f, 'cellID')
                if cellID in listCellIDs:
                    f_path = os.path.join(src_path, f)
                    df = pd.read_csv(f_path, sep=None, engine='python')
                    if filter_fitID != None:
                        fltr = df['id'].apply(lambda x : re.match(filter_fitID, x) != None)
                        df = df[fltr]
                    res[cellID] = df
                    
    return(res)
    
    


def getFitsInTable(mecaDf, fitsSubDir = '', fitType = 'stressGaussian', filter_fitID = None):
    """
    Merge a mecaDf with the required kind of fit data, by calling getMatchingFits().
    
    Parameters
    ----------
    mecaDf : pandas.DataFrame
        A table returned by getMergedTable
    
    fitType = 'stressRegion' | 'stressGaussian' | 'nPoints'
        The type of fit wanted
    
    filter_fitID : string, optionnal
        Filter the imported data according to the 'id' column of the tables.
      
    Returns
    -------
    mergedDf
        The resulting merged DataFrame
    
    Examples
    --------
    >>> import TrackAnalyser as taka
    >>> data_main = taka.getMergedTable('Global_MecaData_Py')
    >>> # Ex1.
    >>> mecaDf = taka.getFitsInTable(mecaDf_main, fitType = 'nPoints', filter_fitID = None)
    >>> # Merge with all fits obtained with 
    >>> # the nPoints method (specified number of points).
    
    >>> # Ex2.
    >>> mecaDf = taka.getFitsInTable(mecaDf_main, fitType = 'gaussianStress', filter_fitID = '200_75')
    >>> # Return only fits in the region 200+/-75 Pa obtained with 
    >>> # the gaussianStress method (gaussian stress window).
    
    >>> # Ex3.
    >>> mecaDf = taka.getFitsInTable(mecaDf_main, fitType = 'regionStress', filter_fitID = '_75')
    >>> # Return only fits in regions of half width 75 Pa, obtained with 
    >>> # the regionStress method (discrete stress window).

    """
    
    fitsDf = getMatchingFits(mecaDf, fitsSubDir = fitsSubDir, fitType = fitType, filter_fitID = filter_fitID,
                             output = 'df')
    mergeCols = ['cellID', 'compNum']
    rd = {c : 'fit_' + c for c in fitsDf.columns if c not in mergeCols}
    fitsDf = fitsDf.rename(columns = rd) 
    mergedDf = pd.merge(mecaDf, fitsDf, how="left", on=mergeCols, suffixes=("_x", "_y"),
    #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
    #     copy=True,indicator=False,validate=None,
    )
        
    mergedDf = ufun.removeColumnsDuplicate(mergedDf)
    
    return(mergedDf)


def getAllH0InTable(mecaDf):
    """
    Merge a mecaDf with the H0 table, by calling getMatchingFits().
    
    Parameters
    ----------
    mecaDf : pandas.DataFrame
        A table returned by getMergedTable
      
    Returns
    -------
    mergedDf
        The resulting merged DataFrame
    """
    
    fitsDf = getMatchingFits(mecaDf, fitType = 'H0', filter_fitID = None, output = 'df')
    mergeCols = ['cellID', 'compNum']
    rd = {c : 'allH0_' + c for c in fitsDf.columns if c not in mergeCols}
    fitsDf = fitsDf.rename(columns = rd) 
    mergedDf = pd.merge(mecaDf, fitsDf, how="left", on=mergeCols, suffixes=("_x", "_y"),
    #     left_on=None,right_on=None,left_index=False,right_index=False,sort=True,
    #     copy=True,indicator=False,validate=None,
    )
        
    mergedDf = ufun.removeColumnsDuplicate(mergedDf)
    
    return(mergedDf)


def computeWeightedAverage(df, valCol, weightCol, groupCol = 'cellId', Filters = [], weight_method = 'ciw'):
    """
    
    
    Parameters
    ----------
    df : pandas.DataFrame
        A table returned by 
        
    valCol
    
    
    weightCol
    
    
    groupCol
    
    
    method = 'ciw' | 'weight'
        
    
      
    Returns
    -------
    df
        The resulting DataFrame
    """
    
    wAvgCol = valCol + '_wAvg'
    wVarCol = valCol + '_wVar'
    wStdCol = valCol + '_wStd'
    wSteCol = valCol + '_wSte'
    
    if len(Filters) > 0:
        global_filter = Filters[0]
        for fltr in Filters[1:]:
            global_filter = global_filter & fltr
        df = df[global_filter]
    
    # 1. Compute the weights if necessary
    if weight_method == 'ciw':
        ciwCol = weightCol
        weightCol = valCol + '_weight'
        df[weightCol] = (df[valCol]/df[ciwCol])**2
    
    df = df.dropna(subset = [weightCol])
    
    # 2. Group and average
    
    groupColVals = df[groupCol].unique()

    # In the following lines, the weighted average and weighted variance are computed
    # using new columns as intermediates in the computation.
    #
    # Col 'A' = K x Weight --- Used to compute the weighted average.
    # 'K_wAvg' = sum('A')/sum('weight') in each category (group by condCol and 'fit_center')
    #
    # Col 'B' = (K - K_wAvg)**2 --- Used to compute the weighted variance.
    # Col 'C' =  B * Weight     --- Used to compute the weighted variance.
    # 'K_wVar' = sum('C')/sum('weight') in each category (group by condCol and 'fit_center')
    
    # Compute the weighted mean
    df['A'] = df[valCol] * df[weightCol]
    grouped1 = df.groupby(by=[groupCol])
    data_agg = grouped1.agg({'A': ['count', 'sum'], weightCol: 'sum'}).reset_index()
    data_agg.columns = ufun.flattenPandasIndex(data_agg.columns)
    data_agg[wAvgCol] = data_agg['A_sum']/data_agg[weightCol + '_sum']
    data_agg = data_agg.rename(columns = {'A_count' : 'count_wAvg'})
    
    # Compute the weighted std
    df['B'] = df[valCol]
    for co in groupColVals:
        weighted_avg_val = data_agg.loc[(data_agg[groupCol] == co), wAvgCol].values[0]
        index_loc = (df[groupCol] == co)
        col_loc = 'B'
        
        df.loc[index_loc, col_loc] = df.loc[index_loc, valCol] - weighted_avg_val
        df.loc[index_loc, col_loc] = df.loc[index_loc, col_loc] ** 2
            
    df['C'] = df['B'] * df[weightCol]
    grouped2 = df.groupby(by=[groupCol])
    data_agg2 = grouped2.agg({'C': 'sum', weightCol: 'sum'}).reset_index()
    data_agg2[wVarCol] = data_agg2['C']/data_agg2[weightCol]
    data_agg2[wStdCol] = data_agg2[wVarCol]**0.5
    
    
    # Combine all in data_agg
    data_agg[wVarCol] = data_agg2[wVarCol]
    data_agg[wStdCol] = data_agg2[wStdCol]
    data_agg[wSteCol] = data_agg[wStdCol] / data_agg['count_wAvg']**0.5
    
    # data_agg = data_agg.drop(columns = ['A_sum', weightCol + '_sum'])
    data_agg = data_agg.drop(columns = ['A_sum'])
    
    return(data_agg)
    
        
        

  
  
  
  
  
