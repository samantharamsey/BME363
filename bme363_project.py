# -*- coding: utf-8 -*-
'''
Created on Wed Apr 15 14:47:49 2020

@author: Samantha Ramsey
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sci
import pandas as pd
from scipy.integrate import odeint


def make_calculations(mass, data):
    '''
    Estimates the spring stiffness and damping coefficient
    Args:
        mass - mass of object
        data - for tracker run
    Returns:
        decrement - logarithmic decrement
        zeta - damping ratio
        k - spring stiffness
        b - damping coefficient
    '''
    
    peaks = sci.find_peaks(data['x(in)'])
    x1 = data['x(in)'].iloc[peaks[0][0]]
    x2 = data['x(in)'].iloc[peaks[0][1]]
    
    decrement = abs(np.log(x1/x2))
    zeta = abs(decrement/(np.sqrt((2*np.pi)**2 + decrement**2)))
    omega = np.pi/(np.sqrt(1 - zeta**2))
    k = mass*omega**2
    b = 2*zeta*omega*mass
    
    return decrement, zeta, k, b


if __name__ == '__main__':
    
    # constants
    mass1 = 1.3 #kg
    mass2 = 1.9 #kg
    initial_length = 0.15240 #m
    mass1_length   = 0.15748 #m
    mass2_length   = 0.17272 #m
    
    ############################ PART 1: MODELING #############################
    
    # spring stiffness
    slope = (mass2_length - mass1_length)/(mass2 - mass1)
    
#    # determined using hand calculations
#    def eqofm(t):
#        f = np.cos(0.1396*t) + 1
#        return f
#    
#    # find position over time using estimated equation of motion
#    result = []
#    for i in time:
#        r = eqofm(i)
#        result.append(r) 

    time = np.linspace(0.01, 2.02, 67)    
    y0 = mass1_length - initial_length
    
    def system(y, t):
        d2ydt = (mass1*9.81 - slope*y)/mass1
        return d2ydt 
        
    y = odeint(system, y0, time)
    
    # plot stuff
    plt.plot(time, y)  
    
    ########################### PART 2: EXPERIMENT ############################
    
    # files
    filepath = r'C:\Users\saman\Downloads'
    filename1 = r'\tracker1.xlsx'
    filename2 = r'\tracker2.xlsx'
    
    # load into a dataframe
    data1 = pd.read_excel(filepath + filename1)
    data2 = pd.read_excel(filepath + filename2)
    
    # calculate stuff
    dec1, zeta1, k1, b1 = make_calculations(mass1, data1)
    dec2, zeta2, k2, b2 = make_calculations(mass2, data2)
        
    # plot stuff
#    plt.plot(data2['t(s)'], data2['x(in)'])
    
    
    ######################## PART 3: UPDATED MODELING #########################
    
    # determined using hand calculations
    def eqofm2(t):
        f = 0.2*np.cos(20.94395*t)*np.exp(-0.013489*t)
        return f
    
    # find position over time using estimated equation of motion
    time = np.linspace(0.01, 2.02, 67)
    result2 = []
    for i in time:
        r2 = eqofm2(i)
        result2.append(r2)
    
    # plot stuff
#    plt.plot(time, result2)
    
    
    ###################### PART 4: STATISTICAL ANALYSIS #######################
    
    n = len(time)
    RMS = 0
    for i in range(n):
        RMS += np.sqrt((result2[i] - data1['x(in)'][i])**2/n)
    
    
    ################################## PLOTS ##################################
   
    
    plt.legend(['part 1 equation of motion', 
                'mass as a function of time', 
                'part 2 updated equation of motion'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    