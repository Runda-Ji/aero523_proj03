# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:58:19 2018

@author: Runda Ji
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    global rho;
    Nx = 500;
    x = np.zeros(Nx);
    rho = np.zeros(Nx);
    f = plt.figure(figsize=(8,6));
    for t in range (1,4):    
        for i in range(0,Nx):
            x[i] = 10.0*i/Nx - 5;
            if t <= 2:
                if x[i] <= -1 - 0.5*t:
                    rho[i] = 0.5;
                if x[i] > -1 - 0.5*t and x[i] <= -t:
                    rho[i] = 1;
                if x[i] > -t and x[i] <= t:
                    rho[i] = 0.5 - 0.5*x[i]/t;
                if x[i] > t:
                    rho[i] = 0
            else:
                if x[i] <= -np.sqrt(2*t):
                    rho[i] = 0.5;
                if x[i] > -np.sqrt(2*t) and x[i] <= t:
                    rho[i] = 0.5 - 0.5*x[i]/t;
                if x[i] > t:
                    rho[i] = 0
        plt.plot(x,rho,label='t = %d s' %t);    
    #------------------------------------------------------------
    plt.xlabel(r'Position, $x$',fontsize = 16);
    plt.ylabel(r'Density, ${\rho}$',fontsize = 16);
    plt.grid();
    plt.legend();
    plt.savefig('density_exact.pdf', dpi=150);
    plt.show();
    plt.close(f);
    return 0;

if __name__=="__main__":
    main()