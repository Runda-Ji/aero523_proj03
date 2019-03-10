# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:14:11 2018

@author: Runda Ji
"""

import numpy as np
import matplotlib.pyplot as plt

def flux(rho):
    f = rho*(1 - rho);
    return f;

def Godunov_flux(rho_L,rho_R):
    if rho_L <= rho_R:
        dis_L = np.abs(rho_L-0.5);
        dis_R = np.abs(rho_R-0.5);
        if dis_L > dis_R:
            F_min = flux(rho_L);
        else:
            F_min = flux(rho_R);
        F = F_min;        
    else:#rho_L > rho_R
        if rho_R <= 0.5 and rho_L > 0.5:
            F_max = flux(0.5);
        if rho_L <= 0.5:
            F_max = flux(rho_L);
        if rho_R > 0.5:
            F_max = flux(rho_R);
        F = F_max;
    return F;

def single_time_step(rho,nElems,dt):
    global F;
    rho_left_BC = 0.5;
    rho_right_BC = 0;
    #compute the flux
    F = np.zeros(nElems+1);
    #flux on the left boundary of the 0^th cell
    F[0] = Godunov_flux(rho_left_BC,rho[0]);
    for i in range(1,nElems):
        #flux on the left boundary of the i^th cell
        F[i] = Godunov_flux(rho[i-1],rho[i]);
    #flux on the right boundary of the last cell
    F[nElems] = Godunov_flux(rho[nElems-1],rho_right_BC);
    #update the state
    for i in range(0,nElems):
        rho[i] = rho[i] + 0.8*(F[i]-F[i+1]);
    return 0;

def main():
    global node_pos,rho;
    dx = 0.01;
    dt = 0.8*dx;
    x_min = -5.0;
    x_max = 5.0;
    nElems = int((x_max-x_min)/dx);
    #initialization
    rho = np.zeros(nElems);
    mid_pt_pos = np.zeros(nElems);
    node_pos = np.zeros(nElems+1);
    for i in range(0,nElems+1):
        node_pos[i] = x_min + i*dx;
    for i in range(0,nElems):
        mid_pt_pos[i] = 0.5*(node_pos[i]+node_pos[i+1]);
        if mid_pt_pos[i] < -1:
            rho[i] = 0.5;
        if mid_pt_pos[i] >= -1 and mid_pt_pos[i] <= 0:
            rho[i] = 1;
        if mid_pt_pos[i] > 0:
            rho[i] = 0;
    #time iteration
    t_max = 5;
    nTimestep = int(t_max/dt);
    f = plt.figure(figsize=(8,6));
    for i in range (0,nTimestep):
        t = 0 + i*dt;
        single_time_step(rho,nElems,dt);
        if t == 1 or t == 2 or t == 3:
            plt.plot(mid_pt_pos,rho,'-',label='Godunov flux, t = %d s' %t);
    
    for t in range (1,4):    
        for i in range(0,nElems):
            if t <= 2:
                if mid_pt_pos[i] <= -1 - 0.5*t:
                    rho[i] = 0.5;
                if mid_pt_pos[i] > -1 - 0.5*t and mid_pt_pos[i] <= -t:
                    rho[i] = 1;
                if mid_pt_pos[i] > -t and mid_pt_pos[i] <= t:
                    rho[i] = 0.5 - 0.5*mid_pt_pos[i]/t;
                if mid_pt_pos[i] > t:
                    rho[i] = 0;
            else:
                if mid_pt_pos[i] <= -np.sqrt(2*t):
                    rho[i] = 0.5;
                if mid_pt_pos[i] > -np.sqrt(2*t) and mid_pt_pos[i] <= t:
                    rho[i] = 0.5 - 0.5*mid_pt_pos[i]/t;
                if mid_pt_pos[i] > t:
                    rho[i] = 0;
        plt.plot(mid_pt_pos,rho,'--',label='Exact solution, t = %d s' %t); 
            
    plt.xlabel(r'Position, $x$',fontsize = 16);
    plt.ylabel(r'Density, ${\rho}$',fontsize = 16);
    plt.grid();
    plt.legend();
    plt.savefig('density_Godunov_flux.pdf', dpi=150);
    plt.show();
    plt.close(f);
    return 0;

if __name__=="__main__":
    main()