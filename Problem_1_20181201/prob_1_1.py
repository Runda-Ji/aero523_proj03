# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:58:19 2018

@author: Runda Ji
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    t_max = 5;
    nt = 100;
    t = np.zeros(nt);
    xs = np.zeros(nt);
    for i in range(0,nt):
        t[i] = t_max*i/nt;
        if t[i] <= 2:
            xs[i] = -1 - 0.5*t[i];
        else:
            xs[i] = -1*np.sqrt(2*t[i]);
    #------------------------------------------------------------
    nx1 = 25;
    x1 = np.zeros((nt,nx1));
    for i in range(0,nt):
        t[i] = t_max*i/nt;
        for j in range(0,nx1):
            x0 = 3.5*j/nx1 - 3.5;
            x = x0;
            if x < xs[i]:
                x1[i,j] = x;
            else:
                x1[i,j] = None;
    #------------------------------------------------------------
    nx2 = 5;
    x2 = np.zeros((nt,nx2));
    for i in range(0,nt):
        t[i] = t_max*i/nt;
        for j in range(0,nx2):
            x0 = j/nx2 - 1;
            x = x0 - t[i];
            if x > xs[i]:
                x2[i,j] = x;
            else:
                x2[i,j] = None;
    #------------------------------------------------------------
    nx3 = 10;
    x3 = np.zeros((nt,nx3));
    for i in range(0,nt):
        t[i] = t_max*i/nt;
        for j in range(0,nx3):
            x0 = 0;
            a = 2*j/nx3 - 1;
            x = x0 + a*t[i];
            if x > xs[i]:
                x3[i,j] = x;
            else:
                x3[i,j] = None;
    #------------------------------------------------------------
    nx4 = 20;
    x4 = np.zeros((nt,nx4));
    for i in range(0,nt):
        t[i] = t_max*i/nt;
        for j in range(0,nx4):
            x0 = 4*j/nx4;
            x = x0 + t[i];
            x4[i,j] = x;
    #------------------------------------------------------------
    f = plt.figure(figsize=(8,6));
    plt.plot(x1,t,'c-');
    plt.plot(x2,t,'b-');
    plt.plot(x4,t,'g-');
    plt.plot(x3,t,'r-');
    plt.plot(xs,t,'k-',lw=3.0);
    plt.grid();
    plt.ylim((0,3));
    plt.xlim((-3,4));
    plt.xlabel(r'Position, $x$',fontsize = 16);
    plt.ylabel(r'Time, $t$',fontsize = 16);
    plt.savefig('characteristics.pdf', dpi=150);
    plt.show();
    plt.close(f);
    return 0;

if __name__=="__main__":
    main()