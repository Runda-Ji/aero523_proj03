# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 18:57:29 2018

@author: Runda Ji
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import matplotlib.pyplot as plt

def initialization(Nx,Ny):
    p = np.zeros([Nx+2,Ny+2]);
    F = np.zeros([Nx+2,Ny+2]);
    G = np.zeros([Nx+2,Ny+2]);
    Hx = np.zeros([Nx+1,Ny+1]);
    Hy = np.zeros([Nx+1,Ny+1]);
    u = np.zeros([Nx+3,Ny+2]);
    u_star = np.zeros([Nx+3,Ny+2]);
    v = np.zeros([Nx+2,Ny+3]);
    v_star = np.zeros([Nx+2,Ny+3]);
    mesh = {'Nx':Nx, 'Ny':Ny, 'p':p, 'F':F, 'G':G, 'Hx':Hx, 'Hy':Hy, 'u':u, 'v':v, 'u_star': u_star, 'v_star': v_star};
    return mesh;

def in_out_flow(mesh,U0,h,d):
    Nx = mesh['Nx'];
    Ny = mesh['Ny'];
    #left
    for j in range(int(Ny/2)+1,Ny+1):
        mesh['u'][0,j] = -((j-0.5)*d-h)*((j-0.5)*d-2*h)*4*U0/h**2;
        mesh['u'][1,j] = -((j-0.5)*d-h)*((j-0.5)*d-2*h)*4*U0/h**2;
    for j in range(int(Ny/2)+1,Ny+2):
        mesh['v'][0,j] = 0;
    #right
    for j in range(1,Ny+1):
        mesh['u'][Nx+1,j] = -(j-0.5)*d*((j-0.5)*d-2*h)*0.5*U0/h**2;
        mesh['u'][Nx+2,j] = -(j-0.5)*d*((j-0.5)*d-2*h)*0.5*U0/h**2;
    for j in range(1,Ny+2):
        mesh['v'][Nx+1,j] = 0;
    return 0;

def wall(mesh):
    Nx = mesh['Nx'];
    Ny = mesh['Ny'];
    #left wall u
    for j in range(1,int(Ny/2)+1):
        mesh['u'][0,j] = 0;
        mesh['u'][1,j] = 0;
    #up and low wall v
    for i in range(1,Nx+1):    
        mesh['v'][i,0] = 0;
        mesh['v'][i,1] = 0;
        mesh['v'][i,Ny+1] = 0;
        mesh['v'][i,Ny+2] = 0;  
    #left wall v
    for j in range(1,int(Ny/2)+1):
        mesh['v'][0,j] = -mesh['v'][1,j];
    #up and low wall u
    for i in range(1,Nx+2):
        mesh['u'][i,Ny+1] = -mesh['u'][i,Ny];
        mesh['u'][i,0] = -mesh['u'][i,1];
    return 0;

def calculate_flux(mesh,d,nu):
    Nx = mesh['Nx'];
    Ny = mesh['Ny'];
    #traverse the interior cells
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            u_ave = 0.5*(mesh['u'][i,j] + mesh['u'][i+1,j]);
            if u_ave > 0:
                phi_u = (3*mesh['u'][i+1,j] + 6*mesh['u'][i,j]   - mesh['u'][i-1,j])/8;
            else:
                phi_u = (3*mesh['u'][i,j]   + 6*mesh['u'][i+1,j] - mesh['u'][i+2,j])/8;
            F_convec = u_ave*phi_u;
            #-------------------------------------------------------------------------
            v_ave = 0.5*(mesh['v'][i,j] + mesh['v'][i,j+1]);
            if v_ave > 0:
                phi_v = (3*mesh['v'][i,j+1] + 6*mesh['v'][i,j]   - mesh['v'][i,j-1])/8;
            else:
                phi_v = (3*mesh['v'][i,j]   + 6*mesh['v'][i,j+1] - mesh['v'][i,j+2])/8;
            G_convec = v_ave*phi_v;
            #-------------------------------------------------------------------------
            mesh['F'][i,j] = F_convec - nu/d*(mesh['u'][i+1,j] - mesh['u'][i,j]);
            mesh['G'][i,j] = G_convec - nu/d*(mesh['v'][i,j+1] - mesh['v'][i,j]);
    
    #traverse the nodes
    for i in range(0,Nx+1):
        for j in range(0,Ny+1):
            #For interior nodes
            if (i>0 and i<Nx) and (j>0 and j<Ny):
                v_ave = 0.5*(mesh['v'][i+1,j+1] + mesh['v'][i,j+1]); 
                if v_ave > 0:
                    phi_u = (3*mesh['u'][i+1,j+1] + 6*mesh['u'][i+1,j]   - mesh['u'][i+1,j-1])/8;
                else:
                    phi_u = (3*mesh['u'][i+1,j]   + 6*mesh['u'][i+1,j+1] - mesh['u'][i+1,j+2])/8;
                Hx_convec = v_ave*phi_u;
                #-------------------------------------------------------------------------
                u_ave = 0.5*(mesh['u'][i+1,j+1] + mesh['u'][i+1,j]);
                if u_ave > 0:
                    phi_v = (3*mesh['v'][i+1,j+1] + 6*mesh['v'][i,j+1]   - mesh['v'][i-1,j+1])/8;
                else:
                    phi_v = (3*mesh['v'][i,j+1]   + 6*mesh['v'][i+1,j+1] - mesh['v'][i+2,j+1])/8;
                Hy_convec = u_ave*phi_v;
            #For boundary nodes
            else:
                Hx_convec = 0;
                Hy_convec = 0;
            mesh['Hx'][i,j] = Hx_convec - nu/d*(mesh['u'][i+1,j+1] - mesh['u'][i+1,j]);
            mesh['Hy'][i,j] = Hy_convec - nu/d*(mesh['v'][i+1,j+1] - mesh['v'][i,j+1]);
    return 0;

def compute_frac_velocity(mesh,d,dt):
    #at the boundary the velocity does not change
    Nx = mesh['Nx'];
    Ny = mesh['Ny'];
    #left and right u
    for j in range(1,Ny+1):
        mesh['u_star'][1,j] = mesh['u'][1,j];
        mesh['u_star'][Nx+1,j] = mesh['u'][Nx+1,j];
    #up and low wall v
    for i in range(1,Nx+1):    
        mesh['v_star'][i,1] = mesh['v'][i,1];
        mesh['v_star'][i,Ny+1] = mesh['v'][i,Ny+1];
    #for all interior edges (vertical,u)
    for i in range(2,Nx+1):
        for j in range (1,Ny+1):
            pF_px  = (mesh['F'][i,j]    - mesh['F'][i-1,j])/d;
            pHx_py = (mesh['Hx'][i-1,j] - mesh['Hx'][i-1,j-1])/d;
            mesh['u_star'][i,j] = mesh['u'][i,j] - dt*(pF_px + pHx_py);
    #for all interior edges (horizontal,v)
    for i in range(1,Nx+1):
        for j in range (2,Ny+1):
            pHy_px = (mesh['Hy'][i,j-1] - mesh['Hy'][i-1,j-1])/d;
            pG_py  = (mesh['G'][i,j]    - mesh['G'][i,j-1])/d;
            mesh['v_star'][i,j] = mesh['v'][i,j] - dt*(pHy_px + pG_py);
    return 0;

def construct_matrix(Nx,Ny):
    global A;
    
    NN = Nx*Ny;
    nnz = (Nx-2)*(Ny-2)*5 + 2*(Nx-2)*4 + 2*(Ny-2)*4 + 4*3;
    data = np.zeros(nnz,dtype=np.float);
    irow = np.zeros(nnz,dtype=np.int);
    icol = np.zeros(nnz,dtype=np.int);
    inz = 0;
        
    #fill the matrix
    for iy in range(1,Ny+1):
        for ix in range(1,Nx+1):
            k = (iy-1)*Nx + ix-1;
            #for diag term
            irow[inz] = k; icol[inz] = k;
            if (ix>1 and ix<Nx) and (iy>1 and iy<Ny):
            #if interior cell
                data[inz] = -4;
            else:
                if ((ix>1 and ix<Nx) and (iy == 1 or iy == Ny)) or ((ix == 1 or ix == Nx) and (iy>1 and iy<Ny)):
                #if at boundary
                    data[inz] = -3;
                else:
                #if at corner
                    data[inz] = -2;
            inz = inz + 1;
            #for non-diag term
            if (ix>1):
                irow[inz] = k; icol[inz] = k-1; data[inz]=1; inz = inz + 1;
            if (ix<Nx):
                irow[inz] = k; icol[inz] = k+1; data[inz]=1; inz = inz + 1;
            if (iy>1):
                irow[inz] = k; icol[inz] = k-Nx; data[inz]=1; inz = inz + 1;
            if (iy<Ny):
                irow[inz] = k; icol[inz] = k+Nx; data[inz]=1; inz = inz + 1;
    print(inz-nnz);
    #replace the last equation with p_NxNy=0
    irow[nnz-3] = Nx*Ny-1; icol[nnz-3] = (Nx*Ny-1)-Nx; data[nnz-3] = 0;
    irow[nnz-2] = Nx*Ny-1; icol[nnz-2] = Nx*Ny-2; data[nnz-2] = 0;
    irow[nnz-1] = Nx*Ny-1; icol[nnz-1] = Nx*Ny-1; data[nnz-1] = 1;
    A = sparse.csr_matrix((data,(irow,icol)),shape = (NN,NN));
    return A;

def solve_PPE(A,mesh,d,dt):
    global right_vec;
    Nx = mesh['Nx'];
    Ny = mesh['Ny'];
    right_vec = np.zeros(Nx*Ny);
    for i in range(1,Nx+1):
        for j in range(1,Ny+1):
            k = (j-1)*Nx + (i-1);
            right_vec[k] = d/dt*(mesh['u_star'][i+1,j] - mesh['u_star'][i,j] \
                                +mesh['v_star'][i,j+1] - mesh['v_star'][i,j]);
    #replace the last equation
    right_vec[Nx*Ny-1] = 0;
    Pv = linalg.spsolve(A,right_vec);
    mesh['p'][1:Nx+1,1:Ny+1] = np.reshape(Pv,(Nx,Ny),order='F');
    return 0;

def correct_velocity(mesh,d,dt):
    Nx = mesh['Nx'];
    Ny = mesh['Ny'];
    #for all interior edges (vertical,u)
    for i in range(2,Nx+1):
        for j in range (1,Ny+1):
            mesh['u'][i,j] = mesh['u_star'][i,j] - dt/d*(mesh['p'][i,j] - mesh['p'][i-1,j]);
    #for all interior edges (horizontal,v)
    for i in range(1,Nx+1):
        for j in range (2,Ny+1):
            mesh['v'][i,j] = mesh['v_star'][i,j] - dt/d*(mesh['p'][i,j] - mesh['p'][i,j-1]);
    return 0;

def compute_res(mesh,d):
    Nx = mesh['Nx'];
    Ny = mesh['Ny'];
    SUM = 0;
    #for all interior edges (vertical,u)
    for i in range(2,Nx+1):
        for j in range (1,Ny+1):
            R = d*(mesh['F'][i,j] + mesh['p'][i,j] - mesh['F'][i-1,j] - mesh['p'][i-1,j])\
              + d*(mesh['Hx'][i-1,j] - mesh['Hx'][i-1,j-1]);
            SUM = SUM + np.abs(R);
    #for all interior edges (horizontal,v)
    for i in range(1,Nx+1):
        for j in range (2,Ny+1):
            R = d*(mesh['G'][i,j] + mesh['p'][i,j] - mesh['G'][i,j-1] - mesh['p'][i,j-1])\
              + d*(mesh['Hy'][i,j-1] - mesh['Hy'][i-1,j-1]);
            SUM = SUM + np.abs(R);
    return SUM;

def save_data(mesh,fname):
    np.save('data\\%s_data' %fname, mesh);
    return 0;

def post_processing(mesh,fname):
    global psi,xx,yy;
    #f0 = plt.figure(figsize=(12,2));
    #plt.pcolor(mesh['u'].transpose(),cmap=plt.cm.jet);
    #plt.colorbar();
    Nx = mesh['Nx'];
    Ny = mesh['Ny'];
    h = 1;
    d = 2*h/Ny;
    x = np.linspace(0, Nx*d, Nx+1);
    y = np.linspace(0, Ny*d, Ny+1);
#------------------------------------------------------------------------------
    f1 = plt.figure(figsize=(10,4));
    xx,yy = np.meshgrid(x, y);
    psi = np.zeros([Nx+1,Ny+1]);    
    for i in range(0,Nx+1):
        for j in range(1,Ny+1):
            psi[i,j] = psi[i,j-1] + mesh['u'][i+1,j]*d;
    plt.contour(xx, yy, psi.transpose(), 20,cmap=plt.cm.jet);
    plt.colorbar();
    plt.grid();
    plt.xlabel('$x/h$',fontsize =12);
    plt.ylabel('$y/h$',fontsize =12);
    plt.savefig('figure\\%s_streamline.pdf' %fname,dpi=150);
    plt.close(f1);
#------------------------------------------------------------------------------
    f2 = plt.figure(figsize=(8,6));
    pu_py = np.zeros(Nx+1);
    for i in range(1,Nx+2):
        pu_py[i-1] = 0.5*mesh['u'][i,1]/d;
    plt.plot(x,pu_py);
    plt.grid();
    plt.xlabel('$x/h$',fontsize =12);
    plt.ylabel('$\partial u/\partial y$',fontsize =12);
    plt.savefig('figure\\%s_pu_py.pdf' %fname,dpi=150);
    plt.close(f2);
    return 0;

def main():
    global mesh;
    U0 = 1;
    h = 1;
    Re_list = [200];
    Ny_list = [16,32];
    f_list = [6,8,12];
    for i in range(0,1):
        for j in range(0,2):
            for k in range(0,3):
                Re = Re_list[i];
                Ny = Ny_list[j];
                f = f_list[k];
                Nx = f*Ny;
                d = 2*h/Ny;
                nu = 2*U0*h/(3*Re);
                dt = 0.8*np.minimum((h**2/(4*nu)),(4*nu/U0**2));
                mesh = initialization(Nx,Ny);
                A = construct_matrix(Nx,Ny);
                for t in range(0,1000000):
                    in_out_flow(mesh,U0,h,d);
                    wall(mesh);
                    calculate_flux(mesh,d,nu);
                    compute_frac_velocity(mesh,d,dt);
                    solve_PPE(A,mesh,d,dt);
                    correct_velocity(mesh,d,dt);
                    R = compute_res(mesh,d);
                    if np.remainder(t,100) == 0:
                        print('iter %d Res %f' %(t,R));
                    if R<1e-5:
                        break;
                save_data(mesh,'Re%d_Ny%d_f%d'%(Re,Ny,f));
                post_processing(mesh,'Re%d_Ny%d_f%d'%(Re,Ny,f));
    return 0;

if __name__=="__main__":
    main()