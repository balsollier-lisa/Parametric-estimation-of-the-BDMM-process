#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lisabalsollier
"""

"""
You will find in this file all the necessary programs to plot the log likelihood 
and the confidence ellipse of the real data.
This file was used to generate Figure 3 presented in the article.
"""


import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import scipy.optimize as op
from matplotlib.patches import Ellipse



"""we start with informations and a programme that are necessary for the rest of the process"""

intertps=0.14
Airecell=42525
plbirth=(1/4.45)*np.array([3.59,0.47,0.39]) #proportions for the birth of brownian, superdiffusive, subdiffusive Langerin (in this order)


with open('maskc.pickle', 'rb') as handle:
    imgmod = pickle.load(handle) 

"""this program allows to say if a point is in the cell"""
def beincell(x):
    if np.floor(x[0])<=0 or np.floor(x[0])+1>=250:
        return(0)
    if np.floor(x[1])<=0 or np.floor(x[1])+1>=283:
        return(0)
    if np.floor(x[0])+0.5<x[0]:
        j=int(np.floor(x[0]))+1
    else :
        j=int(np.floor(x[0]))
    if np.floor(x[1])+0.5<x[1]:
        i=int(np.floor(x[1]))+1
    else :
        i=int(np.floor(x[1]))
    return(int(imgmod[i][j][0]))


"""the following lines are used to extract what we need from the files 
containing the actual data (called data-Rab11.csv and data-Langerin.csv), 
which must be in the same folder. """
 
def extract_real_data():
    """
    Parameters
    ----------
    None.

    Returns
    -------
    drnx : LIST
        list that contains the coordinates of the Langerin particles at the time of his birth
    dravtnx : LIST
        list that contains list of the coordinates of the Rab11 particles that are present at the time of the births of new Langerin particle
    resfinalrab : LIST
        list that contains at position i a list of the coordinates of the Rab11 particles that are present at the time i
    """
    
    resfinalrab=[]
    f=open ('data-Rab11.csv','r')
    reader=csv.DictReader(f, delimiter=',')
    nframe=np.max([int(row['t (frame)']) for row in reader])
    for i in range(1,nframe+1):
        f=open ('data-Rab11.csv','r')
        reader=csv.DictReader(f, delimiter=',')
        pts2=[]
        for row in reader :
            if int(row['t (frame)'])==i :
                pts2+=[float(row['X (pixel)']), float(row['Y(pixel)']),1,int(row['Motion Type'])]
        resfinalrab+=[pts2]
    
    
    f=open ('data-Rab11.csv','r')
    reader=csv.DictReader(f, delimiter=',')
    nframe=np.max([int(row['t (frame)']) for row in reader])
    drtpsnx=[]
    drnx=[]
    dravtnx=[]
    dravtnx2=[]
    for i in range(2,nframe+1):
        vec1=[]
        vec2=[]
        fl=open ('data-Langerin.csv','r')
        readerl=csv.DictReader(fl, delimiter=',')
        for row in readerl:
            if int(row['t (frame)'])==i:
                vec2+=[ int(row['Track Number'])]
            if int(row['t (frame)'])==i-1:
                vec1+=[ int(row['Track Number'])]
        naissancestraj=list(set(vec2)-(set(vec1)&set(vec2)))
        print(naissancestraj)
        if len(naissancestraj)!=0:
            drtpsnx+=[(i-1)*intertps]*len(naissancestraj)
            avtnxj=[]
            avtnxj2=[]
            f=open ('data-Rab11.csv','r')
            reader=csv.DictReader(f, delimiter=',')
            for row in reader :
                if int(row['t (frame)'])==i :
                    avtnxj+=[float(row['X (pixel)']), float(row['Y(pixel)']),1,int(row['Motion Type'])]
                if int(row['t (frame)'])==i-1 :
                    avtnxj2+=[float(row['X (pixel)']), float(row['Y(pixel)']),1,int(row['Motion Type'])]
            dravtnx+=[avtnxj]*len(naissancestraj)
            dravtnx2+=[avtnxj2]*len(naissancestraj)
            for newtraj in naissancestraj :
                fl=open ('data-Langerin.csv','r')
                readerl=csv.DictReader(fl, delimiter=',')
                for row in readerl :
                    if int(row['Track Number'])==newtraj and int(row['t (frame)'])==i :
                        drnx+=[[float(row['X (pixel)']), float(row['Y(pixel)']),0,int(row['Motion Type'])]]
        return(drnx, dravtnx, resfinalrab)



"""this function calculates the log-likelihood as a function of the parameters sigma and p.
To make the results easier to read, we have used a logarithmic scale for the sigma parameter."""

def logLplt(sigma,p,avtnx, nx):
    """
    Parameters
    ----------
    avtnx : LIST
        result of extract_real_data
    nx : LIST
        result of extract_real_data
    sigma : FLOAT
        value of the sigma parameter for which we want to calculate the log-likelihood
    p : FLOAT
        value of the p parameter for which we want to calculate the log-likelihood

    Returns
    -------
    gs : FLOAT
        value of the log-likelihood at the point (p, sigma) entered as an argument
    """
    gs=0
    for j in range(len(nx)):
        n=0
        xx=avtnx[j]
        yy=nx[j]
        for i in range(len(xx)//4):
            #v=(x[3*i]-y[0])**2+(x[3*i+1]-y[1])**2
            #v=distance([xx[3*i],xx[3*i+1]],[yy[0],yy[1]])
            v=np.linalg.norm([xx[4*i]-yy[0],xx[4*i+1]-yy[1]])**2
            #n+=np.exp(-v/(2*sigma**2)) 
            n+=np.exp(-v/(2*np.log(sigma)**2)) 
        #gs+=np.log((p*n)/((len(xx)//3)*(sigma**2)*2*np.pi)+(1-p)/Airecell)
        gs+=np.log((p*n)/((len(xx)//4)*(np.log(sigma)**2)*2*np.pi)+(1-p)/Airecell)
    return(gs)        



"""this program calculates the maximum likelihood argument"""

def maxvrs(avtnx, nx) :
    """
    Parameters
    ----------
    avtnx : LIST
        result of extract_real_data
    nx : LIST
        result of extract_real_data
    
    Returns
    -------
    [p,s] : LIST
        list containing the argmax of the log-likelihood with p in first place and sigma in second place
    """
    #(resfinal,TpsSauts,tabecarts,track,compteurs,tracknaissance, trackmort, nx, avtnx, tpsnx) =b 
    g=lambda th:-logLplt(th[0],th[1],avtnx,nx)
    aa=op.minimize(g,[3,0.06],bounds=[(1,5),(0,0.2)],method='L-BFGS-B') 
    #aa=op.minimize(g,[1.5,0.15],bounds=[(0.5,5),(0.001,0.7)],method='L-BFGS-B')#, tol=1e-06)
    print(aa)
    p=aa['x'][1]
    s=aa['x'][0]
    return([p,s])



"""this program returns the matrix required to plot the log-likelihood heat map"""
def logvrsdim2(P,T, avtnx, nx):
    """
    Parameters
    ----------
    P,T : ARRAY
        mesh grid for plotting the function
    avtnx : LIST
        result of extract_real_data
    nx : LIST
        result of extract_real_data
    
    Returns
    -------
    Z : ARRAY
        array of the same size as P and T, which gives the value of the log-likelihood on the points of the mesh grid P,T
    """
    (n,m)=P.shape
    Z=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            #Z[i,j]+=(1/(2*np.sqrt(2*np.pi)*theta))*np.exp(-(np.linalg.norm([xi[k]-X[i,j],yi[k]-Y[i,j]])**2)/(2*theta**2))
            #Z[i,j]+=(1/(np.sqrt(2*np.pi)*theta*np.sqrt(len(xi)))**2)*np.exp(-(distanceperio([xi[k],yi[k]],[X[i,j],Y[i,j]]))/(2*theta**2))
            Z[i,j]=logLplt(T[i,j],P[i,j],avtnx,nx)
    return(Z)



"""this program calculates the value of the function which is under the integrale in 
the defintion of matrix J(p, sigma) in Theorem 2 of the linked article."""
def matssint(sigma,p, X, y):
    """
    Parameters
    ----------
    sigma : FLOAT
        value of the sigma parameter for which we want to calculate the J(p, sigma) matrix
    p : FLOAT
        value of the p parameter for which we want to calculate the (p, sigma) matrix
    X : LIST
        list containing the values of the points presents at the time the function under the integral is calculated, denoted X_s in Theorem 2
    y : LIST
        point y of Theorem 2

    Returns
    -------
    coef1 : FLOAT
        value of the coefficient (1,1) of the matrix which is under the integral in Theorem 2 calculates for X_s=X and y=y
    coef1 : FLOAT
        value of coefficients (1,2) and (2,1) (it's the same since J is symetric) of the matrix which is under the integral in Theorem 2 calculates for X_s=X and y=y
    coef3 : FLOAT
        value of the coefficient (2,2) of the matrix which is under the integral in Theorem 2 calculates for X_s=X and y=y
    """
        
    ros1=0
    vio1=0
    ros2=0
    vio2=0
    ros3=0
    vio3=0
    for i in range(len(X)//4):
        v=np.linalg.norm([X[4*i]-y[0],X[4*i+1]-y[1]])**2
        #print(v)
        if X[4*i+3]==3 :
            ros3+=((v/(np.log(sigma)**2))-2)*np.exp(-v/(2*np.log(sigma)**2))
            vio3+= np.exp(-v/(2*np.log(sigma)**2))
        if X[4*i+3]==1 :
            ros1+=((v/(np.log(sigma)**2))-2)*np.exp(-v/(2*np.log(sigma)**2))
            vio1+= np.exp(-v/(2*np.log(sigma)**2))
        if X[4*i+3]==2 :
            ros2+=((v/(np.log(sigma)**2))-2)*np.exp(-v/(2*np.log(sigma)**2))
            vio2+= np.exp(-v/(2*np.log(sigma)**2))
        #print(den)
    #print((num**2)/den)
    f=(4.45/(2.98+4.45))*p/((len(X)//4)*2*np.pi*np.log(sigma)**3)
    g=(4.45/(2.98+4.45))*1/((len(X)//4)*2*np.pi*np.log(sigma)**2)
    h1=vio1*g-(4.45/(2.98+4.45))*plbirth[0]*1/Airecell
    h2=vio2*g-(4.45/(2.98+4.45))*plbirth[1]*1/Airecell
    h3=vio3*g-(4.45/(2.98+4.45))*plbirth[2]*1/Airecell
    k1=p*h1+plbirth[0]/Airecell
    k2=p*h2+plbirth[1]/Airecell
    k3=p*h3+plbirth[2]/Airecell
    coef3=4.45*((f*ros1)**2/k1+(f*ros2)**2/k2+(f*ros3)**2/k3)
    coef2=4.45*(f*ros1*h1/k1+f*ros2*h2/k2+f*ros3*h3/k3)
    coef1=4.45*(h1**2/k1+h2**2/k2+h3**2/k3)
    return(coef1,coef2,coef3)   

"""using the previous program, this program calculates the value of the matrix J(p, sigma)
using a rectangular method in dimension 2, i.e. integrating the function 
under the integral, taking it to be constant over small rectangles"""
def matcovarempfin(pmax, smax,resfinal):
    """
    Parameters
    ----------
    pmax : FLOAT
        value of the p parameter for which we want to calculate the J(p, sigma) matrix (normally the likelihood argmax)
    smax : FLOAT
        value of the sigma parameter for which we want to calculate the (p, sigma) matrix (normally the likelihood argmax)
    resfinal : LIST
        list containing the values of the points presents during the time the integral is calculated, denoted X_s in Theorem 2

    Returns
    -------
    coef1 : FLOAT
        value of the coefficient (1,1) of the matrix J(p, sigma)
    coef1 : FLOAT
        value of coefficients (1,2) and (2,1) (it's the same since J is symetric) of the matrix J(p, sigma)
    coef3 : FLOAT
        value of the coefficient (2,2) of the matrix J(p, sigma)
    """
    xa = np.linspace(0, 249, 10)
    xo = np.linspace(0, 282, 10)
    coef1=0
    coef2=0
    coef3=0
    for k in range(len(resfinal)):
        print(k)
        for i in range(1,len(xa)):
            for j in range(1,len(xo)):
                if beincell([(xa[i]+xa[i-1])/2,(xo[j]+xo[j-1])/2])*True :
                    r= matssint(smax,pmax,resfinal[k],[(xa[i]+xa[i-1])/2,(xo[j]+xo[j-1])/2])
                    coef1+=r[0]*(xa[i]-xa[i-1])*(xo[j]-xo[j-1])*intertps
                    coef2+=r[1]*(xa[i]-xa[i-1])*(xo[j]-xo[j-1])*intertps
                    coef3+=r[2]*(xa[i]-xa[i-1])*(xo[j]-xo[j-1])*intertps
    return(coef1,coef2,coef3)


"""this program can be used to draw an ellipse on a graph"""
def plot_confidence_ellipse(mu, cov, alph, ax, clabel=None, label_bg='white', **kwargs):
    """Display a confidence ellipse of a bivariate normal distribution
    
    Arguments:
        mu {array-like of shape (2,)} -- mean of the distribution
        cov {array-like of shape(2,2)} -- covariance matrix
        alph {float btw 0 and 1} -- level of confidence
        ax {plt.Axes} -- axes on which to display the ellipse
        clabel {str} -- label to add to ellipse (default: {None})
        label_bg {str} -- background of clabel's textbox
        kwargs -- other arguments given to class Ellipse
    """
    c = -2 * np.log(1 - alph)  # quantile at alpha of the chi_squarred distr. with df = 2
    Lambda, Q = np.linalg.eig(cov)  # eigenvalues and eigenvectors (col. by col.)
    
    ## Compute the attributes of the ellipse
    width, heigth = 2 * np.sqrt(c * Lambda)
    # compute the value of the angle theta (in degree)
    theta = 180 * np.arctan(Q[1,0] / Q[0,0]) / np.pi if cov[1,0] else 0
        
    ## Create the ellipse
    if 'fc' not in kwargs.keys():
        kwargs['fc'] = 'None'
    level_line = Ellipse(mu, width, heigth, angle=theta, **kwargs)
    
    ## Display a label 'clabel' on the ellipse
    if clabel:
        col = kwargs['ec'] if 'ec' in kwargs.keys() and kwargs['ec'] != 'None' else 'black'  # color of the text
        pos = Q[:,1] * np.sqrt(c * Lambda[1]) + mu  # position along the heigth
        
        ax.text(*pos, clabel, color=col,
           rotation=theta, ha='center', va='center', rotation_mode='anchor', # rotation
           bbox=dict(boxstyle='round',ec='None',fc=label_bg, alpha=1)) # white box
    return ax.add_patch(level_line)


"""IInstructions for displaying graphics"""

(drnx, dravtnx, resfinalrab)=extract_real_data()

p = np.linspace(0, 0.2, 50)
t = np.linspace(1.1, 10, 50)
P, T = np.meshgrid(p, t)
Z=logvrsdim2(P,T, dravtnx, drnx)

#to calculate the argmax of log-likelihood 
[pmax,smax]=maxvrs(dravtnx, drnx)

#to calculate the matrix J(pmax,smax)
a=matcovarempfin(pmax, smax,resfinalrab)

#to plot the log-likelihood heat map
fig, ax = plt.subplots()
ax.set_title('heat map of the log-likelihood')
pc = ax.pcolormesh(P, T, Z, cmap='jet', shading='gouraud',vmin=-1682, vmax = -1675)
fig.colorbar(pc)
plt.show()

#to plot the log-likelihood heat map and overlay in black the estimated 95% confidence ellipse, centered at the maximum likelihood,
fig, ax = plt.subplots()
pc = ax.pcolormesh(P, T, Z,vmin=-7930, vmax = -7888, cmap='jet',shading="gouraud")
fig.colorbar(pc)
cov=np.array([[a[0],a[1]],[a[1],a[2]]])
invcov=np.linalg.inv(cov)
plot_confidence_ellipse([pmax, smax],invcov,0.95,ax,edgecolor='black', fill=0)
plt.show()

