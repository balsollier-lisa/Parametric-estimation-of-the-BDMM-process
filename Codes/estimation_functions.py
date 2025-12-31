#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:26:21 2024

@author: lisabalsollier
"""


import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import scipy.optimize as op
from matplotlib.patches import Ellipse



# Log likelihood of the birth kernel where the marks are independent from the birth location (used for the data analysis)
def logLplt(theta,p,avtnx, nx, areacell):
    """
    Parameters
    ----------
    theta : FLOAT
        value of the theta parameter for which we want to calculate the log-likelihood
        theta=exp(sigma) where sigma is the sd of the Gaussian distributions
    p : FLOAT
        value of the p parameter for which we want to calculate the log-likelihood
    avtnx : LIST of all Rab11 particles present at each birth time of a Langerin particle
    nx : LIST of each Langerin particle apparing at their birth time
    areacell : FLOAT area of the cell where the particles live

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
        v=np.linalg.norm([xx[4*i]-yy[0],xx[4*i+1]-yy[1]])**2
        n+=np.exp(-v/(2*np.log(theta)**2))
      gs+=np.log((p*n)/((len(xx)//4)*(np.log(theta)**2)*2*np.pi)+(1-p)/areacell)
    return(gs)

# Log likelihood of the birth kernel with dependent marks (used for the simulations)
# where the 3 marks of Langering are similar to Rab11 if colocalization, and otherwise random wrt plbirth
def logLplt_marks(theta,p,avtnx, nx, areacell,plbirth):
    """
    Parameters
    ----------
    theta : FLOAT
        value of the theta parameter for which we want to calculate the log-likelihood
        theta=exp(sigma) where sigma is the sd of the Gaussian distributions
    p : FLOAT
        value of the p parameter for which we want to calculate the log-likelihood
    avtnx : LIST of all Rab11 particles present at each birth time of a Langerin particle
    nx : LIST of each Langerin particle apparing at their birth time
    areacell : FLOAT area of the cell where the particles live
    plbirth : vector of the 3 probabilities of marks when there is no colocalization

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
            v=np.linalg.norm([xx[4*i]-yy[0],xx[4*i+1]-yy[1]])**2
            samemark=int(xx[4*i+3]==yy[3])
            n+=np.exp(-v/(2*np.log(theta)**2))*samemark
        if yy[3]==1 or yy[3]==3:
          mark=int(yy[3]-1)
        else:
          mark=1
        gs+=np.log((p*n)/((len(xx)//4)*(np.log(theta)**2)*2*np.pi)+plbirth[mark]*(1-p)/areacell)
    return(gs)



# This function calculates the maximum likelihood argument
def maxvrs(avtnx, nx, areacell,init=[1.5,0.45],bounds=[(1.01, 2),(0.1,1)],plbirth=None) :
    """
    Parameters
    ----------
    avtnx : LIST of all Rab11 particles present at each birth time of a Langerin particle
    nx : LIST of each Langerin particle apparing at their birth time
    areacell : FLOAT area of the cell where the particles live
    plbirth : vector of the 3 probabilities of marks when there is no colocalization (None if the marks are independent)

    Returns
    -------
    [p,s] : LIST
        list containing the argmax of the log-likelihood with p in first place and theta in second place
    """
    if plbirth==None:
      g=lambda th:-logLplt(th[0],th[1],avtnx,nx,areacell)
      aa=op.minimize(g,init,bounds=bounds,method='L-BFGS-B') 
    else:
      g=lambda th:-logLplt_marks(th[0],th[1],avtnx,nx,areacell,plbirth)
      aa=op.minimize(g,init,bounds=bounds,method='L-BFGS-B') 
    ma=aa['fun']
    p=aa['x'][1]
    s=aa['x'][0]
    return([p,s],ma)
  
  

# This function returns the matrix required to plot the log-likelihood heat map
def logvrsdim2(P,T, avtnx, nx, areacell,plbirth=None):
    """
    Parameters
    ----------
    P,T : ARRAY
        mesh grid for plotting the function
    avtnx : LIST of all Rab11 particles present at each birth time of a Langerin particle
    nx : LIST of each Langerin particle apparing at their birth time
    areacell : FLOAT area of the cell where the particles live
    plbirth : vector of the 3 probabilities of marks when there is no colocalization (None if the marks are independent)

    Returns
    -------
    Z : ARRAY
        array of the same size as P and T, which gives the value of the log-likelihood on the points of the mesh grid P,T
    """
    (n,m)=P.shape
    Z=np.zeros((n,m))
    if plbirth==None:
      for i in range(n):
        for j in range(m):
          Z[i,j]=logLplt(T[i,j],P[i,j],avtnx,nx,areacell)
    else:
      for i in range(n):
        for j in range(m):
          Z[i,j]=logLplt_marks(T[i,j],P[i,j],avtnx,nx,areacell,plbirth)
    return(Z)
  
  
  
# This function calculates the value of the integrand in 
# the defintion of the matrix J(p, theta) (the inverse of the asymptotic covariance matrix of the MLE)
def matssint(theta,p, X, y, areacell,beta,plbirth=None):
  """
  Parameters
  ----------
  theta : FLOAT
      value of the theta parameter for which we want to calculate the J(p, theta) matrix
  p : FLOAT
      value of the p parameter for which we want to calculate the J(p, theta) matrix
  X : LIST
      list containing the values of the points presents at the time the function under the integral is calculated, denoted X_s in Theorem 2
  y : LIST
      point y of Theorem 2
  areacell : FLOAT area of the cell where the particles live
  beta : value of the birth intensity function at X
  plbirth : vector of the 3 probabilities of marks when there is no colocalization (None if the marks are independent)

  Returns
  -------
  coef1 : FLOAT
      value of the coefficient (1,1) of the matrix which is under the integral in Theorem 2 calculates for X_s=X and y=y
  coef2 : FLOAT
      value of coefficients (1,2) and (2,1) (it's the same since J is symetric) of the matrix which is under the integral in Theorem 2 calculates for X_s=X and y=y
  coef3 : FLOAT
      value of the coefficient (2,2) of the matrix which is under the integral in Theorem 2 calculates for X_s=X and y=y
  """
    
  if plbirth==None:
    ros=0
    vio=0
    for i in range(len(X)//4):
      v=np.linalg.norm([X[4*i]-y[0],X[4*i+1]-y[1]])**2
      ros+=((v/(np.log(theta)**2))-2)*np.exp(-v/(2*np.log(theta)**2))
      vio+= np.exp(-v/(2*np.log(theta)**2))
    f=p/((len(X)//4)*2*np.pi*np.log(theta)**3*theta)
    g=1/((len(X)//4)*2*np.pi*np.log(theta)**2)
    h=vio*g-1/areacell
    k=p*h+1/areacell
    coef3=(f*ros)**2/k
    coef2=f*ros*h/k
    coef1=h**2/k
  else:
    ros1=0     
    vio1=0
    ros2=0
    vio2=0
    ros3=0
    vio3=0
    for i in range(len(X)//4):
         v=np.linalg.norm([X[4*i]-y[0],X[4*i+1]-y[1]])**2
         if X[4*i+3]==3:
           ros3+=((v/(np.log(theta)**2))-2)*np.exp(-v/(2*np.log(theta)**2))
           vio3+= np.exp(-v/(2*np.log(theta)**2))
         if X[4*i+3]==1:
           ros1+=((v/(np.log(theta)**2))-2)*np.exp(-v/(2*np.log(theta)**2))
           vio1+= np.exp(-v/(2*np.log(theta)**2))
         if X[4*i+3]==2:
           ros2+=((v/(np.log(theta)**2))-2)*np.exp(-v/(2*np.log(theta)**2))
           vio2+= np.exp(-v/(2*np.log(theta)**2))
    f=p/((len(X)//4)*2*np.pi*np.log(theta)**3*theta)
    g=1/((len(X)//4)*2*np.pi*np.log(theta)**2)
    h1=vio1*g-plbirth[0]*1/areacell
    h2=vio2*g-plbirth[1]*1/areacell
    h3=vio3*g-plbirth[2]*1/areacell
    k1=p*h1+plbirth[0]*1/areacell
    k2=p*h2+plbirth[1]*1/areacell
    k3=p*h3+plbirth[2]*1/areacell
    coef3=(f*ros1)**2/k1+(f*ros2)**2/k2+(f*ros3)**2/k3
    coef2=f*ros1*h1/k1+f*ros2*h2/k2+f*ros3*h3/k3
    coef1=h1**2/k1+h2**2/k2+h3**2/k3
  return(beta*coef1,beta*coef2,beta*coef3)



# This function calculates the matrix J(p, theta) (the inverse of the asymptotic covariance matrix of the MLE)
def matJ(pmax, smax,resfinal,areacell,intertps,beta,plbirth,beincell,xa = np.linspace(-10, 10, 10),xo = np.linspace(-10, 10, 10)):
    """
    Parameters
    ----------
    pmax : FLOAT
        value of the p parameter for which we want to calculate the J(p, theta) matrix (normally the likelihood argmax)
    smax : FLOAT
        value of the theta parameter for which we want to calculate the (p, theta) matrix (normally the likelihood argmax)
    resfinal : LIST
        list containing the values of the points present during the time the integral is calculated, denoted X_s in Theorem 2
    areacell : FLOAT area of the cell where the particles live
    intertps : inter-time between two observations
    plbirth : vector of the 3 probabilities of marks when there is no colocalization (None if the marks are independent)
    beincell : FUNCTION that tests if a point is in the cell
    xa : vector of x-axis points where to evaluate the discretized integral
    xo : vector of y-axis points where to evaluate the discretized integral
  
    Returns
    -------
    coef1 : FLOAT
        value of the coefficient (1,1) of the matrix J(p, theta)
    coef2 : FLOAT
        value of coefficients (1,2) and (2,1) (it's the same since J is symetric) of the matrix J(p, theta)
    coef3 : FLOAT
        value of the coefficient (2,2) of the matrix J(p, theta)
    """
    #xa = np.linspace(-10, 10, 10)
    #xo = np.linspace(-10, 10, 10)
    coef1=0
    coef2=0
    coef3=0
    for k in range(len(resfinal)):
        for i in range(1,len(xa)):
            for j in range(1,len(xo)):
                if beincell([(xa[i]+xa[i-1])/2,(xo[j]+xo[j-1])/2])*True :
                    r= matssint(smax,pmax,resfinal[k],[(xa[i]+xa[i-1])/2,(xo[j]+xo[j-1])/2],areacell,beta,plbirth)
                    coef1+=r[0]*(xa[i]-xa[i-1])*(xo[j]-xo[j-1])*intertps
                    coef2+=r[1]*(xa[i]-xa[i-1])*(xo[j]-xo[j-1])*intertps
                    coef3+=r[2]*(xa[i]-xa[i-1])*(xo[j]-xo[j-1])*intertps
                    
    return(coef1,coef2,coef3)

  
      
# This function can be used to draw a confidence ellipse from a bivariate Gaussian distribution
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





# This function allows to extract the characteristics we need from the files containing the data (called data-Rab11.csv and data-Langerin.csv) 
def extract_real_data(intertps):
    """
    Parameters
    ----------
    intertps : inter-time between two consecutive observation.

    Returns
    -------
    drnx : LIST
        list that contains the coordinates of the Langerin particles at the time of his birth
    dravtnx : LIST
        list that contains list of the coordinates of the Rab11 particles that are present at the time of the births of new Langerin particle
    resfinalrab : LIST
        list that contains at position i a list of the coordinates of the Rab11 particles that are present at the time i
    """
    
    f=open ('Data/data-Rab11.csv','r')
    reader=csv.DictReader(f, delimiter=',')
    nframe=np.max([int(row['t (frame)']) for row in reader])
    resfinalrab=[]
    for i in range(1,nframe+1):
        f=open ('Data/data-Rab11.csv','r')
        reader=csv.DictReader(f, delimiter=',')
        pts2=[]
        for row in reader :
            if int(row['t (frame)'])==i :
                pts2+=[float(row['X (pixel)']), float(row['Y(pixel)']),1,int(row['Motion Type'])]
        resfinalrab+=[pts2]
    
    drtpsnx=[]
    drnx=[]
    dravtnx=[]
    dravtnx2=[]
    for i in range(2,nframe+1):
        vec1=[]
        vec2=[]
        fl=open ('Data/data-Langerin.csv','r')
        readerl=csv.DictReader(fl, delimiter=',')
        for row in readerl:
            if int(row['t (frame)'])==i:
                vec2+=[ int(row['Track Number'])]
            if int(row['t (frame)'])==i-1:
                vec1+=[ int(row['Track Number'])]
        naissancestraj=list(set(vec2)-(set(vec1)&set(vec2)))
        if len(naissancestraj)!=0:
            drtpsnx+=[(i-1)*intertps]*len(naissancestraj)
            avtnxj=[]
            avtnxj2=[]
            f=open ('Data/data-Rab11.csv','r')
            reader=csv.DictReader(f, delimiter=',')
            for row in reader :
                if int(row['t (frame)'])==i :
                    avtnxj+=[float(row['X (pixel)']), float(row['Y(pixel)']),1,int(row['Motion Type'])]
                if int(row['t (frame)'])==i-1 :
                    avtnxj2+=[float(row['X (pixel)']), float(row['Y(pixel)']),1,int(row['Motion Type'])]
            dravtnx+=[avtnxj]*len(naissancestraj)
            dravtnx2+=[avtnxj2]*len(naissancestraj)
            for newtraj in naissancestraj :
                fl=open ('Data/data-Langerin.csv','r')
                readerl=csv.DictReader(fl, delimiter=',')
                for row in readerl :
                    if int(row['Track Number'])==newtraj and int(row['t (frame)'])==i :
                        drnx+=[[float(row['X (pixel)']), float(row['Y(pixel)']),0,int(row['Motion Type'])]]
                        
    return(drnx, dravtnx, resfinalrab)







