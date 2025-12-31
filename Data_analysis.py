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

from Codes import estimation_functions as estimation


# We extract the characteristics we need from the files containing the data (called data-Rab11.csv and data-Langerin.csv), 
# These files must be in the folder Data
# This step takes 5 minutes
(drnx, dravtnx, resfinalrab)=estimation.extract_real_data(intertps=0.14)


# Import the region of interest, that is the cell. 
# The file maskc.pickle must be in the folder Data
with open('Data/maskc.pickle', 'rb') as handle:
    imgmod = pickle.load(handle) 

#This function allows us to test if a point is in the cell
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


#Parameter needed for the following
Areacell_data=42525 #total area of the observed cell


#Estimation of the parameter
[pmax,smax], ma=estimation.maxvrs(dravtnx, drnx,areacell=Areacell_data,init=[3,0.06],bounds=[(1,5),(0,0.2)])
print(pmax,smax)

#Estimation of the matrix J(pmax,smax), which is the inverse asymptotic covariance matrix of the estimators
a=estimation.matJ(pmax, smax,resfinalrab,areacell=Areacell_data,intertps=0.14,beta=4.45,plbirth=None,beincell=beincell,xa = np.linspace(0, 249, 10),xo = np.linspace(0, 282, 10))



#Plot of the log-likelihood heat map along with the confidence ellipsoid, centered at the maximum likelihood estimator
p = np.linspace(0, 0.2, 50)
t = np.linspace(1.1, 10, 50)
P, T = np.meshgrid(p, t)
Z=estimation.logvrsdim2(P,T, dravtnx, drnx,areacell=Areacell_data) #takes a couple of minutes

fig, ax = plt.subplots()
pc = ax.pcolormesh(P, T, Z,vmin=-7930, vmax = -7888, cmap='jet',shading="gouraud")
fig.colorbar(pc)
invcov=np.array([[a[0],a[1]],[a[1],a[2]]])
cov=np.linalg.inv(invcov)
estimation.plot_confidence_ellipse([pmax, smax],cov,0.95,ax,edgecolor='black', fill=0)
ax.scatter(pmax, smax, marker='x', color='black')
plt.show()
plt.close()
