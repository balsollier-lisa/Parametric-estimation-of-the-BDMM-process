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

import estimation

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




"""Instructions for displaying graphics"""

(drnx, dravtnx, resfinalrab)=extract_real_data()

p = np.linspace(0, 0.2, 50)
t = np.linspace(1.1, 10, 50)
P, T = np.meshgrid(p, t)
Z=estimation.logvrsdim2(P,T, dravtnx, drnx)

#to calculate the argmax of log-likelihood 
[pmax,smax]=estimation.maxvrs(dravtnx, drnx)

#to calculate the matrix J(pmax,smax)
a=estimation.matcovarempfin(pmax, smax,resfinalrab)

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
estimation.plot_confidence_ellipse([pmax, smax],invcov,0.95,ax,edgecolor='black', fill=0)
plt.show()

