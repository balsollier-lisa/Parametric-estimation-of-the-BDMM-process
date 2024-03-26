
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lisabalsollier
"""

"""
You will find in this file all the necessary programs to generate with the 
functions of process.py file, a simulation of BDM which approaches the real data.
This file was used to generate the simulations presented in the article.
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import pickle
import scipy.optimize as op
from matplotlib.patches import Ellipse

import process
import draw




"""
Definition of auxiliary functions used in the main function process.proctotal 
to generate simulations.
"""




"""this program allows to say if a point is in the cell"""
def beincell(x):
    if np.abs(x[0])>10:
        return(0)
    else : 
        return(1)
  


"""this program generates a random initial situation"""
def generatesituation(M):
    """
    Parameters
    ----------
    M : INTEGER
        number of particles of the configuration that will be returned.

    Returns
    -------
    X : ARRAY
        vector that contains the abscissae of the points that form the configuration that will be returned.
    Y : ARRAY
        vector that contains the ordinates of the points that form the configuration that will be returned.
    R : ARRAY
        vector that contains the type (0 for Langerin and 1 for Rab11 of the points that form the configuration that will be returned.
    C : TYPE
        vector that contains the style of movement (1 for brownian, 2 for superdiffusive and 3 for confined) of the points that form the configuration that will be returned.

    """
    X=[]
    Y=[]
    C=[]
    R=[]
    for k in range(M):
        x=stats.uniform.rvs(loc=-10,scale=20)
        y=stats.uniform.rvs(loc=-10,scale=20)
        X+=[x]
        Y+=[y]
        
        R+=[0]
        C+=[1]
        
        track=[[i for i in range(1,M+1)]]
        cpttrack=M
    return(X,Y,R,C,track,cpttrack)


"""this program generates a Brownian motion of diffusion coefficient s1lang 
or s1rab with as initial position the point (xi,yi,ri,ci)"""
def brownianmove(xi,yi,ri,ci,ecarts,N,s1lang):
    if ri==0 :
        normx=stats.norm.rvs(loc=0, scale=s1lang*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s1lang*np.sqrt(np.array(ecarts)))
        xx=np.cumsum(np.concatenate((np.array([xi]),normx)))
        yy=np.cumsum(np.concatenate((np.array([yi]),normy)))
        co=[[xx[m],yy[m]] for m in range(len(xx))]
        #print(co)
        ide=np.argwhere(np.array([beincell(co[p]) for p in range(len(co))])==0)
        #print('chgmt')
        #print(ide)
        while len(ide)!=0:
            k=ide[0][0]
            #print(i)
            normx[k-1]=stats.norm.rvs(loc=0, scale=s1lang*np.sqrt(ecarts[k-1]))
            normy[k-1]=stats.norm.rvs(loc=0, scale=s1lang*np.sqrt(ecarts[k-1]))
            #normx[k-1]=normx[k-1]-2*np.sign(xx[k])
            #print(normx)
            xx=np.cumsum(np.concatenate((np.array([xi]),normx)))
            yy=np.cumsum(np.concatenate((np.array([yi]),normy)))
            co=[[xx[m],yy[m]] for m in range(len(xx))]
            ide=np.argwhere(np.array([beincell(co[p]) for p in range(len(co))])==0)
            #print(ide)
    cc=[ci]*(N)
    rr=[ri]*(N)
    #print('fini')
    return(xx,yy,rr,cc)



"""this program generates the motion of a particle up to time t according to its coordinate c"""
def move(t,x,y,r,c,d) :
    """
    Parameters
    ----------
    t : FLOAT
        time of the movement that we want to generate.
    x : ARRAY
        vector that contains the abscissae of the points that form the initial condition.
    y : ARRAY
        vector that contains the ordinates of the points that form the initial condition.
    r : ARRAY
        vector that contains the type (0 for Langerin and 1 for Rab11 of the points that form the initial condition.
    c : TYPE
        vector that contains the style of movement (1 for brownian, 2 for superdiffusive and 3 for confined) of the points that form the initial condition..
    d : FLOAT
        pace discretization of the motion.

    Returns
    -------
    res : ARRAY
        table that contains all the coordinates of the points of the simulation. There are as many lines as there are discretizations of time t. Each lines of the array contains all the points present (in the order abscissa, ordinate, type, regime).
    ecarts : ARRAY
        vector that contains the time between two lines of resfinal.
    tps : ARRAY
        cumsum of ecarts 

    """
    P=len(x)
    if t>4*d :
        tps=np.concatenate((np.arange(0,t,d),np.array([t])))
        N=len(tps)
        ecarts=[d]*(len(tps)-2)+[t-(d*(len(tps)-2))]
    else :
        N=4
        tps=np.linspace(0,t,N)
        ecarts=[tps[i+1]-tps[i] for i in range(len(tps)-1)]
    #print(tps)
    #print(ecarts)
    res= np.zeros((N,4*P))
    for i in range (P) :
        if c[i]==1 :
            #print('br1')
            #xx,yy,rr,cc=brownianmove(x[i],y[i],r[i],c[i],ecarts,N, np.sqrt(np.random.choice(brlang)),np.sqrt(np.random.choice(brrab)))
            xx,yy,rr,cc=brownianmove(x[i],y[i],r[i],c[i],ecarts,N, s1lang)
            #print('br2')
        res[:,4*i]=xx
        res[:,4*i+1]=yy
        res[:,4*i+3]=cc
        res[:,4*i+2]=rr
        #print(i==P-1)
    return(res,ecarts,tps)



"""birth intensity : constant and equal to 10"""
def beta(x) : 
    return(10)
    #return(4.45+2.98)#*n(x))


"""death intensity : constant and equal to 10"""
def delta(x): 
    if process.n(x)==1:
        return(0)
    else :
        return(10)
        #return(0.12*nl(x)+0.14*nr(x))
        #return(0.15*process.nlb(x)+0.1*process.nlsp(x)+0.058*process.nlsb(x)+0.18*process.nrb(x)+0.22*process.nrsp(x)+0.07*process.nrsb(x))
    
    
"""mutation intensity : equal to zero : no transition """
def tau(x) : 
    #return(16/167.86)
    return(0)
    
    
def alpha(x) :
    return(beta(x)+delta(x)+tau(x))  



"""birth kernel : as in the section 5.1 of the article """
def birthkernel(depart):
    """
    Parameters
    ----------
    depart : ARRAY
        Array that contains the coordinates of the point configuration present
        just before the new birth (in the order abscissa, ordinate, type, regime).

    Returns
    -------
    The coordinates of the new point (in the order abscissa, ordinate, type, regime).
    """
    R=0
    p=stats.uniform.rvs(0,1)
    if p<q :
        P=process.nl(depart)
        i=stats.randint.rvs(1,P+1)
        m=0
        s=0
        while s<i and m<len(depart)/4:
            if depart[4*m+2]== 1 :
                s+=1
            m+=1
        X=stats.norm.rvs(loc=depart[4*(m-1)],scale=theta)
        Y=stats.norm.rvs(loc=depart[4*(m-1)+1],scale=theta)
        while beincell([X,Y])==0:
            X=stats.norm.rvs(loc=depart[4*(m-1)],scale=theta)
            Y=stats.norm.rvs(loc=depart[4*(m-1)+1],scale=theta)
        C=1
    else :
        X=stats.uniform.rvs(loc=-10,scale=20)
        Y=stats.uniform.rvs(loc=-10,scale=20)
        while beincell([X,Y])==0 :
            X=stats.uniform.rvs(loc=-10,scale=20)
            Y=stats.uniform.rvs(loc=-10,scale=20)
        C=1
#R=stats.randint.rvs(0,2)    
    return(X,Y,R,C)


"""death kernel : uniform on present particle"""
def deathkernel(depart): 
    """
    Parameters
    ----------
    depart : ARRAY
        Array that contains the coordinates of the point configuration present
        just before the next death (in the order abscissa, ordinate, type, regime).

    Returns
    -------
    The same array as in the input but without the coordinates of the dead center.
    """
    P=process.n(depart)
    
     
    i=stats.randint.rvs(1,P+1)
        
    tab=len(depart)*[True]
    tab[4*(i-1)]=False
    tab[4*(i-1)+1]=False
    tab[4*(i-1)+2]=False
    tab[4*(i-1)+3]=False
    return(depart[tab],0,1,i-1)


"""transition kernel : useless since there is no transition"""
def transitionkernel(depart): 
    """
    Parameters
    ----------
    depart : ARRAY
        Array that contains the coordinates of the point configuration present
        just before the next transition (in the order abscissa, ordinate, type, regime).

    Returns
    -------
    The same array as in the input but with the new coordinate "regime" of the point that has changed.
    """
    return(None)







"""
To calculate the log-likelihood

"""





"""this function calculates the log-likelihood as a function of the parameters sigma and p.
To make the results easier to read, we have used a logarithmic scale for the sigma parameter."""

def logLplt(inc,p,avtnx, nx):
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
            #n+=np.exp(-v/(2*inc**2)) 
            n+=np.exp(-v/(2*np.log(inc)**2)) 
        #gs+=np.log((p*n)/((len(xx)//3)*(inc**2)*2*np.pi)+(1-p)/Airecell)
        gs+=np.log((p*n)/((len(xx)//4)*(np.log(inc)**2)*2*np.pi)+(1-p)/Airecell)
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
     
    g=lambda inc:-logLplt(inc[0],inc[1],avtnx,nx)
    aa=op.minimize(g,[2,0.4],bounds=[(1.8, 3.5),(0.1,1)],method='L-BFGS-B') 
    
    ma=aa['fun']
    p=aa['x'][1]
    th=aa['x'][0]
    return([p,th], ma)

"""this program returns the matrix required to plot the log-likelihood heat map"""
def tracerlogvrsdim2(P,T, avtnx, nx):
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
            Z[i,j]=logLplt(T[i,j],P[i,j],avtnx,nx)
    return(Z)



"""this program calculates the value of the function which is under the integrale in 
the defintion of matrix J(p, sigma) in Theorem 2 of the linked article."""
def matssint(inc,p, X, y):
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
    ros=0
    vio=0
    for i in range(len(X)//4):
        v=np.linalg.norm([X[4*i]-y[0],X[4*i+1]-y[1]])**2
        #print(v)
        ros+=((v/(np.log(inc)**2))-2)*np.exp(-v/(2*np.log(inc)**2))
        #print(num)
        vio+= np.exp(-v/(2*np.log(inc)**2))
        #print(den)
    #print((num**2)/den)
    f=p/((len(X)//4)*2*np.pi*np.log(inc)**3)
    g=1/((len(X)//4)*2*np.pi*np.log(inc)**2)
    h=vio*g-1/Airecell
    k=p*h+1/Airecell
    coef3=(f*ros)**2/k
    coef2=f*ros*h/k
    coef1=h**2/k
    return(beta(X)*coef1,beta(X)*coef2,beta(X)*coef3)  

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
    xa = np.linspace(-10, 10, 10)
    xo = np.linspace(-10, 10, 10)
    coef1=0
    coef2=0
    coef3=0
    for k in range(len(resfinal)):
        #print(k)
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








def simu(N):
    vectp=[]
    vects=[]
    matcovtot=np.zeros((N,3))
    for i in range(1,N+1):
        b=process.proctotal(T,n_init,d, Delta, generatesituation, move, beta, delta, tau, alpha, birthkernel, deathkernel, transitionkernel) 
        (resfinal,TpsSauts,tabecarts,track,compteurs,tracknaissance, trackmort, nx, avtnx, tpsnx) =b
        (traj, tracktraj,trajmutantes,trajdec,tracktrajdec,couleurstrajdec, restronque,restottronque, tracktottronque)=draw.extract_traj(tabecarts, resfinal, track, 0, intertps)
        
        [pmax, smax], ma=maxvrs(avtnx, nx)
        
        a=matcovarempfin(pmax, smax,restottronque)
         
        vectp+=[pmax]
        vects+=[smax]
        matcovtot[i-1,0]=a[0]
        matcovtot[i-1,1]=a[1]
        matcovtot[i-1,2]=a[2]  
    return(vectp, vects,matcovtot)


"""
Main: Simulation
"""



#General parameters

Airecell=400 #surface of the cell [-10, 10]x[-10, 10]
T=40 #final time of the simulation
n_init= 20 # number of particles at initial time
d=0.01 #size of discretization pace between 2 images
Delta=0.2 # taumax of the alogrithm

theta=0.8 #parameter of variance in the birth kernel
q=0.4 #parameter that rules the uniform birth in the birth kernel

s1lang=0.1 #diffusion coefficient of the brownian move


intertps=0.14 # extraction each 0.14ms (intertps=d for full extraction)

xcontour=[-10,10,10,-10, -10]
ycontour=[-10,-10, 10, 10, -10]


"""to generate a simulation and plot the heat map:"""
b=process.proctotal(T,n_init,d, Delta, generatesituation, move, beta, delta, tau, alpha, birthkernel, deathkernel, transitionkernel) 
(resfinal,TpsSauts,tabecarts,track,compteurs,tracknaissance, trackmort, nx, avtnx, tpsnx) =b


(traj, tracktraj,trajmutantes,trajdec,tracktrajdec,couleurstrajdec, restronque,restottronque, tracktottronque)=draw.extract_traj(tabecarts, resfinal, track, 0, intertps)

#to draw the trajectories
#draw.traj(trajdec,couleurstrajdec, xcontour, ycontour)  


p = np.linspace(0.1, 0.8, 20)
s = np.linspace(1.5, 4.5, 20)
P, S = np.meshgrid(p, s)
Z=tracerlogvrsdim2(P,S, avtnx, nx)

[pmax, smax], ma=maxvrs(avtnx, nx)

a=matcovarempfin(pmax, smax,restottronque)

#to draw the heat map
fig, ax = plt.subplots()
pc = ax.pcolormesh(P, S, Z, cmap='jet', shading='gouraud',vmin=-ma-30, vmax = -ma)
ax.set_title('heat map of log-likelihood of simu \n'+r'$[\hat p, \hat \sigma] = $'+ str([float(format(pmax,'.3e')),float(format(np.log(smax),'.3e'))]))
ax.scatter([pmax],[smax], marker='x', color='black')
fig.colorbar(pc)
cov=np.array([[a[0],a[1]],[a[1],a[2]]])
invcov=np.linalg.inv(cov)
plot_confidence_ellipse([pmax, smax],invcov,0.95,ax,edgecolor='black', fill=0)






#to run 50 simulations and plot the argmax on a plane as well as the mean confidence ellipse 
vectp, vects,matcovtot=simu(50)
pmoy=np.mean(vectp)
smoy=np.mean(vects)
coeff1=np.mean(matcovtot[:,0])
coeff2=np.mean(matcovtot[:,1])
coeff3=np.mean(matcovtot[:,2])
moycov=np.array([[coeff1,coeff2],[coeff2,coeff3]])
fig, ax = plt.subplots()

invmoycov=np.linalg.inv(moycov)
ax.set_title('Estimation results for '+r'$(p, log( \sigma)) = (0.4, 2.225)$'+' by MLE,\n based on 500 simulations and in red 95% Gaussian confidence ellipsoid')
ax.scatter(vectp, vects, marker='x', color='royalblue')
ax.scatter([pmoy],[smoy], marker='x', color='red')
plot_confidence_ellipse([pmoy, smoy],invmoycov,0.95,ax,edgecolor='red', fill=0)
plt.show()      
       
