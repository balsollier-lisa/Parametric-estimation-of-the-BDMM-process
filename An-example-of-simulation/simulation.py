
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
        
        R+=[np.random.choice([0,1])]
        c=np.random.choice([1,2,3])
        if c==2 :
            C+=[stats.uniform.rvs(0,2*np.pi)]
        else :
            C+=[c]
        track=[[i for i in range(1,M+1)]]
        cpttrack=M
    return(X,Y,R,C,track,cpttrack)

def brownianmove(xi,yi,ri,ci,ecarts,N,s1lang,s1rab):
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
    else :
        normx=stats.norm.rvs(loc=0, scale=s1rab*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s1rab*np.sqrt(np.array(ecarts)))
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
            normx[k-1]=stats.norm.rvs(loc=0, scale=s1rab*np.sqrt(ecarts[k-1]))
            normy[k-1]=stats.norm.rvs(loc=0, scale=s1rab*np.sqrt(ecarts[k-1]))
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


"""this program generates a subdiffusive move with diffusion coefficient 
s3lang or s3rab and attraction coefficient lalang or larab 
with as initial position the point (xi,yi,ri,ci)"""
def subdiffusifmove(xi,yi,ri,ci,ecarts,tps,N,s3lang,s3rab, lalang, larab):
    if ri==0:
        normx=stats.norm.rvs(loc=0, scale=s3lang*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s3lang*np.sqrt(np.array(ecarts)))
        xx=[xi]
        yy=[yi]
        n=0
        expla=[(np.exp(lalang*tps[i+1])+np.exp(lalang*tps[i]))/2 for i in range(len(tps)-1)]
        while n <= (len(ecarts)-1):
            nx=xi+np.exp(-lalang*tps[n+1])*(np.sum((expla*normx)[:n+1]))
            #print(nx)
            ny=yi+np.exp(-lalang*tps[n+1])*(np.sum((expla*normy)[:n+1]))
            #print(ny)
            if beincell([nx,ny])==1 :
                xx+=[nx]
                yy+=[ny]
                n=n+1
                #print(n)
            else:
                normx[n]=stats.norm.rvs(loc=0, scale=s3lang*np.sqrt(ecarts[n]))
                normy[n]=stats.norm.rvs(loc=0, scale=s3lang*np.sqrt(ecarts[n]))
            #else:
               # normx[n]=-normx[n]
               # normy[n]=-normy[n]
    else :
        normx=stats.norm.rvs(loc=0, scale=s3rab*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s3rab*np.sqrt(np.array(ecarts)))
        xx=[xi]
        yy=[yi]
        n=0
        expla=[(np.exp(larab*tps[i+1])+np.exp(larab*tps[i]))/2 for i in range(len(tps)-1)]
        while n <= (len(ecarts)-1):
            nx=xi+np.exp(-larab*tps[n+1])*(np.sum((expla*normx)[:n+1]))
            #print(nx)
            ny=yi+np.exp(-larab*tps[n+1])*(np.sum((expla*normy)[:n+1]))
            #print(ny)
            if beincell([nx,ny])==1 :
                xx+=[nx]
                yy+=[ny]
                n=n+1
                #print(n)
            else:
                normx[n]=stats.norm.rvs(loc=0, scale=s3rab*np.sqrt(ecarts[n]))
                normy[n]=stats.norm.rvs(loc=0, scale=s3rab*np.sqrt(ecarts[n]))
            #else:
               # normx[n]=-normx[n]
               # normy[n]=-normy[n]
    cc=[ci]*(N)
    rr=[ri]*(N)
    return(xx,yy,rr,cc)


"""this program generates a subdiffusive move with diffusion coefficient 
s3lang or s3rab and drift coefficient drlang or drrab 
with as initial position the point (xi,yi,ri,ci)"""
def superdiffusifmove(xi,yi,ri,ci,ecarts,N,s2lang,s2rab,drlang, drrab):
    if ri==0:
        normx=stats.norm.rvs(loc=0, scale=s2lang*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s2lang*np.sqrt(np.array(ecarts)))
        xx=[xi]
        yy=[yi]
        angle=ci
        cc=[ci]*(N)
        m=0
        #q=0
        while m <= (len(ecarts)-1):
            #if q==m+10:
             #   print('nul')
             #   return('nul')
            nx=xx[-1]+normx[m]+drlang*np.cos(angle)*ecarts[m]
            ny=yy[-1]+normy[m]+drlang*np.sin(angle)*ecarts[m]
            #print([nx,ny])
            if beincell([nx,ny])==1 :
                xx+=[nx]
                yy+=[ny]
                m=m+1
                #print(m)
            else :
                angle=stats.uniform.rvs(0,2*np.pi)
                normx[m]=stats.norm.rvs(loc=0, scale=s2lang*np.sqrt(ecarts[m]))
                normy[m]=stats.norm.rvs(loc=0, scale=s2lang*np.sqrt(ecarts[m]))
                cc[m:]=[angle]*len(cc[m:])
                #print('ici')
    else :
        normx=stats.norm.rvs(loc=0, scale=s2rab*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s2rab*np.sqrt(np.array(ecarts)))
        xx=[xi]
        yy=[yi]
        angle=ci
        cc=[ci]*(N)
        m=0
        #q=0
        while m <= (len(ecarts)-1):
            #if q==m+10:
             #   print('nul')
             #   return('nul')
            nx=xx[-1]+normx[m]+drrab*np.cos(angle)*ecarts[m]
            ny=yy[-1]+normy[m]+drrab*np.sin(angle)*ecarts[m]
            #print([nx,ny])
            if beincell([nx,ny])==1 :
                xx+=[nx]
                yy+=[ny]
                m=m+1
                #print(m)
            else :
                angle=stats.uniform.rvs(0,2*np.pi)
                normx[m]=stats.norm.rvs(loc=0, scale=s2rab*np.sqrt(ecarts[m]))
                normy[m]=stats.norm.rvs(loc=0, scale=s2rab*np.sqrt(ecarts[m]))
                cc[m:]=[angle]*len(cc[m:])
                #print('ici')
    rr=[ri]*(N)
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
            xx,yy,rr,cc=brownianmove(x[i],y[i],r[i],c[i],ecarts,N, s1lang,s1rab)
            #print('br2')
        elif c[i]==3 :
            #print('sub1')
            #xx,yy,rr,cc=subdiffusifmove(x[i],y[i],r[i],c[i],ecarts,tps,N,np.sqrt(np.random.choice(sblangdiff)),np.sqrt(np.random.choice(sbrabdiff)),np.random.choice(sblanglb), np.random.choice(sblanglb))
            #xx,yy,rr,cc=subdiffusifmove(x[i],y[i],r[i],c[i],ecarts,tps,N,np.sqrt(np.random.choice(sblangdiff)),np.sqrt(np.random.choice(sbrabdiff)),lalang, larab)
            xx,yy,rr,cc=subdiffusifmove(x[i],y[i],r[i],c[i],ecarts,tps,N,s3lang,s3rab,lalang, larab)
            #print('sub2')
        else :
            #print('sup1')
            #xx,yy,rr,cc=superdiffusifmove(x[i],y[i],r[i],c[i],ecarts,N,np.sqrt(np.random.choice(splangdiff))/2,np.sqrt(np.random.choice(sprabdiff))/2, np.random.choice(splangdrift), np.random.choice(sprabdrift))
            #xx,yy,rr,cc=superdiffusifmove(x[i],y[i],r[i],c[i],ecarts,N,np.sqrt(np.random.choice(splangdiff)),np.sqrt(np.random.choice(sprabdiff)), drlang, drrab)
            xx,yy,rr,cc=superdiffusifmove(x[i],y[i],r[i],c[i],ecarts,N,s2lang,s2rab, drlang, drrab)
            #print('sup2')
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
        return(0.2*process.n(x))
        #return(0.12*nl(x)+0.14*nr(x))
        #return(0.15*process.nlb(x)+0.1*process.nlsp(x)+0.058*process.nlsb(x)+0.18*process.nrb(x)+0.22*process.nrsp(x)+0.07*process.nrsb(x))
    
    
"""mutation intensity : equal to zero : no transition """
def tau(x) : 
    return(16/167.86)
    #return(0)
    
    
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
    r=stats.uniform.rvs(0,1)
    if r<(1/2):
        R=1
        X=stats.uniform.rvs(loc=-10,scale=20)
        Y=stats.uniform.rvs(loc=-10,scale=20)
        while beincell([X,Y])==0 :
            X=stats.uniform.rvs(loc=-10,scale=20)
            Y=stats.uniform.rvs(loc=-10,scale=20)
        CC=np.random.choice([1,2,3])
        if CC==2 :
            C=stats.uniform.rvs(0,2*np.pi)
        else :
            C=CC
    else :
        R=0
        p=stats.uniform.rvs(0,1)
        if p<q and process.nr(depart)>0:
            P=process.nr(depart)
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
            C=depart[4*(m-1)+3]
        else :
            X=stats.uniform.rvs(loc=-10,scale=20)
            Y=stats.uniform.rvs(loc=-10,scale=20)
            while beincell([X,Y])==0 :
                X=stats.uniform.rvs(loc=-10,scale=20)
                Y=stats.uniform.rvs(loc=-10,scale=20)
            CC=np.random.choice([1,2,3])
            if CC==2 :
                C=stats.uniform.rvs(0,2*np.pi)
            else :
                C=CC
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



"""transition kernel"""
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
    x=depart
    r=stats.uniform.rvs(0,1)
    if r<(1/16):
       P=process.nr(depart)
       i=stats.randint.rvs(1,P+1)
       m=0
       s=0
       while s<i and m<len(depart)/4:
           if depart[4*m+2]== 1 :
               s+=1
           m+=1 
    else:
        P=process.nl(depart)
        i=stats.randint.rvs(1,P+1)
        m=0
        s=0
        while s<i and m<len(depart)/4:
            if depart[4*m+2]== 0 :
                s+=1
            m+=1
    if depart[4*(m-1)+3]==1:
        nt=np.random.choice([2,3],p=[1/4,3/4])
    elif depart[4*(m-1)+3]==3:
        nt=1
    else:
        nt=np.random.choice([1,3],p=[1/2,1/2])
    if nt==2:
        nt=stats.uniform.rvs(0,2*np.pi)
    x[4*(m-1)+3]=nt
    return(x)







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
    #aa=op.minimize(g,[2,0.4],bounds=[(1.8, 3.5),(0.1,1)],method='L-BFGS-B') 
    aa=op.minimize(g,[1.5,0.45],bounds=[(1.01, 2),(0.1,1)],method='L-BFGS-B') 
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
n_init= 60 # number of particles at initial time
d=0.01 #size of discretization pace between 2 images
Delta=0.2 # taumax of the alogrithm

theta=0.3 #parameter of standard deviation in the birth kernel
q=0.2 #parameter that rules the uniform birth in the birth kernel

s1lang=0.4 #diffusion coefficient of the brownian move
s2lang=0.4
drlang=0.4
s3lang=0.2 #diffusion coefficient in the subdiffusive move
lalang=9
s1rab=0.4 #diffusion coefficient of the brownian move
s2rab=0.4
drrab=0.4
s3rab=0.2 #diffusion coefficient in the subdiffusive move
larab=9


intertps=0.14 # extraction each 0.14ms (intertps=d for full extraction)

xcontour=[-10,10,10,-10, -10]
ycontour=[-10,-10, 10, 10, -10]


"""to generate a simulation and plot the heat map:"""
b=process.proctotal(T,n_init,d, Delta, generatesituation, move, beta, delta, tau, alpha, birthkernel, deathkernel, transitionkernel) 
(resfinal,TpsSauts,tabecarts,track,compteurs,tracknaissance, trackmort, nx, avtnx, tpsnx) =b


(traj, tracktraj,trajmutantes,trajdec,tracktrajdec,couleurstrajdec, restronque,restottronque, tracktottronque)=draw.extract_traj(tabecarts, resfinal, track, 0, intertps)

#to draw the trajectories
draw.traj(trajdec,couleurstrajdec, xcontour, ycontour)  
[pmax, smax], ma=maxvrs(avtnx, nx)

print(pmax, smax)



p = np.linspace(0.05, 0.7, 20)
s = np.linspace(1.2, 4.5, 20)
P, S = np.meshgrid(p, s)
Z=tracerlogvrsdim2(P,S, avtnx, nx)


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
invcovtot=np.zeros((len(matcovtot),3))
for i in range(len(matcovtot)):
    a=matcovtot[i,:]
    cov=np.array([[a[0],a[1]],[a[1],a[2]]])
    invcov=np.linalg.inv(cov)
    invcovtot[i,:]=np.array([invcov[0,0], invcov[0,1], invcov[1,1]])
coeff1mi=np.mean(invcovtot[:,0])
coeff2mi=np.mean(invcovtot[:,1])
coeff3mi=np.mean(invcovtot[:,2])
moyinvcov=np.array([[coeff1mi,coeff2mi],[coeff2mi,coeff3mi]])
fig, ax = plt.subplots()
plot_confidence_ellipse([pmoy, smoy],moyinvcov,0.95,ax,edgecolor='red', fill=0)
ax.set_title('Estimation results for '+r'$(p, log( \sigma)) = (0.2, 1.34)$'+' by MLE,\n based on 500 simulations and in red 95% Gaussian confidence ellipsoid')
ax.scatter(vectp, vects, marker='x', color='royalblue')
ax.scatter([pmoy],[smoy], marker='x', color='red')
plt.show()
  
