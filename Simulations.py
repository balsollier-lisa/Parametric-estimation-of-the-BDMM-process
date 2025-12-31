
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: lisabalsollier
"""

"""
This code allows to generate a BDM as in the article.
It relies on the general functions in process_functions.py 
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.optimize as op
from matplotlib.patches import Ellipse

from Codes import process_functions as process
from Codes import draw_functions as draw



"""
Definition of auxiliary functions used in the main function process.proctotal 
to generate simulations.
"""


#this function allows to say whether a point is in the cell
def beincell(x):
    if np.abs(x[0])>10:
        return(0)
    else : 
        return(1)
  


#this function generates a random initial situation
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
    C : ARRAY
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


#this function generates a browniane move with diffusion coefficient 
# s1lang or s1rab with as initial position the point (xi,yi,ri,ci)
def brownianmove(xi,yi,ri,ci,ecarts,N,s1lang,s1rab):
    if ri==0 :
        normx=stats.norm.rvs(loc=0, scale=s1lang*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s1lang*np.sqrt(np.array(ecarts)))
        xx=np.cumsum(np.concatenate((np.array([xi]),normx)))
        yy=np.cumsum(np.concatenate((np.array([yi]),normy)))
        co=[[xx[m],yy[m]] for m in range(len(xx))]
        ide=np.argwhere(np.array([beincell(co[p]) for p in range(len(co))])==0)
        while len(ide)!=0:
            k=ide[0][0]
            normx[k-1]=stats.norm.rvs(loc=0, scale=s1lang*np.sqrt(ecarts[k-1]))
            normy[k-1]=stats.norm.rvs(loc=0, scale=s1lang*np.sqrt(ecarts[k-1]))
            xx=np.cumsum(np.concatenate((np.array([xi]),normx)))
            yy=np.cumsum(np.concatenate((np.array([yi]),normy)))
            co=[[xx[m],yy[m]] for m in range(len(xx))]
            ide=np.argwhere(np.array([beincell(co[p]) for p in range(len(co))])==0)
    else :
        normx=stats.norm.rvs(loc=0, scale=s1rab*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s1rab*np.sqrt(np.array(ecarts)))
        xx=np.cumsum(np.concatenate((np.array([xi]),normx)))
        yy=np.cumsum(np.concatenate((np.array([yi]),normy)))
        co=[[xx[m],yy[m]] for m in range(len(xx))]
        ide=np.argwhere(np.array([beincell(co[p]) for p in range(len(co))])==0)
        while len(ide)!=0:
            k=ide[0][0]
            normx[k-1]=stats.norm.rvs(loc=0, scale=s1rab*np.sqrt(ecarts[k-1]))
            normy[k-1]=stats.norm.rvs(loc=0, scale=s1rab*np.sqrt(ecarts[k-1]))
            xx=np.cumsum(np.concatenate((np.array([xi]),normx)))
            yy=np.cumsum(np.concatenate((np.array([yi]),normy)))
            co=[[xx[m],yy[m]] for m in range(len(xx))]
            ide=np.argwhere(np.array([beincell(co[p]) for p in range(len(co))])==0)
    cc=[ci]*(N)
    rr=[ri]*(N)
    return(xx,yy,rr,cc)


#this function generates a subdiffusive move with diffusion coefficient 
#s3lang or s3rab and attraction coefficient lalang or larab 
#with as initial position the point (xi,yi,ri,ci)
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
            ny=yi+np.exp(-lalang*tps[n+1])*(np.sum((expla*normy)[:n+1]))
            if beincell([nx,ny])==1 :
                xx+=[nx]
                yy+=[ny]
                n=n+1
            else:
                normx[n]=stats.norm.rvs(loc=0, scale=s3lang*np.sqrt(ecarts[n]))
                normy[n]=stats.norm.rvs(loc=0, scale=s3lang*np.sqrt(ecarts[n]))
    else :
        normx=stats.norm.rvs(loc=0, scale=s3rab*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s3rab*np.sqrt(np.array(ecarts)))
        xx=[xi]
        yy=[yi]
        n=0
        expla=[(np.exp(larab*tps[i+1])+np.exp(larab*tps[i]))/2 for i in range(len(tps)-1)]
        while n <= (len(ecarts)-1):
            nx=xi+np.exp(-larab*tps[n+1])*(np.sum((expla*normx)[:n+1]))
            ny=yi+np.exp(-larab*tps[n+1])*(np.sum((expla*normy)[:n+1]))
            if beincell([nx,ny])==1 :
                xx+=[nx]
                yy+=[ny]
                n=n+1
            else:
                normx[n]=stats.norm.rvs(loc=0, scale=s3rab*np.sqrt(ecarts[n]))
                normy[n]=stats.norm.rvs(loc=0, scale=s3rab*np.sqrt(ecarts[n]))
    cc=[ci]*(N)
    rr=[ri]*(N)
    return(xx,yy,rr,cc)


#this function generates a drifted Brownian motion with diffusion coefficient 
#s2lang or s2rab and drift coefficient drlang or drrab 
#with initial position the point (xi,yi,ri,ci)
def superdiffusifmove(xi,yi,ri,ci,ecarts,N,s2lang,s2rab,drlang, drrab):
    if ri==0:
        normx=stats.norm.rvs(loc=0, scale=s2lang*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s2lang*np.sqrt(np.array(ecarts)))
        xx=[xi]
        yy=[yi]
        angle=ci
        cc=[ci]*(N)
        m=0
        while m <= (len(ecarts)-1):
            nx=xx[-1]+normx[m]+drlang*np.cos(angle)*ecarts[m]
            ny=yy[-1]+normy[m]+drlang*np.sin(angle)*ecarts[m]
            if beincell([nx,ny])==1 :
                xx+=[nx]
                yy+=[ny]
                m=m+1
            else :
                angle=stats.uniform.rvs(0,2*np.pi)
                normx[m]=stats.norm.rvs(loc=0, scale=s2lang*np.sqrt(ecarts[m]))
                normy[m]=stats.norm.rvs(loc=0, scale=s2lang*np.sqrt(ecarts[m]))
                cc[m:]=[angle]*len(cc[m:])
    else :
        normx=stats.norm.rvs(loc=0, scale=s2rab*np.sqrt(np.array(ecarts)))
        normy=stats.norm.rvs(loc=0, scale=s2rab*np.sqrt(np.array(ecarts)))
        xx=[xi]
        yy=[yi]
        angle=ci
        cc=[ci]*(N)
        m=0
        while m <= (len(ecarts)-1):
            nx=xx[-1]+normx[m]+drrab*np.cos(angle)*ecarts[m]
            ny=yy[-1]+normy[m]+drrab*np.sin(angle)*ecarts[m]
            if beincell([nx,ny])==1 :
                xx+=[nx]
                yy+=[ny]
                m=m+1
            else :
                angle=stats.uniform.rvs(0,2*np.pi)
                normx[m]=stats.norm.rvs(loc=0, scale=s2rab*np.sqrt(ecarts[m]))
                normy[m]=stats.norm.rvs(loc=0, scale=s2rab*np.sqrt(ecarts[m]))
                cc[m:]=[angle]*len(cc[m:])
    rr=[ri]*(N)
    return(xx,yy,rr,cc)


#this function generates the motion of a particle up to time t according to its coordinate c
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
    c : ARRAY
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
    res= np.zeros((N,4*P))
    for i in range (P) :
        if c[i]==1 :
            xx,yy,rr,cc=brownianmove(x[i],y[i],r[i],c[i],ecarts,N, s1lang,s1rab)
        elif c[i]==3 :
            xx,yy,rr,cc=subdiffusifmove(x[i],y[i],r[i],c[i],ecarts,tps,N,s3lang,s3rab,lalang, larab)
        else :
            xx,yy,rr,cc=superdiffusifmove(x[i],y[i],r[i],c[i],ecarts,N,s2lang,s2rab, drlang, drrab)
        res[:,4*i]=xx
        res[:,4*i+1]=yy
        res[:,4*i+3]=cc
        res[:,4*i+2]=rr
    return(res,ecarts,tps)



#birth intensity : constant and equal to 10
def beta(x) : 
    return(10)
    

#death intensity : proportional to the number of particle
def delta(x): 
    if process.n(x)==1:
        return(0)
    else :
        return(0.2*process.n(x))
        
    
#mutation intensity : constant and equal to 16/167.86
def tau(x) : 
    return(16/167.86)
    
    
    
def alpha(x) :
    return(beta(x)+delta(x)+tau(x))  



#birth kernel : as in Section 4.1 of the article
def birthkernel(depart):
    """
    Parameters
    ----------
    depart : ARRAY
        Array that contains the coordinates of the point configuration present
        just before the new birth (in the order abscissa, ordinate, type, regime).
    
    q and theta : global parameters (see at the end of the program for their values) 
    q : probability of colocalisation
    theta : standard deviation of the Gaussian distribution when colocalisation

    Returns
    -------
    The coordinates of the new point (in order: abscissa, ordinate, type, regime).
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





#death kernel : uniform on present particle
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



#transition kernel : according a transition matrix
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
Main: Simulation
"""

#General parameters
T=40 #final time of the simulation
n_init= 60 # number of particles at initial time
d=0.01 #size of discretization between 2 images
Delta=0.2 # taumax of the algorithm

theta=0.3 #parameter of standard deviation in the birth kernel
q=0.2 #parameter that rules the uniform birth in the birth kernel

s1lang=0.4 #diffusion coefficient of the brownian move
s2lang=0.4 #diffusion coefficient of the drifted brownian motion
drlang=0.4 #drift coefficient of the drifted brownian motion
s3lang=0.2 #diffusion coefficient in the subdiffusive move (Orstein-Uhlenbeck)
lalang=9 #attraction coefficien in the subdiffusive move (Orstein-Uhlenbeck)

s1rab=0.4 #diffusion coefficient of the brownian move
s2rab=0.4 #diffusion coefficient of the drifted brownian motion
drrab=0.4 #drift coefficient of the drifted brownian motion
s3rab=0.2 #diffusion coefficient in the subdiffusive move (Orstein-Uhlenbeck)
larab=9 #attraction coefficien in the subdiffusive move (Orstein-Uhlenbeck)


intertps=0.14 # extraction each 0.14ms (intertps=d for full extraction)

xcontour=[-10,10,10,-10, -10]
ycontour=[-10,-10, 10, 10, -10]


#One simulation (5 seconds)
b=process.proctotal(T,n_init,d, Delta, generatesituation, move, beta, delta, tau, alpha, birthkernel, deathkernel, transitionkernel) 
(resfinal,TpsSauts,tabecarts,track,compteurs,tracknaissance, trackmort, nx, avtnx, tpsnx) =b

#Extraction of characteristics
(traj, tracktraj,trajmutantes,trajdec,tracktrajdec,couleurstrajdec, restronque,restottronque, tracktottronque)=draw.extract_traj(tabecarts, resfinal, track, 0, intertps)

#Drawing the trajectories
draw.traj(trajdec,couleurstrajdec, xcontour, ycontour)



###NOT RUN : saving 500 simulations
###
# import pickle
# from joblib import Parallel, delayed
# import multiprocessing
# from tqdm import tqdm
# 
# def f(i):
#   b=process.proctotal(T,n_init,d, Delta, generatesituation, move, beta, delta, tau, alpha, birthkernel, deathkernel, transitionkernel) 
#   nom=f"Synthetic_data/results_simu{i}.pickle"
#   with open(nom, "wb") as f:
#       pickle.dump(b,f)
# 
# 
# results = Parallel(n_jobs=6)(
#     delayed(f)(i) for i in tqdm(range(1,501))
# )


## Visualizing saved simulations
## Example for the first simulation 
import pickle

with open("Synthetic_data/results_simu1.pickle", "rb") as f:
    data = pickle.load(f)

(resfinal,TpsSauts,tabecarts,track,compteurs,tracknaissance, trackmort, nx, avtnx, tpsnx) =data
#Extraction of characteristics
(traj, tracktraj,trajmutantes,trajdec,tracktrajdec,couleurstrajdec, restronque,restottronque, tracktottronque)=draw.extract_traj(tabecarts, resfinal, track, 0, intertps)
#Drawing the trajectories
draw.traj(trajdec,couleurstrajdec, xcontour, ycontour)
