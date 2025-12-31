import pickle
import importlib
from Codes import estimation_functions as estimation
from Codes import draw_functions as draw


#Global parameters
areacell=400 #area of the cell
intertps=0.14 #inter-time between two observations (total time is T=40)
plbirth=[1/3,1/3,1/3] #probability of marks when no-colocalisation
def beincell(x): #function testing if a point is in the cell
    if np.abs(x[0])>10:
        return(0)
    else : 
        return(1)

#Estimation for the first simulation 
with open("Synthetic_data/results_simu1.pickle", "rb") as f:
    data = pickle.load(f)

(resfinal,TpsSauts,tabecarts,track,compteurs,tracknaissance, trackmort, nx, avtnx, tpsnx) =data

#Estimation by maximum likelihood of p and theta
[pmax, smax], ma=estimation.maxvrs(avtnx, nx,areacell,plbirth=plbirth)
print(pmax, smax)


#Estimation of the J matrix (the inverse of the asymptotic covariance matrix of the MLE)  
#First extracting characterstics of the trajectories
(traj, tracktraj,trajmutantes,trajdec,tracktrajdec,couleurstrajdec, restronque,restottronque, tracktottronque)=draw.extract_traj(tabecarts, resfinal, track, 0, intertps)
#Then estimating J
a=estimation.matJ(pmax, smax,restottronque,areacell,intertps=0.14,beta=len(tpsnx)/40,plbirth=plbirth,beincell=beincell)
print(a) #J(1,1), J(1,2)=J(2,1), J(2,2)


#Plot of the log-likelihood heat map along with the confidence ellipsoid, centered at the maximum likelihood estimator
p = np.linspace(0.05, 0.7, 20)
s = np.linspace(1.1, 4, 20)
P, S = np.meshgrid(p, s)
Z=estimation.logvrsdim2(P,S, avtnx, nx,areacell,plbirth)

invcov=np.array([[a[0],a[1]],[a[1],a[2]]])
cov=np.linalg.inv(invcov)

fig, ax = plt.subplots()
pc = ax.pcolormesh(P, S, Z, cmap='jet', shading='gouraud',vmin=-ma-70, vmax = -ma)
estimation.plot_confidence_ellipse([pmax, smax],cov,0.95,ax,edgecolor='black', fill=0)
ax.set_title('Heat map of log-likelihood of simu 1 \n'+r'$[\hat p, \widehat{log(\sigma)}] = $'+ str([float(format(pmax,'.3e')),float(format(smax,'.3e'))]))
ax.scatter([pmax],[smax], marker='x', color='black')
fig.colorbar(pc)
plt.show()
plt.close()


########################################
###### NOT RUN Estimation over all 500 simulations
###### and saving the results

# from joblib import Parallel, delayed
# import multiprocessing
# from tqdm import tqdm
# 
# def f(i):
#   nom=f"Synthetic_data/results_simu{i}.pickle"
#   with open(nom, "rb") as f:
#     data = pickle.load(f)
#   (resfinal,TpsSauts,tabecarts,track,compteurs,tracknaissance, trackmort, nx, avtnx, tpsnx) =data
#   (traj, tracktraj,trajmutantes,trajdec,tracktrajdec,couleurstrajdec, restronque,restottronque, tracktottronque)=draw.extract_traj(tabecarts, resfinal, track, 0, intertps)
#   [pmax, smax], ma=estimation.maxvrs(avtnx, nx,areacell,plbirth=plbirth)
#   a=estimation.matJ(pmax, smax,restottronque,areacell,intertps=0.14,beta=len(tpsnx)/40,plbirth=plbirth,beincell=beincell)
#   return([pmax,smax,a])
# 
# results = Parallel(n_jobs=6)(
#     delayed(f)(i) for i in tqdm(range(1,501))
# )
# 
# pvec=[]
# svec=[]
# invmatcov=[]
# for i in range(len(results)):
#   pvec+=[results[i][0]]
#   svec+=[results[i][1]]
#   invmatcov+=[results[i][2]]
# 
# with open("Synthetic_data/results_p.pickle", "wb") as f:
#     pickle.dump(pvec,f)
# with open("Synthetic_data/results_exp(sigma).pickle", "wb") as f:
#     pickle.dump(svec,f)
# with open("Synthetic_data/results_invmatcov.pickle", "wb") as f:
#     pickle.dump(invmatcov,f)


#### Loading the estimation results for the 500 simulations
with open("Synthetic_data/results_p.pickle", "rb") as f:
    pvec = pickle.load(f)
with open("Synthetic_data/results_exp(sigma).pickle", "rb") as f:
    svec = pickle.load(f)
with open("Synthetic_data/results_invmatcov.pickle", "rb") as f:
    invmatcov = pickle.load(f)



####Plotting all estimations around the true value (0.2,np.exp(0.3)=1.35)
fig, ax = plt.subplots()
ax.scatter(pvec, svec, marker='x', color='royalblue')
plt.axhline(y=np.exp(0.3),color='grey',linestyle='--')
plt.axvline(x=0.2,color='grey',linestyle='--')
plt.show()
plt.close()
  



####Estimation of the probability p : histogram and coverage of confidence intervals
plt.hist(pvec, bins=10, edgecolor='black')   
plt.axvline(x=0.2,color='grey',linestyle='--')
#plt.title("Histogram of $\hat p$")
plt.show()
plt.close()

coverage=[]
for i in range(len(pvec)):
  est=np.asarray(pvec[i])
  a=invmatcov[i]
  diff=est-np.asarray(0.2)
  d=np.abs(diff*np.sqrt(a[0]))
  coverage+=[d <= 1.96]
np.mean(coverage)#0.95


####Estimation of theta (=exp(sigma) where sigma=sd) : histogram and coverage of confidence intervals
plt.hist(svec, bins=10, edgecolor='black')   
plt.axvline(x=np.exp(0.3),color='grey',linestyle='--')
#plt.title(r"Histogram of $\hat\theta$")
plt.show()
plt.close()

coverage=[]
for i in range(len(svec)):
  est=np.asarray(svec[i])
  a=invmatcov[i]
  diff=est-np.asarray(np.exp(0.3))
  d=np.abs(diff*np.sqrt(a[2]))
  coverage+=[d <= 1.96]
np.mean(coverage)#0.946



###Coverage of the confidence ellipsoid
c = -2 * np.log(1 - 0.95)
coverage=[]
for i in range(len(pvec)):
  pmax =pvec[i]
  smax=svec[i]
  a=invmatcov[i]
  est=np.asarray([pmax,smax])
  diff=est-np.asarray([0.2,np.exp(0.3)])
  invcov=np.array([[a[0],a[1]],[a[1],a[2]]])
  d2=diff @ invcov @ diff.T
  coverage+=[d2 <= c]
np.mean(coverage) #0.96 
