# Syntetic Data

This folder contains the results of 500 simulations of a BDMM process, following  the specifications described in Section 4.1 of the article, together with the corresponding maximum-likelihood parameter estimates for each of them.

In detail:

- `results_simu1.pickle` to `results_simu500.pickle` contain the result of the 500 simulations. The data in each of these files are similar as the output of the simulation of a BDMM, as done in the main script `Simulations.py`. 

- `results_p.pickle` contains the 500 estimations of the parameter p involved in the birth kernel, for the 500 simulations above.

- `results_exp(sigma).pickle` contains the 500 estimations of the parameter exp(sigma) involved in the birth kernel, for the 500 simulations above.

- `results_invmatcov.pickle` contains the 500 estimations of the inverse asymptotic covariance matrix of the estimators of (p,exp(sigma)), for the 500 simulations above.


Regarding the specifications of the model, in a nutshell:

- There are two types of particles: Langerin and Rab11

- Three possible motion regimes: Brownian, or subdiffusive (Ornstein-Uhlenbeck), or superdiffusive (Brownian with drift)

- The Brownian motions have a diffusion coefficient equal to 0.4

- The Ornstein-Uhlenbeck motions have a diffusion coefficient equal to 0.2 and an attraction coefficient equal to 9

- The drifted Brownian motions have a diffusion coefficient equal to 0.4 and a drift coefficient equal to 0.4

- The birth intensity is constant and equal to 10

- The death intensity is constant and equal to 16/167.86

- The mutation intensity is constant and equal to 0.2 n(x)

- The death kernel is uniform over the existing particles

- The birth kernel is the one presented in Section 4.1 of the article, with proportion of colocalisation p=0.2 and variance of colocalisation sigma=0.3 so that exp(sigma)=1.35

- The simulation is done on the square [-10,10]^2

- The final time of simulation is T=40ms

- The discretisation step for the simulation is 0.01ms

- The data is recorded every 0.14ms. This represents the intertime between two frames


