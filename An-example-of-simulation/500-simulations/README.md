# Some-simulation-results 

This folder contains the results of 500 simulations. Each simulation is carried out using the following model:
- a single type of particle
- all the particles follow a Brownian motion with a diffusion coefficient equal to 0.1
- the birth intensity is constant and equal to 10
- the intensity of death is constant and equal to 10
- the mutation intensity is constant and equal to 0 (no mutation possible)
- the death nucleus is uniform over the particles present
- the birth kernel is that presented in section 5.1 of the article


Theses simulations are generated with the parameters :

- T=40 #final time of the simulation
- n_init= 20 # number of particles at initial time
- sigma=0.8 or log(sigma)=2.226 #parameter of variance in the birth kernel
- p=0.4  #parameter that rules the uniform birth in the birth kernel

In each sub-folder, you can find:

- "trajectories.png": a plot of the trajectories of the particle
- "heat_map.png": heat map of the log-likelihood, with the estimated 95% confidence ellipse overlaid in black, centered at the maximum likelihood (marked with a 'x')
- "caracteristics.txt": value of the argmax of log-likelihood and valu of the covariance matrix 

