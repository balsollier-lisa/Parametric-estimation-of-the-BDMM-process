# Results-500-simulations 

This folder contains:


- "results_p.pickle" is a python file that contains all the values of p in the argmax of the log-likelihood for the 500 simulations of 500-simulations folder.

- "results_log(sigma).pickle" is a python file that contains all the values of log(sigma) in the argmax of the log-likelihood for the 500 simulations of 500-simulations folder.

- "results_matcov.pickle" is a python file that contains all the value of the J matrix for the 500 simulations of 500-simulations folder.


- "plot_argmax_with_ellipse.png" the result of the estimation of (p, log(sigma)) = (0.2 , 1.34) with 500 simulations of 500-simulations folder. The mean of the 95% confidence ellipses estimated (constructed from the mean of the estimates of the covariance matrix J on all the simulations and centered in the mean of the estimators) is plotted in red.