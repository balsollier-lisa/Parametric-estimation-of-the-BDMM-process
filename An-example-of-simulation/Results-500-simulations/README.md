# Results-500-simulations 

This folder contains:


- "results_p.pickle" is a python file that contains all the values of p (see Section 5.1) in the argmax of the log-likelihood for the 500 simulations of the "../500-simulations" folder.

- "results_log(sigma).pickle" is a python file that contains all the values of log(sigma) (see Section 5.1) in the argmax of the log-likelihood for the 500 simulations of the "../500-simulations" folder.

- "results_invmatcov.pickle" is a python file that contains all the value of the estimated J matrix (the inverse of which is the asymptotic covariance matrix of the MLE) for the 500 simulations of he "../500-simulations" folder.


- "plot_argmax_with_ellipse.png" the result of all estimations of (p, log(sigma)) = (0.2 , 1.34) over the 500 simulations of the "../500-simulations" folder, along with, in red, the average confidence ellipsoid (constructed as the Gaussian ellipsoid centered at the mean of all estimations and with covariance the mean of all estimated covariance matrix). 
