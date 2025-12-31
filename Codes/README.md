# Codes

This folder contains the following source files:

- ``process_functions.py`` gathers all Python functions necessary to generate a BDMM process. The main function, named `proctotal`, takes in argument all the characteristics of the process (final time of simulation, move, intensities of births, of deaths and of transformations, transition probabilities for the births, for the deaths and for the transformations) and returns the coordinates of the generated particles at each time of the simulation (over a fine temporal grid). These programs are needed to run the main script `Simulations.py`.

- `draw_functions.py` contains several functions that take as input the output of the `proctotal` function (see above) to plot the generated trajectories, the histograms of their lengths, the boxplots of the number of particles per frame, or to display a movie of the simulated process. These functions are required to run the main script `Simulations.py`.


- `estimation_functions.py` contains the functions needed to compute the log-likelihood of the birth kernel, as parameterised in the article, and to deduce the MLE and its associated covariance matrix. These functions are required to run the main scripts `Synthetic_data_analysis.py` and `Data_analysis.py`.
