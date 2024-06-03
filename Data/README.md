# Data

In this folder, you find two folders:

In the folder "Characteristics", you find :

- "M10_Rab11mCherry_09_GaussianWeights.mp4" and "M10_LangerinYFp_09_GaussianWeights.mp4" are the raw sequences for each type of proteins.

- "Utrack_Langerin_GaussianWeights.mp4" and "Utrack_Rab11_GaussianWeights.mp4" are the results of proteins tracking (by the U-track algorithm) for the previous sequences.

- "real-data-video.mp4" gathers the two previous sequences by representing  Langerin vesicles as circles and Rab11 vesicles as triangles. Moreover, each particle is coloured with respect to its estimated motion regime: Brownian in blue, superdiffusive in red and subdiffusive in green. 

- The two .csv files are the results of the tracking procedure for each type of protein. Each file contains the spatial coordinates of the particles, at each frame, the index of the trajectory to which they belong to and their estimated motion regime (1: Brownian, 2: superdiffusive, 3: subdiffusive).

- The two images "trajectories_Langerin.png" and "trajectories_Rab11.png" depict all trajectories of Langerin and  Rab11 proteins, tracked over the image sequences, with the same color label as above (blue: Brownian, red: superdiffusive, green: subdiffusive).  

In the folder "Maximum-Likelihood-Estimation", you find :

- "maximum_likelihood_estimation.py" calls the data in the "Characteristics" folder and uses the programs in "../Code" to calculate the log-likelihood for the parametric model studied in the article (Section 4) for this dataset. It also contains programs for displaying this log-likelihood as a heat map, along with the confidence ellipsoid centered at the MLE. 


- “heat_map_with_ellipse.png” contains the heat map of the log-likelihood for our dataset, with the estimated 95% confidence ellipse overlaid in black, centered at the MLE (marked with a 'x'). This plot is obtained by "maximum_likelihood_estimation.py".
