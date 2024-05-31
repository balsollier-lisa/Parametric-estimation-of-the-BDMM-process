# Real-dataset

In this folder, you find two folders:

In folder "Characteristics", you find :

- "M10_Rab11mCherry_09_GaussianWeights.mp4" and "M10_LangerinYFp_09_GaussianWeights.mp4" are the raw sequence for each type of proteins.

- "Utrack_Langerin_GaussianWeights.mp4" and "Utrack_Rab11_GaussianWeights.mp4" are the results of proteins tracking (by the U-track algorithm) for the previous sequences.

- "real-data-video.mp4" gathers the two previous sequences by representing  Langerin vesicles as circles and Rab11 vesicles as triangles. Moreover, each particle is coloured with respect to its estimates motion regime: Brownian in blue, superdiffusive in red and subdiffusive in green. 

- The two .csv files are the results of the tracking procedure for each type of protein. Each file contains the spatial coordinates of the particles, at each frame, the index of the trajectory to which they belong and their estimated motion regime (1: Brownian, 2: superdiffusive, 3: subdiffusive).

- The two images "trajectories_Langerin.png" and "trajectories_Rab11.png" depicts all trajectories of Langerin and  Rab11 proteins, tracked over the image sequences, with the same color label as above (blue: Brownian, red: superdiffusive, green: subdiffusive).  

In folder "Maximum-Likelihood-Estiamtion", you find :

- "maximum_likelihood_estimation.py" contains programs for calculating the log-likelihood associated with the parametric model studied in the article (Section 4) and deducing the MLE. It also contains programs for displaying this log-likelihood in the form of a heat map with the ellipse around the maximum. This file is used to generate the image “heat_map_with_ellipse.png”. 

- “heat_map_with_ellipse.png” contains the heat map of the log-likelihood, with the estimated 95% confidence ellipse overlaid in black, centered at the maximum likelihood (marked with a 'x'), obtained with the programs that are in "maximum_likelihood_estimation.py".
