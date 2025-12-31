# Data

This folder contains:

- `M10_Rab11mCherry_09_GaussianWeights.mp4` and `M10_LangerinYFp_09_GaussianWeights.mp4` are the raw sequences for each type of proteins.

- `Utrack_Langerin_GaussianWeights.mp4` and `Utrack_Rab11_GaussianWeights.mp4` are the results of proteins tracking (by the U-track algorithm) for the previous sequences.

- `real-data-video.mp4` gathers the two previous sequences by representing  Langerin vesicles as circles and Rab11 vesicles as triangles. Moreover, each particle is coloured with respect to its estimated motion regime: Brownian in blue, superdiffusive in red and subdiffusive in green. 


- The two images `trajectories_Langerin.png` and `trajectories_Rab11.png` depict all trajectories of Langerin and  Rab11 proteins, tracked over the image sequences, with the same color label as above (blue: Brownian, red: superdiffusive, green: subdiffusive).  

- The two .csv files contain the results of the tracking procedure for each protein type. Each file includes the spatial coordinates of the particles at each frame, the index of the trajectory to which they belong, and their estimated motion regime (1: Brownian, 2: superdiffusive, 3: subdiffusive). These are the files needed for the analysis of real data, as done in the main script `Data_analysis.py`.

- `maskc.pickle` encodes the region of interest, that is the cell containing the proteins. It is needed in the main script `Data_analysis.py`.

