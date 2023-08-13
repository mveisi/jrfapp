# Jrfapp Package
## Introduction:
### The Jrfapp package is a python package used for performing joint inversion of Receiver Function and apparent velocity data. It outputs an estimated shear velocity model. For more information, please refer to the "manuscript title".


## Installation:
To run this code, you will need the following software and tools:

- Computer Program in Seismology
- Python 3.8
- matplotlib
- numpy
- obspy
- rf
1. You can install the Computer Program in Seismology (CPS) from [here](https://www.eas.slu.edu/eqc/eqccps.html).
Please ensure that you have added the binary path of CPS to your .bashrc before proceeding to the next step. This package depends on the hrftn96, and you can verify the installation by running the hrftn96 command in the terminal. The correct output of hrftn96 should appear as follows:


`>> Model not specified`\n
`>> USAGE: hrftn96 [-P] [-S] [-2] [-r] [-z] -RAYP p -ALP alpha -DT dt -NSAMP nsamp -M model`
`....`



2. You can install all the required python packages for Jrfapp by running `pip install jrfapp`.
<div class="alert alert-block alert-info"> <b>Tip:</b> It is highly recommended to create a conda environment and install the package in this environment. </div>

## Examples:
I have included four tutorials on the GitHub page that explain the main usage of the package. 


