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


`>> Model not specified`
`>> USAGE: hrftn96 [-P] [-S] [-2] [-r] [-z] -RAYP p -ALP alpha -DT dt -NSAMP nsamp -M model`
`>> -P           (default true )    Incident P wave`
`>> -S           (default false)    Incident S wave`
`>> -RAYP p      (default 0.05 )    Ray parameter in sec/km`
`>> -DT dt       (default 1.0  )    Sample interval for synthetic`
`>> -NSAMP nsamp (default 512  )    Number samples for synthetic`
`>> -M   model   (default none )    Earth model name`
`>> -ALP alp     (default 1.0  )    Number samples for synthetic`
`>>      H(f) = exp( - (pi freq/alpha)**2) `
`>>      Filter corner ~ alpha/pi `
`>> -2           (default false)    Use 2x length internally`
`>> -r           (default false)    Output radial   time series`
`>> -z           (default false)    Output vertical time series`
`>>      -2  (default false) use double length FFT to`
`>>      avoid FFT wrap around in convolution `
`>> -D delay     (default 5 sec)    output delay sec before t=0`
`>> -?                   Display this usage message`
`>> -h                   Display this usage message`
`>>  SAC header values set by hrftn96`
`>>   B     :  delay`
`>>   USERO :  gwidth        KUSER0:  Rftn`
`>>   USER4 :  rayp (sec/km)`
`>>   USER5 :  fit in % (set at 100)`
`>>   KEVNM :  Rftn          KUSER1:  hrftn96`
`>> The program creates the file names hrftn96.sac`
`>> This is the receiver fucntion, Z or R trace according to the command line flag`



2. You can install all the required python packages for Jrfapp by running `pip install jrfapp`.
<div class="alert alert-block alert-info"> <b>Tip:</b> It is highly recommended to create a conda environment and install the package in this environment. </div>

## Examples:
I have included four tutorials on the GitHub page that explain the main usage of the package. 


