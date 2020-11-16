
# Import pyArgus source files
from pyargus.antennaArrayPattern import array_rad_pattern_plot
import pyargus.beamform as bf
import pyargus.tests.beamformTest as bft
from pyargus.tests.patternPlotTest import demo_ULA_plot, demo_UCA_plot

import matplotlib.pyplot as plt
import numpy as np

#################
#################
theta_soi = 60  # Incident angle of the signal of interest
theta_interf = np.array([30, 130])  # Incident angles of the interferences

Pnoise   = 0.01              # noise variance  
d        = 0.5                # distance between antenna elements [lambda]
N        = 8                  # number of antenna elements

i = np.arange(N)

# Create array response vector for SOI
aS = np.exp(i*1j*2*np.pi*d*np.cos(np.deg2rad(theta_soi))) 
aS = np.matrix(aS).reshape(N,1)
# Create SOI autocorrelation matrix
Rss = aS * aS.getH() 

# Create interference autocorrelation matrix
Rnunu = np.matrix(np.zeros((N,N)))

for k in np.arange(np.size(theta_interf)):    
    aI = np.exp(i*1j*2*np.pi*d*np.cos(np.deg2rad(theta_interf[k])))  
    aI = np.matrix(aI).reshape(N,1)

    # Create interference autocorrelation matrix ( interferece signals are not correlated ) 
    Rnunu = Rnunu + aI * aI.getH()

# Create noise autocorrelation matrix
Rnn = np.matrix(np.eye(N)) * Pnoise

# Create noise + interferences autocorr matrix (interferences and thermal noise are not correlated)    
Rnunu = Rnunu + Rnn

# a figure instance to plot on
figure = plt.figure()

# create an axis
ax = figure.add_subplot(111)

# mark incident angles on the figure
ax.axvline(linestyle = '--',linewidth = 2,color = 'r',x = theta_soi)

for k in np.arange(np.size(theta_interf)):
    ax.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_interf[k])


#Calculate MSINR solution
SINR,w_msinr = bf.MSINR_beamform(Rss,Rnunu)# Rss replace 0
w_msinr /= np.sqrt(np.dot(w_msinr,w_msinr.conj()))

p_array_alignment = np.array(([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4],[0, 0, 0, 0, 0, 0, 0, 0]))
pattern = array_rad_pattern_plot(w = w_msinr,axes = ax, array_alignment = p_array_alignment) 
print('Signal to interference and noise ratio :',np.abs(SINR))


plt.show()