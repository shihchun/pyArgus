from pyargus.directionEstimation import *
import matplotlib.pyplot as plt
import numpy as np
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 14:50:23 2020

@author: geek
"""
d = 0.5 # Inter element spacing [lambda]
# M = 4  # number of antenna elements in the antenna system (ULA)
N = 2**12  # sample size used for the simulation          
# theta = 60 # incident angle of the test signal [deg]

# Number of antenna elements
M = 16

# Interelement spacing is half-wavelength
d= 0.5

# Incident angle of source 1
theta_1 =50

# Incident angle of source 2
theta_2 =80

# Incident angle of source 3
theta_3 = 110

# Generate ULA scanning vectors
array_alignment = np.arange(0, M, 1)* d
incident_angles= np.arange(0,181,1)
scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)

# Array response vectors of test source 1
a_1 = np.exp(np.arange(0,M,1)*1j*2*np.pi*d*np.cos(np.deg2rad(theta_1)))

# Array response vectors of test source 2
a_2 = np.exp(np.arange(0,M,1)*1j*2*np.pi*d*np.cos(np.deg2rad(theta_2)))

# Array response vectors of test source 2
a_3 = np.exp(np.arange(0,M,1)*1j*2*np.pi*d*np.cos(np.deg2rad(theta_3)))

# Generate multichannel test signal 
soi = np.random.normal(0,1,N)  # Signal of Interest
soi_matrix  = ( np.outer( soi, a_1) + np.outer( soi, a_2) + np.outer( soi, a_3)).T 

# Generate multichannel uncorrelated noise
noise = np.random.normal(0,np.sqrt(10**-1),(M,N))

# Create received signal array
rec_signal = soi_matrix + noise 

# Estimating the spatial correlation matrix
R = corr_matrix_estimate(rec_signal.T, imp="mem_eff")

# Caclulate the forward-backward spatial correlation matrix
R_fb = forward_backward_avg(R)

# Estimate DOA 
Bartlett = DOA_Bartlett(R_fb, scanning_vectors)
Capon = DOA_Capon(R_fb, scanning_vectors)
MEM = DOA_MEM(R_fb, scanning_vectors, column_select = 1)
LPM = DOA_LPM(R_fb, scanning_vectors, element_select = 1)
MUSIC = DOA_MUSIC(R_fb, scanning_vectors, signal_dimension = 3)

# Get matplotlib axes object
plt.figure()
axes = plt.axes()

# Plot results on the same fiugre
DOA_plot(Bartlett, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(Capon, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(MEM, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(LPM, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(MUSIC, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)

axes.legend(("Bartlett","Capon","MEM","LPM","MUSIC"))

# Mark nominal incident angles
axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_1)
axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_2)
axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_3)

# Sub-array size
P = 7

# Calculate the forward-backward spatially smotthed correlation matrix
R_ss = spatial_smoothing(rec_signal.T, P=P, direction="forward-backward")

# Regenerate the scanning vector for the sub-array
array_alignment = np.arange(0, P, 1)* d
incident_angles= np.arange(0,181,1)
scanning_vectors = gen_ula_scanning_vectors(array_alignment, incident_angles)

# Estimate DOA 
Bartlett = DOA_Bartlett(R_ss, scanning_vectors)
Capon = DOA_Capon(R_ss, scanning_vectors)
MEM = DOA_MEM(R_ss, scanning_vectors, column_select = 1)
LPM = DOA_LPM(R_ss, scanning_vectors, element_select = 1)
MUSIC = DOA_MUSIC(R_ss, scanning_vectors, signal_dimension = 3)

# Get matplotlib axes object
plt.figure()
axes = plt.axes()

# Plot results on the same fiugre
DOA_plot(Bartlett, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(Capon, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(MEM, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(LPM, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)
DOA_plot(MUSIC, incident_angles, log_scale_min = -50, axes=axes, alias_highlight=False)

axes.legend(("Bartlett","Capon","MEM","LPM","MUSIC"))

# Mark nominal incident angles
axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_1)
axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_2)
axes.axvline(linestyle = '--',linewidth = 2,color = 'black',x = theta_3)

plt.show()