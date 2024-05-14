# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:19:21 2024

@author: Eve
"""

#import modules in
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.odr import *
import random
from numpy.polynomial.polynomial import polyfit
import scipy.stats
import time
import matplotlib 

#Read in the data. Define the header as 0 as this is where the column heads are.
df = pd.read_csv("20240514_SO2_processed_data_1.txt", header=0, usecols = ["time_ms","Cts_diff","off_cts_norm","on_cts_norm","off_lsr_pwr_V","on_lsr_pwr_V"])
#check that the code above works
#df = df[(df[12] >= 1) & (df[12] <= 600)]
print(df['Cts_diff'])
#get the main overview plot for the data, to then get the subsections to use for the calcs

# this plots an overview graph, where you can attain the values for the count_data and count_data_err, from the observations noted during the experiment.
fig, CO_timeseries = plt.subplots(1, 1)
CO_timeseries.plot(df['Cts_diff'], color= "blue")
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.show()

# this plots the means of the low, medium and high data and plots them against the three cylinder concentrations
#simply change the values for each section that is being used.
count_data = [np.mean(df['Cts_diff'][:]),
              np.mean(df['Cts_diff'][:]),
              np.mean(df['Cts_diff'][:])]
print(count_data)  
count_data_err = [scipy.stats.sem(df['Cts_diff'][:]),
              scipy.stats.sem(df['Cts_diff'][:]),
              scipy.stats.sem(df['Cts_diff'][:])]
print(count_data_err)

#need to take the sqrt n and need the number of points 

#these are the three values and errors for the cylinders from the NOAA WMO CO_X2014A scale
x = np.array([]) # cylinder concs
x_err = np.array([]) # cylinder errors

#values obtained above
y = count_data
y_err = count_data_err

#Below is the xy-orthogonal weighting

# Define a function (linear) to fit the data with.
def f(B, x):
    return B[0]*x + B[1]

# Create a model for fitting.
linear_model = Model(f)

# Create a RealData object using our initiated data from above.
data = RealData(x, y, sx = x_err, sy = y_err)

# Set up ODR with the model and data.
odr = ODR(data, linear_model, beta0=[0., 1.])

# Run the regression.
out = odr.run()

# Use the in-built print method to give us results.
error = np.sqrt(np.diag(out.cov_beta))
print(error)
print(out.beta[0], out.beta[1])#m=0, c=1

#Calculating the R^2 value
corr_matrix = np.corrcoef(x, y) 
corr = corr_matrix[0,1]
R_sq = corr**2
print(R_sq)

#Adding labels
fig,weight = plt.subplots(1,1)
s = [20**2]
weight.scatter(x, y ,s=s, marker='+', color = "black")

# Fit with polyfit
b, m = polyfit(x, y, 1)
plt.plot(x, b + m * x, '-')

plt.xlabel("Cylinder concentration (ppt)",fontsize=20)
plt.ylabel("Counts (cps/mW)",fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
#This plots the m,c and R^2 values on the plot. sig figs can be changed by changing the number before "f"
plt.annotate("m = %.2f \u00B1 %.2f Hz/ppb \nc = %.0f \u00B1 %.0f Hz\nR2 = %.5f" % (out.beta[0],error[0],out.beta[1], error[1], R_sq), xycoords = "axes fraction", xy = (0.6, 0.1),fontsize=20)
plt.show()