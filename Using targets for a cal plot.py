# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 13:56:24 2024

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

#al55co01_2023-07-05_13
df = pd.read_csv("20240117_LIF_processed_data_02.txt", header=0,usecols=["time_s", "on_cts_norm","off_cts_norm","cts_norm","off_lsr_pwr_V","on_lsr_pwr_V","off_ref_cts_norm","on_ref_cts_norm","task","cal_no_mfc","cell_flow","seed_laser_t"])
"""
fig, CO_timeseries = plt.subplots(1, 1)
CO_timeseries.plot(df['sens'], color= "blue")
plt.ylabel("Raw counts (Hz)", fontsize =20)
plt.xlabel("UTC", fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
"""
count_data = [np.mean(df["on_cts_norm"][81680:83817]),
              np.mean(df["on_cts_norm"][88840:91460]),
              np.mean(df["on_cts_norm"][79330:81620]),
                            np.mean(df["on_cts_norm"][91530:93688]),
                            np.mean(df["on_cts_norm"][76660:77760])]
print("This is calibration")
count_data_err = [scipy.stats.sem(df["on_cts_norm"][81680:83817]),
              scipy.stats.sem(df["on_cts_norm"][88840:91460]),
              scipy.stats.sem(df["on_cts_norm"][79330:81620]),
                            scipy.stats.sem(df["on_cts_norm"][91530:93688]),
                            scipy.stats.sem(df["on_cts_norm"][76660:77760])]
print(count_data_err, "This is the count data error")

x = np.array([1.96001568, 3.918495298, 5.875440658, 7.830853563, 9.784735812]) # cylinder concs
x_err = np.array([0.05, 0.05, 0.05, 0.05, 0.05]) # cylinder errors

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
print(error, "Error from the ODR")
print(out.beta[0], out.beta[1])#m=0, c=1

#Calculating the R^2 value
corr_matrix = np.corrcoef(x, y) 
corr = corr_matrix[0,1]
R_sq = corr**2
print(R_sq, "R squared value")

# Fit with polyfit
b, m = polyfit(x, y, 1)
#plt.plot(x, b + m * x, '-')

print("Here is where we calculate our targets")
#this is where we calc our targets!
target_data = [np.mean(df["on_cts_norm"][84000:85850]),np.mean(df["on_cts_norm"][86000:88800])]                                                               
print("These are the average counts for the targets measured", target_data)

#get the average/interpolated value for the cals
sens_av = m
sens_err_av=error[0]
zero_av=b
zero_err_av=error[1]
print("These are the concs of the targets (give them two one dp)")
#get the target conc, just need to specify which one is which
target_conc = (  ((target_data[0]-zero_av)/sens_av) ,  
               ((target_data[1]-zero_av)/sens_av) )
print(target_conc[0], "target one\n", 
      target_conc[1], "target two\n")
#this gives the % error on the cylinder for the conc
target_error=((sens_err_av/sens_av)+(zero_err_av/zero_av))
print(f'The error overall for this bracket is {target_error:.2f}' )
print("This is the offset for each of the target concs calculated above")
calc_ppb_offset=(target_conc[0]*target_error,  
                 target_conc[1]*target_error)
print(calc_ppb_offset[0], "Offset one\n",
      calc_ppb_offset[1], "Offset two\n")