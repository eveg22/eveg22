# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:11:36 2024

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
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
#df = pd.read_csv("LIFHK_20240111 8.txt", header=0,usecols=["EOM_Output_PD"])


#df = pd.read_csv("LIFHK_20240111 9.txt", header=0, delim_whitespace=True, names="Time_s	SO2_ppt	Signal_Dk_Counts	Ref_Dk_Counts	Signal_LIF_Counts	Ref_LIF_Counts	Cal_SB_MFC_Read_NA	Preamp_LD1_PD_Read	Preamp_LD1_T_Read	Preamp_LD2_PD_Read	Preamp_LD2_T_Read	Seed_Laser_PD	DC24V_mon	Main_Pump_1_PD	Main_Pump_2_PD	SO2_Lo_P	SO2_Hi_P	Cell_Flow	Cell_Pressure	EOM_Output_PD	Laser_Power_PT_0	Laser_Power_PT_1	Rep_Rate_Hz	seed_delay_ticks	regen_gate_delay_ticks	regen_gate_length_ticks	counting_gate_delay_ticks	counting_gate_length_ticks	Mod_DC_Fine	Mod_DC_Coarse	seed_laser_T_set	main_LD_current_set	Binary_Out	Binary_In	Binary_Settings	BV_MF_Set_cRIO	BV_MF_Set_Arduino	BV_MF_Read	BV_MF_Position	BV_P_Set_cRIO	BV_P_Set_Arduino	BV_P_Read	BV_P_Position	Task	MF_Gain	P_Gain	Seed_PD	Inlet_ZA_MFC_Read	Cal_ZA_MFC_Read	Ref_ZA_MFC_Read	Cal_SO2_MFC_Read	ZA_SB_MFC_Read	Inlet_ZA_MFC_P	Cal_ZA_MFC_P	Ref_ZA_MFC_P	Cal_SO2_MFC_P	ZA_SB_MFC_P	BBO_Position	Ref_Signal	Ref_Online	Ref_Offline	Seed_Laser_T_Read	SOA_T_Read	PC_Pressure	T_board_main_LD1	T_board_main_LD2	T_board_LIF_cell	T_board_REF_cell	T_board_LSRBOX_air	T_board_NLO	Thermistor_7	Thermistor_8 	Sig_Online	Sig_Offline	Seed_Online_V	Seed_Offline_V	Ref_Solenoid	Cal_SB_Vl	Cal_SO2_MFC_set	Inlet_ZA_MFC_set	PPLN_T	Free_VirtMem	Free_PhysMem	CPU_load_1	CPU_load_2	Dither_Remain_Min".split(' '))

#THIS IS TO GET THE CSV FILE FOR THE DATA!

#df = pd.read_csv('LIFHK_20240116 9.txt', sep='\s+', header=None)
#df.to_csv('LIFHK_20240116 9.csv', header=None)

df = pd.read_csv('Overnight EOM check.csv', header=0, usecols=["Time_s","EOM_Output_PD", "Main_Pump_1_PD",	"Main_Pump_2_PD", "Cell_Flow",	"Cell_Pressure" , "seed_laser_T_set",	"main_LD_current_set", "Seed_Laser_T_Read", "SOA_T_Read", "Sig_Online", "Sig_Offline" , "PPLN_T", "Laser_Power_PT_0", 	"Laser_Power_PT_1", "EOM_set_param"])

#print(df["EOM_Output_PD"])
#fig, laser = plt.subplots(1,1)
#laser.plot(df["Time_s"], df["EOM_Output_PD"])
#text = laser.twinx()
#lns2 = text.plot(df["Time_s"],df["Main_Pump_1_PD"])
#lns3 = text.plot(df["Time_s"],df["Main_Pump_2_PD"])
#laser.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
##plt.show()


"""
df = pd.read_csv('LIFHK_20240111 9.csv', header=0, usecols=["Time_s","EOM_Output_PD"])

#print(df["EOM_Output_PD"])
fig, laser = plt.subplots(1,1)
laser.plot(df["Time_s"], df["EOM_Output_PD"])
laser.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
"""
"""
Main_Pump_1_PD	Main_Pump_2_PD
Cell_Flow	Cell_Pressure
seed_laser_T_set	main_LD_current_set
Seed_Laser_T_Read	SOA_T_Read
Sig_Online	Sig_Offline
PPLN_T
Laser_Power_PT_0	Laser_Power_PT_1
"""

fig, ax = plt.subplots()
#fig.subplots_adjust(right=0.75)

twin1 = ax.twinx()
#twin2 = ax.twinx()

# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
#twin2.spines.right.set_position(("axes", 1.2))

p1, = ax.plot(df["Time_s"],df["EOM_Output_PD"], "b-", label="EOM signal")
p2, = twin1.plot(df["Time_s"],df["EOM_set_param"], "r-", label="EOM_set_param")
#p3, = twin2.plot(df["Time_s"], df["Cell_Pressure"], "g-", label="Cell_Pressure")
formatter = matplotlib.ticker.FuncFormatter(lambda s, x: time.strftime('%H:%M:%S', time.gmtime(s)))
ax.xaxis.set_major_formatter(formatter)
twin1.xaxis.set_major_formatter(formatter)
#twin2.xaxis.set_major_formatter(formatter)
ax.locator_params(axis='x', nbins=6)
ax.set_xlim()
ax.set_ylim(ymin = 0.005, ymax = 0.040)
twin1.set_ylim(ymin = 0.005, ymax = 0.040)
#twin2.set_ylim()

ax.set_xlabel("Time (UTC)",fontsize=20)
ax.set_ylabel("EOM",fontsize=20)
twin1.set_ylabel("EOM_set_param",fontsize=20)
#twin2.set_ylabel("Cell_Pressure",fontsize=20)

ax.yaxis.label.set_color(p1.get_color())
twin1.yaxis.label.set_color(p2.get_color())
#twin2.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
ax.tick_params(axis='y',labelsize=20)#, colors=p1.get_color(), **tkw)
twin1.tick_params(axis='y',labelsize=20)#, colors=p2.get_color(), **tkw)
#twin2.tick_params(axis='y', labelsize=20)#colors=p3.get_color(), **tkw,labelsize=20)
ax.tick_params(axis='x', **tkw, labelsize=20)

ax.legend(handles=[p1,p2], fontsize =20)

plt.show()
