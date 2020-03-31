# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 18:44:14 2020

@author: Jordi Bolibar

Climate projections analysis with the ADAMONT (Verfaillie et al., 2019) 
and SAFRAN (Durand et al., 2009) datasets. 

"""

## Dependencies: ##
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import dask.array as da
import os

######   FILE PATHS    #######

workspace = str(Path(os.getcwd())) + '\\'
path_climate_members = workspace + 'climate_members\\'

path_safran_forcings = 'C:\\Jordi\\PhD\\Data\\SAFRAN-Nivo-2017\\'
path_adamont_forcings = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\treated\\'
path_massif_IDs = 'C:\\Jordi\\PhD\\Data\\ADAMONT\\' 

SAFRAN_year_raw_filepaths = np.asarray(os.listdir(path_safran_forcings))
ADAMONT_proj_filepaths = np.asarray(os.listdir(path_adamont_forcings))

SAFRAN_year_filepaths = np.array([])
for safran_year in SAFRAN_year_raw_filepaths:
#    if((float(safran_year) < 18) or (float(safran_year) > 66)):
    if((float(safran_year) < 18)):
        SAFRAN_year_filepaths = np.append(SAFRAN_year_filepaths, path_safran_forcings + safran_year + '\\FORCING.nc') 

# Open the index for SAFRAN massif IDs
massif_IDs = np.genfromtxt(path_massif_IDs + 'Massifs_SAFRAN_Alpes.csv', delimiter=';', skip_header=1, dtype=str) 

#####   FLAGS    ########
#n_massif = 3 # Mont-Blanc
n_massif = 8 # Belledonne
massif = massif_IDs[np.where(massif_IDs[:,1] == str(n_massif)), 2][0][0]
altitude = 2100 #m
aspect = -1

#######################    FUNCTIONS    ##########################################################

def find_nearest_altitude(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def get_SAFRAN_idx(massif, altitude, aspect, climate_data):
    
#    print("\nSAFRAN: ")
    massif_idx = np.where(massif == climate_data['massif_number'][0,:])[0]
    aspect_idx = np.where(aspect == climate_data['aspect'][0,:])[0]
    massif_aspect_idx = np.array(list(set(massif_idx).intersection(aspect_idx)))
    
    alt_idx = find_nearest_altitude(climate_data['ZS'][0,:][massif_aspect_idx], altitude).data
    final_idx = massif_aspect_idx[alt_idx]
    
#    print("Massif: " + str(climate_data['massif_number'][0,:][final_idx].data))
#    print("Aspect: " + str(climate_data['aspect'][0,:][final_idx].data))
#    print("Altitude: " + str(climate_data['ZS'][0,:][final_idx].data))
    
    return final_idx

def get_ADAMONT_idx(massif, altitude, climate_data):
    
#    print("\nADAMONT: ")
    massif_idx = np.where(massif == climate_data['MASSIF_NUMBER'])[0]
    alt_idx = find_nearest_altitude(climate_data['ZS'][massif_idx], altitude).data
    final_idx = massif_idx[alt_idx]
    
#    print("Massif: " + str(climate_data['MASSIF_NUMBER'][final_idx].data))
#    print("Altitude: " + str(climate_data['ZS'][final_idx].data))
    
    return final_idx

##################################################################################################


###############################################################################
###                           MAIN                                          ###
###############################################################################

# Open all SAFRAN netCDF files in a single time series
safran_climate = xr.open_mfdataset(SAFRAN_year_filepaths, concat_dim="time", combine='by_coords', parallel=True)

# We get the SAFRAN idx for the selected topographical position
safran_idx = get_SAFRAN_idx(n_massif, altitude, aspect, safran_climate)

# Monthly 
safran_tmean_mon = safran_climate['Tair'][:,safran_idx].resample(time="1MS").mean() -273.15
safran_snow_mon = safran_climate['Snowf'][:,safran_idx].resample(time="1MS").sum()*3600

# Annual 
safran_tmean_a = safran_climate['Tair'][:,safran_idx].resample(time="1AS").mean() -273.15
safran_snow_a = safran_climate['Snowf'][:,safran_idx].resample(time="1AS").sum()*3600

# Iterate ADAMONT projection members
for i in range(0, ADAMONT_proj_filepaths.size, 2):
    
    member_name = str(ADAMONT_proj_filepaths[i])[8:-37]
    print("\nProcessing " + str(member_name))
    
    # Locate and determine the paths of the current members
    current_ADAMONT_model_daymean = path_adamont_forcings + str(ADAMONT_proj_filepaths[i])
    current_ADAMONT_model_daysum = path_adamont_forcings + str(ADAMONT_proj_filepaths[i+1])
    
    # We open the files with xarray
    climate_daymean = xr.open_dataset(current_ADAMONT_model_daymean)
    climate_daysum = xr.open_dataset(current_ADAMONT_model_daysum)
    
    climate_daymean = climate_daymean.rename({"TIME": "time"})
    climate_daysum = climate_daysum.rename({"TIME": "time"})
    adamont_idx = get_ADAMONT_idx(n_massif, altitude, climate_daymean)
    
    #### Retrieve temperature data
    # Monthly
    adamont_tmean_mon = climate_daymean['Tair'][:,adamont_idx].resample(time="1MS").mean() 
    adamont_tmean_mon = adamont_tmean_mon.sel(time = slice('2017-01-01', '2100-12-31')) -273.15
    
    # Annual
    adamont_tmean_a = climate_daymean['Tair'][:,adamont_idx].resample(time="1AS").mean() 
    adamont_tmean_a = adamont_tmean_a.sel(time = slice('2017-01-01', '2100-12-31')) -273.15
    
    #### Retrieve snow data
    # Monthly
    adamont_snow_mon = climate_daysum['SNOW'][:,adamont_idx].resample(time="1MS").sum()
    adamont_snow_mon = adamont_snow_mon.sel(time = slice('2017-01-01', '2100-12-31'))
    
    # Annual
    adamont_snow_a = climate_daysum['SNOW'][:,adamont_idx].resample(time="1AS").sum()
    adamont_snow_a = adamont_snow_a.sel(time = slice('2017-01-01', '2100-12-31'))
    
    ### Combine SAFRAN and ADAMONT data to create a 2000-2100 time series
    # Monthly
    common_tmean_months = np.concatenate((safran_tmean_mon.time, adamont_tmean_mon.time))
    common_tmean_mon = da.concatenate((safran_tmean_mon.data, adamont_tmean_mon.data))
    common_snow_months = np.concatenate((safran_snow_mon.time, adamont_snow_mon.time))
    common_snow_mon = da.concatenate((safran_snow_mon.data, adamont_snow_mon.data))
    
    # Annual
    common_tmean_years = np.concatenate((safran_tmean_a.time, adamont_tmean_a.time))
    common_tmean_a = da.concatenate((safran_tmean_a.data, adamont_tmean_a.data))
    common_snow_years = np.concatenate((safran_snow_a.time, adamont_snow_a.time))
    common_snow_a = da.concatenate((safran_snow_a.data, adamont_snow_a.data))
    
    # Create folders if needed
    current_massif_path_mon = path_climate_members + 'monthly\\' + str(massif) + '\\'
    current_massif_path_a = path_climate_members + 'annual\\' + str(massif) + '\\'
    if(not os.path.exists(current_massif_path_mon)):
        os.makedirs(current_massif_path_mon)
    if(not os.path.exists(current_massif_path_a)):
        os.makedirs(current_massif_path_a)
        
    #######   Plot data  ###############
    # Monthly data
    fig1, (ax11, ax12) = plt.subplots(2,1, figsize=(14, 8))
    fig1.suptitle(member_name + " - " + massif + " - " + str(altitude) + " m")
    
    ax11.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax11.plot(common_tmean_months, common_tmean_mon, linewidth=1, label='Mean monthly temperature', c='darkred')
    ax11.axvline(x=safran_tmean_mon.time.data[-1], color='grey', linewidth=2.0, linestyle='--')
    ax11.set_ylabel('Temperature (°C)')
    ax11.set_xlabel('Year')
    ax11.legend()
    
    ax12.plot(common_snow_months, common_snow_mon, linewidth=1, label='Mean monthly snowfall', c='steelblue')
    ax12.axvline(x=safran_snow_mon.time.data[-1], color='grey', linewidth=2.0, linestyle='--')
    ax12.set_ylabel('Precipitation (mm)')
    ax12.set_xlabel('Year')
    ax12.legend()
    
    # Save the current plot
    fig1.savefig(current_massif_path_mon + member_name + '_climate_' + str(massif) + '_' + str(altitude) + '.png')   
    plt.close()
    
    # Annual data
    fig2, (ax21, ax22) = plt.subplots(2,1, figsize=(14, 8))
    fig2.suptitle(member_name + " - " + massif + " - " + str(altitude) + " m")
    
    ax21.axhline(y=0, color='black', linewidth=0.7, linestyle='-')
    ax21.plot(common_tmean_years, common_tmean_a, linewidth=1, label='Mean annual temperature', c='darkred')
    ax21.axvline(x=safran_tmean_a.time.data[-1], color='grey', linewidth=2.0, linestyle='--')
    ax21.set_ylabel('Temperature (°C)')
    ax21.set_xlabel('Year')
    ax21.legend()
    
    ax22.plot(common_snow_years, common_snow_a, linewidth=1, label='Mean annual snowfall', c='steelblue')
    ax22.axvline(x=safran_snow_a.time.data[-1], color='grey', linewidth=2.0, linestyle='--')
    ax22.set_ylabel('Precipitation (mm)')
    ax22.set_xlabel('Year')
    ax22.legend()
    
    # Save the current plot
    fig2.savefig(current_massif_path_a + member_name + '_climate_' + str(massif) + '_' + str(altitude) + '.png')   
    plt.close()
    
#    import pdb; pdb.set_trace()

print("\nAll climate members processed")



    