#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday October 05 2024

Loop through a series of seismic recordings and find the seismic events, then save them as into seperate DataFrames

@author: william.mcm.p
"""

import ray
import pandas as pd
from src.utilis import load_dataframe, save_dataframe, get_file_names, get_mseed_path, get_sample_rate, get_col_from_pattern
from src.signal_processing import find_seismic_event

# Initialize Ray
ray.init(num_cpus=12) # Chnage this number based on the number CPU cores you have avaliable on you system

loc_name = "S16_GradeB/"

# The path where all the raw_data files are located
signal_data_folder = 'E:/Users/William/Uni/Swinburne OneDrive/OneDrive - Swinburne University/2024 NASA Space Apps Challange/Data/mars/training/data/'

# Where the saved dataframes/stats will be exported to
seismic_event_output_folder = 'E:/Users/William/Uni/Swinburne OneDrive/OneDrive - Swinburne University/2024 NASA Space Apps Challange/Data/mars/diff_diff_model/training/'

# Where the plots will be saved to
plot_paths = 'E:/Users/William/Uni/Swinburne OneDrive/OneDrive - Swinburne University/2024 NASA Space Apps Challange/Analysis/plots/secsimic events/mars plots/training/'

# The reference data from the training data catalog
reference_df = load_dataframe("E:/Users/William/Uni/Swinburne OneDrive/OneDrive - Swinburne University/2024 NASA Space Apps Challange/Data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv")

# Get the list of names from the signal data folder
list_of_names = get_file_names(signal_data_folder, '.csv')


# Define the task to run in parallel
@ray.remote
def process_file(index, name):
    print(f"Computing file {index}/{len(list_of_names)}")

    ref_row = reference_df.iloc[index]  # the expected information from the cataloged data
    signal_data = load_dataframe(signal_data_folder + name)  # loading the signal data

    # # Accessing the recordings metadata ---> needed to set up for the rest of the project
    velocity_field = get_col_from_pattern(signal_data, 'vel')
    time_field = get_col_from_pattern(signal_data, 'rel')

    # Find the sample frequency from the minimized data file
    sample_rate = get_sample_rate(get_mseed_path(signal_data_folder + name))

    # Find the seismic event from the signal data
    seismic_event, seismic_event_start = find_seismic_event(
        signal_data, 
        velcoity_field=velocity_field,
        time_field=time_field,
        sampleFreq=sample_rate, 
        showPlot=True, 
        plotPath=plot_paths + name + '.png', 
        # expected_start=ref_row[get_col_from_pattern(reference_df, 'rel')]
    )

    # Save the signal data
    # save_dataframe(seismic_event, seismic_event_output_folder + name)

    # Return the stats for this file
    return {
        'filename': name,
        'Expected start (s)': ref_row[get_col_from_pattern(reference_df, 'rel')],
        'Calculated start (s)': seismic_event_start,
        'Orignal recording length (s)': signal_data[time_field].max(),
        'Cut recording length (s)': seismic_event[time_field].max() - seismic_event[time_field].min(),
        'Cropped percent (%)': 100 * (seismic_event[time_field].max() - seismic_event[time_field].min()) / signal_data[time_field].max(),
        'Percent start difference to orignal length (%)':  100* (seismic_event_start - ref_row[get_col_from_pattern(reference_df, 'rel')]) / signal_data[time_field].max(),
        'Percent start difference to cropped length (%)':  100* (seismic_event_start - ref_row[get_col_from_pattern(reference_df, 'rel')]) / (seismic_event[time_field].max() - seismic_event[time_field].min())
    }

# Launch parallel tasks and get their results
futures = [process_file.remote(index, name) for index, name in enumerate(list_of_names)]
results = ray.get(futures)


# Shutdown Ray after the tasks are complete
ray.shutdown()

stats_df = pd.DataFrame.from_dict(results)

save_dataframe(stats_df, seismic_event_output_folder + '_model stats.csv')
