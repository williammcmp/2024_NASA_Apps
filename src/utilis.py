#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday October 05 2024

Utilities for required across the repo

@author: william.mcm.p
"""

from obspy import read
import pandas as pd
import os


from src.loading_and_saving_data import load_data_from_csv, load_data_from_excel

def save_dataframe(df: pd.DataFrame, file_path: str, force_format: str = None, headers=True, index = False, silent_mode = False):
    """
    Save a pandas DataFrame to a specified file format.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - file_path (str): The path where the DataFrame should be saved.
    - force_format (str, optional): Desired file format to force. Supported formats: 'parquet' (recommended), 'csv', 'excel', 'txt'.

    Returns:
    - bool: True if saving was successful, False otherwise.
    """
    try:
        # Determine the file format to use
        file_format = force_format or file_path.split('.')[-1].lower()
        
        if file_format == 'parquet': # the binary type --> fastest
            df.to_parquet(file_path)
        elif file_format == 'csv':
            df.to_csv(file_path, index=index)
        elif file_format == 'excel' or file_format == 'xlsx':
            df.to_excel(file_path, index=index)

        else:
            # Incase someone wants to save to an unsupported formate
            raise ValueError(f"Unsupported file format: {file_format}")

        if not silent_mode:
            print(f"DataFrame successfully saved to {file_path}")

        return True

    except Exception as e:
        print(f"Error saving DataFrame: {e}")
        return False
    
def load_dataframe(file_path: str, enable_pyarrow = False):
    """
    Load a pandas DataFrame from a specified file format.
    
    Parameters:
    - file_path (str): The path to the file to load.
    - enable_pyarrow (bool): Eanbled an optimised method for loading parquet files

    Returns:
    - pd.DataFrame: The loaded DataFrame if successful, None otherwise.
    """
    try:
        # gets the file extension
        file_format = file_path.split('.')[-1].lower()

        
        if file_format == 'parquet': # the binary type --> fastest
            if enable_pyarrow:
                # This may cause loading problems for strange DataFrames
                df = pd.read_parquet(file_path, engine='pyarrow') # Use the C++ optimised engine for loading 
            else:
                df = pd.read_parquet(file_path) # let pandas decide what engine to use
        elif file_format == 'csv':
            df = load_data_from_csv(file_path)
        elif file_format in ['xls', 'xlsx']: # For the various types of Excel file types
            df = load_data_from_excel(file_path)
            # For the losers who want to load un-listed formats
            raise ValueError(f"Unsupported file format: {file_format}")

        # Check if the DataFrame is empty
        if df.empty:
            print(f"Warning: The DataFrame loaded from {file_path} is empty.")
            return None

        print(f"DataFrame successfully loaded from {file_path}")
        return df

    except Exception as e:
        print(f"Error loading DataFrame: {e}")
        return None
    
def get_file_names(folder_path : str, filter_pattern : str = None):
    """
    Get a list of filenames in the specified folder path.

    Args:
        folder_path (str): The path to the folder from which to retrieve filenames.
        filter_pattern (str): A pattern to filter the list of filenames by --> recommended to use file types

    Returns:
        list of str: A list of filenames in the specified folder.
    """

    file_names = os.listdir(folder_path)

    # filter out the list of filenames based on a pattern
    if filter_pattern is not None:
        file_names = [col for col in file_names if filter_pattern in col.lower()]

    return file_names


def get_timed_window( data: pd.DataFrame, start_time: int, end_time: int, field:str = 'time_rel(sec)') -> pd.DataFrame:
    """
    Returns a windowed portion of the DataFrame based on the provided time range.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        start_time (int): The starting second of the window.
        end_time (int): The ending second of the window.
        field (str, optional): The column name representing time. Defaults to 'time_rel(sec)'.
    
    Returns:
        pd.DataFrame: The sliced window of the DataFrame.
    """

    # Check if the start time is before the end time
    if start_time > end_time:
        print(f"Start time [{start_time}] must be before end time [{end_time}]")
        return data

    # Create a mask for the desired time window
    mask = (data[field] >= start_time) & (data[field] <= end_time)
    return data[mask]

def get_mseed_path(filename: str) -> str:
    """
    Converts a filename with a '.csv' extension to a '.mseed' extension.
    
    Args:
        filename (str): The original filename with '.csv' extension.
    
    Returns:
        str: The filename with the '.mseed' extension.
    """
    name, ext = os.path.splitext(filename)
    mseed_filename = name + '.mseed'
    return mseed_filename

def get_sample_rate(mseed_file: str) -> float:
    """
    Retrieves the sampling rate from a '.mseed' file.
    
    Args:
        mseed_file (str): The path to the '.mseed' file.
    
    Returns:
        float: The sampling rate of the data in the file.
    """
    st = read(mseed_file)
    return st[0].stats.sampling_rate


def get_col_from_pattern(data : pd.DataFrame, pattern:  str) -> str:
    """

    Gets the specific colums from the data frame that matches the column pattern
    - Time --> use 'rel' to target 'rel_time*' or 'time_rel*
    - Velocity --> use 'vel) to target 'Velocity*'

    Args:
        data (pd.DataFrame): the loaded dataFrame
        patternstr (str): the patten to match the column names against

    Returns:
        str: The targeted colunm name that matches the pattern
    """
    return [col for col in data.columns if pattern in col.lower()][0]

