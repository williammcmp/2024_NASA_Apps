#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday October 05 2024

Utilities for required across the repo

@author: william.mcm.p
"""

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


def get_window_df( data: pd.DataFrame, start_index: int, end_index: int = None, length: int = None) -> pd.DataFrame:
    """
    Returns a windowed portion of the DataFrame based on the provided index range.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        start_index (int): The starting index of the window.
        end_index (int, optional): The ending index of the window. Defaults to None.
        length (int, optional): The length of the window if end_index is not provided. Defaults to None.
    
    Returns:
        pd.DataFrame: The sliced window of the DataFrame.
    """
    
    # Allow for lengths to be defined
    if length is not None and end_index is None:
        end_index = start_index + length

    # Check the end index is after start index
    if end_index < start_index:
        print(f"Start index [{start_index}] must be before end index [{end_index}]")
        return data

    return data.iloc[start_index:end_index]
    