#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday October 05 2024

Loading and saving data functions

@author: william.mcm.p
"""

import pandas as pd

def load_data_from_csv(file_path : str):
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame or None: A pandas DataFrame containing the data from the CSV file.
            Returns None if the file is not found or an error occurs during loading.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from CSV file: {str(e)}")
        return None
    

def load_data_from_excel(file_path, sheet_name=0, load_all_sheets=False):
    """
    Load data from an Excel file.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str or int, optional): The sheet name or index to load.
            Defaults to 0 (the first sheet).
        load_all_sheets (bool, optional): If True, loads all sheets and returns
            a dictionary of DataFrames. Defaults to False.

    Returns:
        pd.DataFrame or dict or None: A pandas DataFrame containing the data from the Excel file
            if load_all_sheets is False, or a dictionary of DataFrames if load_all_sheets is True.
            Returns None if the file is not found or an error occurs during loading.
    """
    try:
        if load_all_sheets:
            data = pd.read_excel(file_path, sheet_name=None)
        else:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
        return data
    except FileNotFoundError:
        print(f"File not found at: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data from Excel file: {str(e)}")
        return None