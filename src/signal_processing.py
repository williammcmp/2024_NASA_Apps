#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday October 05 2024

Signal processing and filtering utils

@author: william.mcm.p
"""
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt


def highpass_filter(
    data_series: pd.Series, cutoff: float, fs: float, order: int = 5
) -> pd.Series:
    """
    Apply a high-pass filter to the data.

    Args:
        data_series (pandas Series): The input signal data to be filtered.
        cutoff (float): The cutoff frequency of the filter in Hz.
        fs (float): The sampling rate of the data in Hz.
        order (int): The order of the filter (default is 5).

    Returns:
        array-like: The filtered signal.
    """
    # Copying the data ensures that the orignal memory is not not overwitten mid process
    data = data_series.copy()

    # This filter breaks when NaN values are present
    if data.isna().any():
        print("Warning: There are NaN values detected in the input data")

    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize the frequency

    # Design the butterworth filter
    b, a = butter(order, normal_cutoff, btype="high", analog=False)

    # Apply the filter
    filtered_data = filtfilt(b, a, data)

    # Some feedback warning about the high-pass filter
    if np.isnan(filtered_data).all():
        print(
            "*** Warning: high-pass filter has resulted in only NaN values generated. You may of parsed in a series with NaN values. ***"
        )

    return filtered_data

def lowpass_filter(
    data_series: pd.Series, cutoff: float, fs: float, order: int = 5
) -> pd.Series:
    """
    Apply a low-pass filter to the data.

    Args:
        data (array-like): The input signal data to be filtered.
        cutoff (float): The cutoff frequency of the filter in Hz.
        fs (float): The sampling rate of the data in Hz.
        order (int): The order of the filter (default is 5).

    Returns:
        array-like: The filtered signal.
    """
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize the cutoff frequency
    # Design the low-pass filter
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # Apply the filter
    filtered_data = filtfilt(b, a, data_series)
    return filtered_data