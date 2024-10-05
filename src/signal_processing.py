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

from src.utilis import get_timed_window

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

def find_seismic_event(signal_df : pd.DataFrame, target_field: str = 'velocity(m/s)', sampleFreq : float = 6.625, time_field : str = 'time_rel(sec)', showPlot: bool = False, plotPath: str = None, expected_start : float = None) -> pd.DataFrame:

    # Validate the the target field in present in the dataFrame
    if target_field in signal_df.columns:
        # Get the signal data series from the dataFrame
        signal = signal_df[target_field]
    else:
        print(f"Warning: {target_field} is NOT a valid filed in {signal_df.columns}")

    # run the band pass filter on the signal
    high_passed_signal = highpass_filter(signal, 0.8, sampleFreq)
    filtered_signal = lowpass_filter(high_passed_signal, 0.6, sampleFreq)

    # Compute the double difference to find the maxiumin rate of change of the seismic event --> assuming this to be the peak of the seismic wave
    filtered_signal_diff = np.diff(np.diff(filtered_signal))

    # Get index of the max signal rate of change
    peak_index = np.argmax(filtered_signal_diff)

    # Get start/end  time of window
    start_time = signal_df[time_field].iloc[peak_index] -500 # offset to capture the front of the wave
    end_time = signal_df[time_field].iloc[peak_index] + 5000 # offset of caputer end of the wave


    seismic_event = get_timed_window(signal_df, start_time, end_time)

    # Plotting the the outcome of the Seismic wave analysis
    if showPlot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, figsize=(8, 6), sharex=False, sharey=False)

        ax[0].plot(signal_df[time_field], signal_df[target_field], label="Raw - Signal")
        ax[0].set_ylabel(target_field)
        ax[0].set_xlabel('Time (s)')
        ax[0].axvline(x=start_time, c='red', label='Guess of start Time')
        ax[0].set_title('Raw Signal')
        ax[0].legend()

        ax[1].plot(signal_df[time_field], filtered_signal, label="Bandpass filtered Signal")
        ax[1].set_ylabel(target_field)
        ax[1].set_xlabel('Time (s)')
        ax[1].axvline(x=start_time, c='red', label='Guess of start Time')
        ax[1].set_title('Filtered Signal')
        ax[1].legend()

        ax[2].plot(seismic_event[time_field], seismic_event[target_field], label="Raw - Seismic Event")
        ax[2].set_ylabel(target_field)
        ax[2].set_xlabel('Time (s)')
        ax[2].axvline(x=start_time, c='red', label='Guess of start Time')
        ax[2].set_title('Seismic Event')
        ax[2].legend()

        if expected_start is not None:
            ax[0].axvline(x=expected_start, c='green', label='Expected Start - Traning Data')
            ax[0].legend()
            ax[1].axvline(x=expected_start, c='green')
            ax[2].axvline(x=expected_start, c='green')

        fig.tight_layout()
        # fig.show()

        # Option to save the plot to an output folder
        if plotPath is not None:
            fig.savefig(plotPath, dpi=300)

    return seismic_event, start_time