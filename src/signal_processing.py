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
    start_time = signal_df[time_field].iloc[peak_index] - 500 # offset to capture the front of the wave
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

def find_dominant_frequencies(input_signal, fs, num_frequencies=3, freq_min=None, freq_max=None, yscale_type='symlog', show_plot=False, method='fft'):
    input_signal_no_nan = input_signal[~np.isnan(input_signal)]
    N = len(input_signal_no_nan)

    if method == 'fft':
        # windowing - hanning
        window = np.hanning(N)
        signal_windowed = input_signal_no_nan * window

        # fft
        fft_result = np.fft.fft(signal_windowed)
        frequencies = np.fft.fftfreq(N, 1/fs)

        # calculate magnitude
        magnitude = np.abs(fft_result)
        
    elif method == 'welch':
        frequencies, magnitude = welch(input_signal_no_nan, fs, nperseg=N)
    else:
        raise ValueError("Method must be 'fft' or 'welch'")

    # limit search range
    if freq_min is not None:
        freq_min_idx = np.searchsorted(frequencies[:N // 2], freq_min)
    else:
        freq_min_idx = 1  # Skip DC component by default

    if freq_max is not None:
        freq_max_idx = np.searchsorted(frequencies[:N // 2], freq_max, side='right')
    else:
        freq_max_idx = N // 2

    # find dominant frequencies within the specified range
    dominant_frequencies = []
    for _ in range(num_frequencies):
        dominant_frequency_index = np.argmax(magnitude[freq_min_idx:freq_max_idx]) + freq_min_idx
        dominant_frequency = frequencies[dominant_frequency_index]
        dominant_frequencies.append((dominant_frequency, magnitude[dominant_frequency_index]))
        magnitude[dominant_frequency_index] = 0  # zero out magnitude of found dominant frequency

    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        # Plot only within the search range or the full range if not specified
        if freq_min is None:
            freq_min = frequencies[1]  # Skip DC component for plot start
        if freq_max is None:
            freq_max = frequencies[N // 2 - 1]
        plot_mask = (frequencies >= freq_min) & (frequencies <= freq_max)
        plt.plot(frequencies[plot_mask], 2.0/N * np.abs(magnitude[plot_mask]), label='Magnitude Spectrum')

        for i, (freq, mag) in enumerate(dominant_frequencies, start=1):
            if freq_min <= freq <= freq_max:
                plt.scatter(freq, 2.0/N * mag, label=f'Dominant Frequency {i}: {freq:.2f} Hz')

        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.yscale(yscale_type)
        plt.title(f'{method.upper()} of Signal')
        plt.xlim([freq_min, freq_max])
        plt.legend()
        plt.grid()
        plt.show()

    return dominant_frequencies

def inter_extra_polate_points(signal, known_indices, known_values, show_plot=False): 
    from scipy.interpolate import PchipInterpolator
    # Cubic Spline Interpolation
    # cs = CubicSpline(known_indices, known_values)
    cs = PchipInterpolator(known_indices, known_values)
    
    # interpolating the known range
    interpolated_indices = np.arange(known_indices.min(), known_indices.max() + 1)
    interpolated_values = cs(interpolated_indices)
    
    # extrapolation
    extrapolate_left = np.interp(np.arange(0, known_indices.min()), 
                                 known_indices[:2], known_values[:2])
    extrapolate_right = np.interp(np.arange(known_indices.max() + 1, len(signal)), 
                                  known_indices[-2:], known_values[-2:])
    
    # combine interpolated and extrapolated values
    full_range = np.arange(0, len(signal))
    full_values = np.concatenate([extrapolate_left, interpolated_values, extrapolate_right])
    
    if show_plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(full_range, signal, label='input signal')
        plt.plot(known_indices, known_values, 'o', label='data points')
        plt.plot(full_range, full_values, '-', label='Interpolated/Extrapolated Signal')
        plt.legend()
        plt.show()
    
    return full_values

def find_min_max_in_non_overlapping_window_with_threshold( 
                                            input_signal,
                                            max_zscore_threshold=3,
                                            min_zscore_threshold=-3,
                                            window_size=500, 
                                            fs=500,
                                            z_score_multiplier=1.2,
                                            show_plot=True,
                                            ):

    """
    Find the minimum and maximum values in a time series, then filter them 
    if two adjacent points fall within the window_size//2.
    
    Args:
        input_signal (np.ndarray): Input signal.
        window_size (int): The size of non-overlapping windows.
        threshold_type (str): Type of thresholding to use ('z_score', 'dynamic_z_score', 'absolute', 'dynamic_absolute').
        dynamic_window_percentage (float): Percentage of samples around the point for dynamic Z-score calculation.
        dynamic_window_length (int): Fixed length of samples around the point for dynamic Z-score calculation.
        show_plot (bool, optional): Whether to show the dot plot (default is True).
    
    Returns:
        tuple: A tuple containing:
            - min_indices (list): The indices of the minimum values in each non-overlapping window.
            - min_values (list): The minimum values in each non-overlapping window.
            - max_indices (list): The indices of the maximum values in each non-overlapping window.
            - max_values (list): The maximum values in each non-overlapping window.
    
    The function calculates the minimum and maximum value within each non-overlapping window of the specified size
    and returns the indices and values of these minima and maxima. Optionally, it can also create a dot plot showing
    the minimum and maximum values in the time series.
    
    Example usage:
        min_indices, min_values, max_indices, max_values = find_min_max_in_non_overlapping_window_with_threshold(signal, window_size=10)
    """

    def check_is_negatives(min_zscore_threshold) -> None:
        """
        Checks if either or both of the given values are negative.

        Args:
            min_zscore_threshold (float or int): The first value to check.

        Returns:
            str: A message indicating if values are positive.
        """
        if min_zscore_threshold > 0:
            print("`min_zscore_threshold` is positive. Ensure the value is negative for correct filtering")

    def filter_adjacent(indices, input_signal, window_size, is_maxima=True) -> list[int]:
        """
        Filters indices to remove adjacent points within a window, keeping maxima or minima.
        
        Args:
            indices (list): Indices of potential extrema.
            input_signal (array-like): The input signal to evaluate.
            window_size (int): Size of the window to filter adjacent points.
            is_maxima (bool): Whether to filter for maxima (True) or minima (False).
        
        Returns:
            list: Filtered list of indices.
        """
        filtered_indices = []
        last_index = -window_size // 2
        for index in indices:
            if index - last_index >= window_size // 2:
                filtered_indices.append(index)
                last_index = index
            else:
                if (is_maxima and input_signal[index] > input_signal[last_index]) or \
                    (not is_maxima and input_signal[index] < input_signal[last_index]):
                    filtered_indices[-1] = index
                    last_index = index
        return filtered_indices
    
    def cal_z_scores(values, z_score_multiplier=1.4826, epsilon=1e-9) -> np.ndarray:
        """
        Computes z-scores with protection against NaNs and zero MAD values.
        
        Args:
            values (array-like): Input data values.
            z_score_multiplier (float): Multiplier to convert MAD to standard deviation.
            epsilon (float): Small value to avoid division by zero.
        
        Returns:
            numpy array: Array of z-scores.
        """
        if np.isnan(values).all():
            return np.array([])
        median_val = np.nanmedian(values)
        mad_val = np.nanmedian(np.abs(values - median_val)) * z_score_multiplier# * 1.4826  # convert MAD to standard deviation equivalent
        if mad_val == 0:
            mad_val = epsilon
        z_scores = (values - median_val) / mad_val
        return z_scores
    

    # Checks if the min thresholds are negative -> filtering methods assume the min values to be negative
    check_is_negatives(min_zscore_threshold)
        
    
    # Keep track of NaN positions and remove NaNs from input_signal
    nan_mask = np.isnan(input_signal)
    clean_signal = input_signal[~nan_mask]

    # Calculate the number of non-overlapping windows
    num_windows = len(clean_signal) // window_size

    min_values, min_indices, max_values, max_indices = [], [], [], []

    for i in range(num_windows):
        window_start = i * window_size
        window_end = (i + 1) * window_size
        window = clean_signal[window_start:window_end]
        
        if np.isnan(window).all():
            continue

        min_value = np.nanmin(window)
        min_index = window_start + np.nanargmin(window)
        min_values.append(min_value)
        min_indices.append(min_index)

        max_value = np.nanmax(window)
        max_index = window_start + np.nanargmax(window)
        max_values.append(max_value)
        max_indices.append(max_index)

    min_indices = filter_adjacent(min_indices, clean_signal, window_size, is_maxima=False)
    max_indices = filter_adjacent(max_indices, clean_signal, window_size, is_maxima=True)
    min_values = clean_signal[min_indices]
    max_values = clean_signal[max_indices]

    # Calculating the z-scores for each values
    z_scores_min = cal_z_scores(min_values, z_score_multiplier)
    z_scores_max = cal_z_scores(max_values, z_score_multiplier)       
            
    # Apply z_score rejection critera
    min_values_kept = min_values[(z_scores_min > min_zscore_threshold) & (np.abs(z_scores_min) > min_zscore_threshold)]
    min_indices_kept = np.array(min_indices)[(z_scores_min > min_zscore_threshold) & (np.abs(z_scores_min) > min_zscore_threshold)]
    
    max_values_kept = max_values[(z_scores_max < max_zscore_threshold) & (np.abs(z_scores_max) < max_zscore_threshold)]
    max_indices_kept = np.array(max_indices)[(z_scores_max < max_zscore_threshold) & (np.abs(z_scores_max) < max_zscore_threshold)]
        
    
    # map the kept indices back to the original input signal
    original_min_indices_kept = np.where(~nan_mask)[0][min_indices_kept]
    original_max_indices_kept = np.where(~nan_mask)[0][max_indices_kept]

    if show_plot:
        import matplotlib.pyplot as plt
        t = np.arange(len(input_signal)) / fs

        plt.figure(figsize=(10, 6))
        plt.plot(t, input_signal, label='Filtered data', linewidth=1)
        plt.scatter(t[original_min_indices_kept], min_values_kept, color='red', marker='o', label='Minimum Values')
        plt.scatter(t[original_max_indices_kept], max_values_kept, color='green', marker='o', label='Maximum Values')
        plt.xlabel('Time (sec)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title('Min/Max in Non-Overlapping Windows')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    return original_min_indices_kept, min_values_kept, original_max_indices_kept, max_values_kept

def get_amplitude_envelope(filtered_signal : pd.Series, max_zscore_threshold=3,
                       min_zscore_threshold=-3,
                       low_cutoff=0.5,
                       high_cutoff=8.5, 
                       fs=500):
    # Find the dominate frequencey of the filtered signal
    dominant_freq = find_dominant_frequencies(filtered_signal, fs, 1, freq_min=low_cutoff, freq_max=high_cutoff, yscale_type='symlog', show_plot=False)[0][0]
    print(f'Calculated Dominant Freq: {dominant_freq:.2f}Hz [expected ~ 1Hz]')


    (min_indices_kept,
     min_values_kept,
     max_indices_kept, 
     max_values_kept) = find_min_max_in_non_overlapping_window_with_threshold(filtered_signal,
                                                      max_zscore_threshold=max_zscore_threshold, 
                                                      min_zscore_threshold=min_zscore_threshold, 
                                                      window_size=int(fs/dominant_freq),
                                                      fs=fs,
                                                      show_plot=False)
    
    # Interpolates the min and max points
    # Required as each min point will not aling with a max point
    max_line = inter_extra_polate_points(filtered_signal, 
                              max_indices_kept,
                              max_values_kept,
                              show_plot=False)

    min_line = inter_extra_polate_points(filtered_signal, 
                              min_indices_kept,
                              min_values_kept,
                              show_plot=False)
    
    return min_line, max_line