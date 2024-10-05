# Loading in the Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import read


from src.utilis import load_dataframe, get_file_names, get_timed_window, get_mseed_path, get_sample_rate
from src.signal_processing import lowpass_filter, highpass_filter, get_amplitude_envelope

def rolling_avg(data, window):
    """
    Calculate rolling average for a given data series.

    Args:
        data (pd.Series): The input data series.
        window (int): The size of the moving window.

    Returns:
        tuple of pd.Series: The lower and upper quantiles for the rolling window.

    Example use:
        lower_quantile, upper_quantile = rolling_quantiles(data, window=100, quantiles=[0.025, 0.975])
    """
    data = pd.Series(data)
    rolling = data.rolling(window).mean()
    return rolling

# Looking at filtering the data
data_path = "E:/Users/William/Uni/Swinburne OneDrive/OneDrive - Swinburne University/2024 NASA Space Apps Challange/Data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1973-06-05HR00_evid00107.csv"

data = load_dataframe(data_path)

# Gets the data window
# data = get_timed_window(data, 0,12000)

# high and low pass onto the data

sample_freq = get_sample_rate(get_mseed_path(data_path))
data_high = highpass_filter(data['velocity(m/s)'], 0.8, sample_freq)
data_high_low = lowpass_filter(data_high, 0.6, sample_freq)

fig, ax = plt.subplots(4, figsize=(7, 12), sharex=True, sharey=True)

ax[0].plot(data['time_rel(sec)'], data['velocity(m/s)'], label="Raw")
ax[0].set_ylabel('Velocity (m/s)')
ax[0].set_xlabel('Time (s)')
ax[0].legend()

ax[1].plot(data['time_rel(sec)'], data_high_low, label="Band Pass")
ax[1].set_ylabel('Velocity (m/s)')
ax[1].set_xlabel('Time (s)')
ax[1].legend()

# Taking only the maxium values
diff = np.diff(data_high_low)
ax[2].plot(data['time_rel(sec)'][:-1], diff, label="Difference")
ax[2].set_ylabel('Velocity (m/s)')
ax[2].set_xlabel('Time (s)')
ax[2].legend()


ddiff = np.diff(diff)
ax[3].plot(data['time_rel(sec)'][:-2], ddiff, label="Double Difference")
ax[3].set_ylabel('Velocity (m/s)')
ax[3].set_xlabel('Time (s)')
ax[3].legend()


fig.tight_layout()
plt.show()
