import os

import numpy as np
import matplotlib.pyplot as plt

from dataset_core.dataset import DataSet

def purging_analysis(dataset):
    """Analyzes the effect of purging on the THz signal by comparing scans taken at different times since purging start. Computes a deviation metric based on the average of the absolute difference between consecutive scans and plots it over time to visualize any trends.
    
    Final output is the decay exponential fit parameters for each file, which can be used to compare the purging dynamics across different samples or conditions."""

    from scipy.optimize import curve_fit
    def exp_decay(t, a, b):
        return a * np.exp(-b * t)

    dev_dict = {}
    decays = []
    for filename, data_obj in dataset.data.items():
        # breakpoint()
        dataX = data_obj.raw_data[:, 0]
        # plt.plot()
        dev_list = []
        # mask = [True if (x > 154) and (x < 158) else False for x in dataX]
        mask = [True for x in dataX]  # use all data points

        timestamps = data_obj.resolve_timestamps()
        normalised_time = np.array([(t - timestamps[0]).total_seconds() for t in timestamps]) if timestamps is not None else None
        for index in range(data_obj.raw_data.shape[1]):
            timestamp = normalised_time[index-1] if index > 0 else None
            if index == 0:
                last_trace = np.zeros_like(dataX[mask])
                continue
            dataY = data_obj.raw_data[mask, index]
            difference = np.abs(dataY - last_trace)
            last_trace = np.copy(dataY)
            metric = np.mean(difference)  # average absolute difference from previous trace
            dev_list.append([timestamp, metric])
        dev_dict[filename] = dev_list

    for filename, dev_list in dev_dict.items():
        data = np.array(dev_list)
        data[:, 1] -= data[:, 1].min()  # shift baseline to zero
        plt.scatter(data[:, 0], data[:, 1], label=filename)
        
        # Fit exponential decay        
        try:
            popt, _ = curve_fit(exp_decay, data[:, 0], data[:, 1], p0=[data[0, 1], 0.1], maxfev=10000)
            t_fit = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
            y_fit = exp_decay(t_fit, *popt)
            plt.plot(t_fit, y_fit, label=f'{filename} (fit: a={popt[0]:.2e}, b={popt[1]:.2e})', linestyle='--')

            # print(f"{filename}: Fitted exponential decay parameters: a = {popt[0]:.2e}, b = {popt[1]:.2e}")
            decays = decays + [(filename, popt[0], popt[1])]
        except:
            pass

    sorted_decays = sorted(decays, key=lambda x: x[2])[::-1]  # sort by decay rate (b)
    print("Purging analysis complete. Decay parameters (sorted by decay rate, fastest first):")
    for filename, a, b in sorted_decays:
        print(f"{filename}: a = {a:.2e}, b = {b:.2e}")
        
    plt.legend()
    plt.title("Deviation Metric Over Time")
    plt.xlabel("Timestamp")
    plt.ylabel("Average Absolute Difference from Previous Trace")
    plt.show()


if __name__ == "__main__":
    dataset = DataSet(r"C:\Users\Samuel\Data\THz\Sam\2026-06-11_CNT-paper\purge")
    dataset.load_all_data(case_insensitive=True)
    purging_analysis(dataset)

