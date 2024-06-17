import pandas as pd
import numpy as np


# Dummy functions for region_df_creation, corr_time_series, and deviation_time_series
def region_df_creation(df, region):
    return df[df['NUTS_ID'] == region]


def corr_time_series(X_T, Y_T):
    diff_X = np.diff(X_T)
    diff_Y = np.diff(Y_T)
    sum_product = np.sum(diff_X * diff_Y)
    sum_product_sqrt = np.sqrt(np.sum(diff_X ** 2) * np.sum(diff_Y ** 2))
    return sum_product / sum_product_sqrt if sum_product_sqrt != 0 else 0


def deviation_time_series(corr_value, X_T, Y_T):
    vol_X_T = np.sum(X_T)
    vol_Y_T = np.sum(Y_T)
    tuned_corr_value = 2 / (1 + np.exp(2 * corr_value)) if corr_value != 0 else 0
    return tuned_corr_value * (vol_X_T - vol_Y_T)


def adjusted_time_series(corr_value, X_T):
    vol_X_T = np.sum(X_T)
    tuned_corr_value = 2 / (1 + np.exp(2 * corr_value)) if corr_value != 0 else 0
    return tuned_corr_value * vol_X_T


def global_STAC(neighbor_weights_dict, final_df, region_df, variable):
    mean_gdp_df = final_df.groupby('TIME_PERIOD')[variable].mean().to_numpy()
    final_array = final_df[['NUTS_ID', 'TIME_PERIOD', variable]].values

    N = len(region_df)
    spatial_w_sum = sum(
        weight for neighbor_weights in neighbor_weights_dict.values() for weight in neighbor_weights.values())
    first_term = N / spatial_w_sum

    numerator = 0.0
    denominator = 0.0

    region_deviations = {}
    for region in neighbor_weights_dict.keys():
        time_series_i = final_array[final_array[:, 0] == region][:, 2]
        corr_i = corr_time_series(time_series_i, mean_gdp_df)
        dev_i = deviation_time_series(corr_i, time_series_i, mean_gdp_df)
        region_deviations[region] = dev_i

    for region_i, neighbor_weights in neighbor_weights_dict.items():
        dev_i = region_deviations[region_i]
        for region_j, weight in neighbor_weights.items():
            dev_j = region_deviations[region_j]
            numerator += weight * dev_i * dev_j
        denominator += dev_i ** 2

    global_STAC_val = first_term * (numerator / denominator)
    return global_STAC_val


def local_STAC(neighbor_weights_dict, final_df, region_df, variable):
    mean_gdp_df = final_df.groupby('TIME_PERIOD')[variable].mean().to_numpy()
    final_array = final_df[['NUTS_ID', 'TIME_PERIOD', variable]].values

    N = len(region_df)
    denominator = 0.0
    LISA_dict = {}

    region_deviations = {}
    for region in neighbor_weights_dict.keys():
        time_series_i = final_array[final_array[:, 0] == region][:, 2]
        corr_i = corr_time_series(time_series_i, mean_gdp_df)
        dev_i = deviation_time_series(corr_i, time_series_i, mean_gdp_df)
        region_deviations[region] = dev_i
        denominator += dev_i ** 2

    for region_i, neighbor_weights in neighbor_weights_dict.items():
        dev_i = region_deviations[region_i]
        sum_j = sum(weight * region_deviations[region_j] for region_j, weight in neighbor_weights.items())
        numerator = N * dev_i * sum_j
        local_val = numerator / denominator
        LISA_dict[region_i] = local_val

    return LISA_dict


def local_STAC_per_region_test2(neighbor_weights_dict, final_df, region_df, variable, region_i):
    mean_gdp_df = final_df.groupby('TIME_PERIOD')[variable].mean().to_numpy()
    final_array = final_df[['NUTS_ID', 'TIME_PERIOD', variable]].values

    N = len(region_df)
    denominator = 0.0

    region_deviations = {}
    for region in neighbor_weights_dict.keys():
        time_series_i = final_array[final_array[:, 0] == region][:, 2]
        corr_i = corr_time_series(time_series_i, mean_gdp_df)
        dev_i = deviation_time_series(corr_i, time_series_i, mean_gdp_df)
        region_deviations[region] = dev_i
        denominator += dev_i ** 2

    dev_i = region_deviations[region_i]
    neighbor_weights = neighbor_weights_dict[region_i]
    sum_j = sum(weight * region_deviations[region_j] for region_j, weight in neighbor_weights.items())
    numerator = N * dev_i * sum_j
    local_val = numerator / denominator

    return local_val
