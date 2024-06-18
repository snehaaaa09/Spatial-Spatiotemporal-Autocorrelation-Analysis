# Imports
import pandas as pd
import numpy as np


def year_df_creation(final_df, year):
    """Create a dataframe of the respective years to act as time series in SAC calculation"""
    time_df = final_df[final_df['TIME_PERIOD'] == year]
    return time_df


def region_df_creation(final_df, region):
    """Create a dataframe of the respective regions to act as time series in SAC calculation"""
    region_df = final_df[final_df['NUTS_ID'] == region]
    return region_df


def even_time_period_GDP(gdf_lvl):
    """
    Create a dataframe for GDP data where each region has equal number of time periods (needed for STAC analysis!)
    :param gdf_lvl:
    :return: dataframe of GDP data with regions having equal time periods
    """
    # Convert 'TIME_PERIOD' column to numeric if it's stored as strings
    gdf_lvl['TIME_PERIOD'] = pd.to_numeric(gdf_lvl['TIME_PERIOD'])
    # Filter the DataFrame to include only rows from 2004 to 2021
    filtered_gdf = gdf_lvl[gdf_lvl['TIME_PERIOD'] == 2004]
    nuts_regions = list(filtered_gdf['NUTS_ID'])
    # Filter the DataFrame to include only rows from 2004 to 2021
    filtered_gdf_lvl2 = gdf_lvl[(gdf_lvl['TIME_PERIOD'] >= 2004) & (gdf_lvl['TIME_PERIOD'] <= 2021)]
    # Filter the DataFrame to include only the regions from the list
    final_df = filtered_gdf_lvl2[filtered_gdf_lvl2['NUTS_ID'].isin(nuts_regions)]

    # Making sure that each region has data for each year from 2004 to 2021 (equal length of time series!)
    # Define the list of required years
    required_years = set(range(2004, 2022))

    # Find NUTS_IDs with complete years

    def has_all_years(group):
        return required_years.issubset(group['TIME_PERIOD'].unique())

    # Filter NUTS_IDs
    valid_nuts_ids = final_df.groupby('NUTS_ID').filter(has_all_years)['NUTS_ID'].unique()
    # Convert to list if needed
    valid_nuts_ids_list = valid_nuts_ids.tolist()
    final_df = final_df[final_df['NUTS_ID'].isin(valid_nuts_ids_list)]

    return final_df


# STAC functions
def corr_time_series(X_T, Y_T):
    "Calculate the correlation between time series data of pairs of locations"
    # Convert data frame columns to numpy arrays
    X_T = X_T.values
    Y_T = Y_T.values
    # Compute the difference between consecutive elements in the time series
    diff_X = X_T[1:] - X_T[:-1]
    diff_Y = Y_T[1:] - Y_T[:-1]
    # Compute the product of the differences
    product_diff = diff_X * diff_Y
    # Sum the products from t to T-1
    sum_product = np.sum(product_diff)

    # Square the differences
    squared_diff_X = diff_X ** 2
    squared_diff_Y = diff_Y ** 2
    # Compute the square root of the squared differences
    sqrt_squared_diff_X = np.sqrt(squared_diff_X)
    sqrt_squared_diff_Y = np.sqrt(squared_diff_Y)
    # Compute the product of the square roots
    product_sqrt = sqrt_squared_diff_X * sqrt_squared_diff_Y
    # Sum the products from t to T-1
    sum_product_sqrt = np.sum(product_sqrt)

    # Calculate the correlation between the time series
    # Check if sum_product_sqrt is zero
    if sum_product_sqrt != 0:
        corr_value = sum_product / sum_product_sqrt
    else:
        corr_value = "undefined"
    return corr_value


def deviation_time_series(corr_value, X_T, Y_T):
    "Calculate the deviation between time series from correlation value calculated before"
    # Calculate sum of X_T and Y_T
    vol_X_T = np.sum(X_T)
    vol_Y_T = np.sum(Y_T)
    # Define the adaptive tuning function
    if corr_value != "undefined":
        tuned_corr_value = 2 / (1 + np.exp(2 * corr_value))
    else:
        tuned_corr_value = "undefined"

    # Calculate deviation between time series
    if tuned_corr_value != "undefined":
        deviation_value = tuned_corr_value * (vol_X_T - vol_Y_T)
    else:
        deviation_value = "undefined"
    return deviation_value


def adjusted_time_series(corr_value, X_T):
    """Calculate the time series of each region adjusted for via correlation between time series and the mean series"""
    # Calculate sum of X_T
    vol_X_T = np.sum(X_T)
    # Define the adaptive tuning function
    if corr_value != "undefined":
        tuned_corr_value = 2 / (1 + np.exp(2 * corr_value))
    else:
        tuned_corr_value = "undefined"

    # Calculate deviation between time series
    if tuned_corr_value != "undefined":
        adjusted_value = tuned_corr_value * vol_X_T
    else:
        adjusted_value = "undefined"
    return adjusted_value


def global_STAC(neighbor_weights_dict, final_df, region_df, variable):
    """Calculate the global spatiotemporal autocorrelation value"""
    # Initializing variables
    spatial_w_sum = 0.0
    numerator = 0.0
    denominator = 0.0
    # Calculate mean GDP over all regions for each year
    # Group by TIME_PERIOD and calculate the mean GDP value summed over all regions for each year
    mean_gdp_sum_per_year = final_df.groupby('TIME_PERIOD')[variable].mean()
    # Convert the resulting series to a DataFrame
    mean_gdp_df = mean_gdp_sum_per_year.to_frame(name='Mean_variable')
    # Calculate sum of all spatial weights
    # Iterate over all regions
    for region, neighbor_weights in neighbor_weights_dict.items():
        # Iterate over all neighbor weights for the current region
        for weight in neighbor_weights.values():
            # Add the weight to the total sum
            spatial_w_sum += weight
    # Calculate total number of regions
    N = len(region_df)
    first_term = N / spatial_w_sum

    for region_i, neighbor_weights in neighbor_weights_dict.items():
        # Iterate over all neighbor weights for the current region
        time_series_i = region_df_creation(final_df, region_i)
        corr_i = corr_time_series(time_series_i[variable], mean_gdp_df['Mean_variable'])
        # counter variable for neighbors' indices
        count = 0
        for weight in neighbor_weights.values():
            neighbor_regions = list(neighbor_weights)
            region_j = neighbor_regions[count]
            time_series_j = region_df_creation(final_df, region_j)
            corr_j = corr_time_series(time_series_j[variable], mean_gdp_df['Mean_variable'])
            numerator += (weight * deviation_time_series(corr_i, time_series_i[variable], mean_gdp_df['Mean_variable'])
                          * deviation_time_series(corr_j, time_series_j[variable], mean_gdp_df['Mean_variable']))
            count += 1
        dev_time_series_i_squared = (deviation_time_series(corr_i, time_series_i[variable],
                                                           mean_gdp_df['Mean_variable'])) ** 2
        denominator += dev_time_series_i_squared

    # Calculate the final global STAC
    global_STAC_val = first_term * (numerator / denominator)
    return global_STAC_val


def global_STAC_val(neighbor_weights_dict, final_df, variable, mean_gdp_array, first_term):
    """Calculate the global spatiotemporal autocorrelation value"""
    numerator = 0.0
    denominator = 0.0
    for region_i, neighbor_weights in neighbor_weights_dict.items():
        # Iterate over all neighbor weights for the current region
        time_series_i = region_df_creation(final_df, region_i)
        corr_i = corr_time_series(time_series_i[variable], mean_gdp_array)
        dev_time_series_i = deviation_time_series(corr_i, time_series_i[variable], mean_gdp_array)
        # counter variable for neighbors' indices
        count = 0
        for weight in neighbor_weights.values():
            neighbor_regions = list(neighbor_weights)
            region_j = neighbor_regions[count]
            time_series_j = region_df_creation(final_df, region_j)
            corr_j = corr_time_series(time_series_j[variable], mean_gdp_array)
            numerator += (weight * dev_time_series_i
                          * deviation_time_series(corr_j, time_series_j[variable], mean_gdp_array))
            count += 1
        dev_time_series_i_squared = dev_time_series_i ** 2
        denominator += dev_time_series_i_squared

    # Calculate the final global STAC
    global_STAC_val = first_term * (numerator / denominator)
    return global_STAC_val


def local_STAC(neighbor_weights_dict, final_df, region_df, variable):
    """Calculate the local spatiotemporal autocorrelation values for each region"""
    # Initializing values
    denominator = 0.0  # to store denominator of each LISA value
    LISA_dict = dict()  # to store all LISA values with its region IDs
    # Calculate mean GDP over all regions for each year
    # Group by TIME_PERIOD and calculate the mean GDP value summed over all regions for each year
    mean_gdp_sum_per_year = final_df.groupby('TIME_PERIOD')[variable].mean()
    # Convert the resulting series to a DataFrame
    mean_gdp_df = mean_gdp_sum_per_year.to_frame(name='Mean_variable')
    # Calculate total number of regions
    N = len(region_df)
    # Calculate denominator (same for each region)
    for region_i, neighbor_weights in neighbor_weights_dict.items():
        time_series_i = region_df_creation(final_df, region_i)
        corr_i = corr_time_series(time_series_i[variable], mean_gdp_df['Mean_variable'])
        dev_i = deviation_time_series(corr_i, time_series_i[variable], mean_gdp_df['Mean_variable'])
        dev_i_squared = dev_i ** 2
        denominator += dev_i_squared
    # Calculate numerator by iterating through all neighbors for each region
    for region_i, neighbor_weights in neighbor_weights_dict.items():
        time_series_i = region_df_creation(final_df, region_i)
        corr_i = corr_time_series(time_series_i[variable], mean_gdp_df['Mean_variable'])
        dev_i = deviation_time_series(corr_i, time_series_i[variable], mean_gdp_df['Mean_variable'])
        # Initialize sum
        sum_j = 0.0
        # Counter variable for neighbors' indices
        count = 0
        for weight in neighbor_weights.values():
            neighbor_regions = list(neighbor_weights)
            region_j = neighbor_regions[count]
            time_series_j = region_df_creation(final_df, region_j)
            corr_j = corr_time_series(time_series_j[variable], mean_gdp_df['Mean_variable'])
            dev_j = deviation_time_series(corr_j, time_series_j[variable], mean_gdp_df['Mean_variable'])
            sum_j += weight * dev_j
            count += 1
        numerator = N * dev_i * sum_j
        local_val = numerator / denominator
        # Storing lisa value for each region in dictionary
        LISA_dict[region_i] = local_val
    return LISA_dict


def local_STAC_per_region(neighbor_weights_dict, final_df, region_df, variable, region_i, denominator, mean_gdp):
    """Calculate the local spatiotemporal autocorrelation value for a specific region"""
    # Calculate total number of regions
    N = len(region_df)
    # Calculate numerator for the specified region
    neighbor_weights = neighbor_weights_dict[region_i]
    time_series_i = region_df_creation(final_df, region_i)
    corr_i = corr_time_series(time_series_i[variable], mean_gdp)
    dev_i = deviation_time_series(corr_i, time_series_i[variable], mean_gdp)
    # Initialize sum
    sum_j = 0.0
    # Counter variable for neighbors' indices
    count = 0
    for weight in neighbor_weights.values():
        neighbor_regions = list(neighbor_weights)
        region_j = neighbor_regions[count]
        time_series_j = region_df_creation(final_df, region_j)
        corr_j = corr_time_series(time_series_j[variable], mean_gdp)
        dev_j = deviation_time_series(corr_j, time_series_j[variable], mean_gdp)
        sum_j += weight * dev_j
        count += 1

    numerator = N * dev_i * sum_j
    local_val = numerator / denominator

    return local_val


