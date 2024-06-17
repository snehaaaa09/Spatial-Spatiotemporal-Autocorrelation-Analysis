# Imports
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import folium
from shapely.geometry import Point
from pysal.lib import weights
from pysal.explore import esda
import seaborn
from esda.moran import Moran_Local
from splot import esda as esdaplot
import time


def data_GDP(level, time_period_option):
    """
    :param level: integer value indicating NUTS level 2 or level 3 for GDP data collected
    :param time_period_option: string value indicating the time period option ('year_2021', 'all_years', 'average_all_years')
    :return: cleaned dataframe of GDP values, including corresponding NUTS geometric information from shapefile data
    """
    gdp_df = pd.read_csv("data/GDP(Nuts" + str(level) + ")/nama_10r_" + str(level) + "gdp_linear.csv")
    # Choosing a unit - PPS (Purchasing Power Standards) per inhabitant
    gdp_df = gdp_df[gdp_df['unit'] == 'PPS_EU27_2020_HAB']
    # Renaming geo column and OBS_VALUE column
    gdp_df.rename(columns={'geo': 'NUTS_ID'}, inplace=True)
    gdp_df.rename(columns={'OBS_VALUE': 'GDP_VALUE'}, inplace=True)
    # Removing unnecessary columns (only one value for all rows)
    gdp_df = gdp_df.drop(columns=['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'OBS_FLAG'])
    # Convert all values to a specific type (e.g., int)
    gdp_df['NUTS_ID'] = gdp_df['NUTS_ID'].astype(object)
    gdp_df['TIME_PERIOD'] = gdp_df['TIME_PERIOD'].astype(int)
    gdp_df['GDP_VALUE'] = gdp_df['GDP_VALUE'].astype(float)

    # Checking if average of GDP values is required
    if time_period_option == 'average_all_years':
        # First, let's remove outliers from each NUTS_ID group
        def remove_outliers(group):
            q1 = group['GDP_VALUE'].quantile(0.25)
            q3 = group['GDP_VALUE'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return group[(group['GDP_VALUE'] >= lower_bound) & (group['GDP_VALUE'] <= upper_bound)]

        # Apply the function to each group and concatenate the results
        gdp_df = gdp_df.groupby('NUTS_ID').apply(remove_outliers).reset_index(drop=True)
        # Now, let's aggregate the cleaned data by taking the mean for each NUTS_ID
        gdp_df = gdp_df.groupby('NUTS_ID')['GDP_VALUE'].mean().reset_index()
    elif time_period_option == 'time_2021':
        # Choosing ONLY most recent years (2021)
        gdp_df = gdp_df[gdp_df['TIME_PERIOD'] == 2021]
    elif time_period_option == 'all_years':
        pass

    # Reading the shapefile data (ESPG:3035 - projected CRS needed for centroid calculation!)
    gdf_3035 = gpd.read_file("data/NUTS_RG_20M_2021_3035.shp/NUTS_RG_20M_2021_3035.shp")
    # Reading the shapefile data (ESPG:4326 - GEOGRAPHIC COORDINATE SYSTEM)
    gdf_4326 = gpd.read_file("data/NUTS_RG_20M_2021_4326.shp/NUTS_RG_20M_2021_4326.shp")
    # Merging gdp_2021 data with shapefile data
    merged_df = pd.merge(gdp_df, gdf_3035, on='NUTS_ID', how='left')
    # Extracting ONLY nuts level 2 regions!
    gdp_lvl_df = merged_df[merged_df['LEVL_CODE'] == level]
    # Selecting only the required columns
    if time_period_option == 'all_years':
        gdp_lvl = gdp_lvl_df[['NUTS_ID', 'NUTS_NAME', 'NAME_LATN', 'TIME_PERIOD', 'GDP_VALUE', 'geometry']].copy()
    else:
        gdp_lvl = gdp_lvl_df[['NUTS_ID', 'NUTS_NAME', 'NAME_LATN', 'GDP_VALUE', 'geometry']].copy()
    # Create a GeoDataFrame
    gdp_lvl = gpd.GeoDataFrame(gdp_lvl, geometry='geometry')

    ##-----Centroid Calculation-------##
    # Now, calculate the centroids (but float values!)
    gdp_lvl['centroid_lon'] = gdp_lvl.geometry.centroid.x
    gdp_lvl['centroid_lat'] = gdp_lvl.geometry.centroid.y

    # dropping na values
    gdp_lvl = gdp_lvl.dropna()
    gdf_lvl = gpd.GeoDataFrame(gdp_lvl, geometry='geometry')
    # Converting geo df into coordinate system (only for visualizations!)
    gdf_lvl.crs = "EPSG:3035"
    gdf_vis = gdf_lvl.to_crs(epsg=4326)

    return gdf_lvl


def data_explanatory(standardize, time_period_option):
    """
    :param standardize: boolean value indicating whether to standardize the variables to allow for easier comparison of
    coefficient values
    :param time_period_option: string value indicating the time period option ('year_2021', 'all_years', 'average_all_years')
    :return: both the list of explanatory variables and the final explanatory variables dataframe
    """

    # Define a function to process each DataFrame
    def process_df(file_path, new_col_name, time_period_option, value_col_name='OBS_VALUE', geo_col_name='geo',
                   additional_filter=None):
        df = pd.read_csv(file_path)

        # Apply additional filters before any time period operations
        if additional_filter:
            for col, val in additional_filter.items():
                if col in df.columns:
                    df = df[df[col] == val]
                else:
                    raise KeyError(f"Column '{col}' not found in the DataFrame.")

        df = df[[geo_col_name, value_col_name, 'TIME_PERIOD']]
        if time_period_option == 'year_2021':
            df = df[df['TIME_PERIOD'] == 2021]
            df = df[[geo_col_name, value_col_name]]
        elif time_period_option == 'average_all_years':
            # First, let's remove outliers from each NUTS_ID group
            def remove_outliers(group):
                q1 = group['OBS_VALUE'].quantile(0.25)
                q3 = group['OBS_VALUE'].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                return group[(group['OBS_VALUE'] >= lower_bound) & (group['OBS_VALUE'] <= upper_bound)]

            # Apply the function to each group and concatenate the results
            df = df.groupby(geo_col_name).apply(remove_outliers).reset_index(drop=True)
            # Now, let's aggregate the cleaned data by taking the mean for each geocode
            df = df.groupby([geo_col_name], as_index=False)[value_col_name].mean()
            df = df[[geo_col_name, value_col_name]]
        elif time_period_option == 'all_years':
            df = df[[geo_col_name, 'TIME_PERIOD', value_col_name]]
            df['TIME_PERIOD'] = df['TIME_PERIOD'].astype(int)
        else:
            print("Valid input needed")

        # Converting all values in OBS_VALUE column to float type
        df['OBS_VALUE'] = df['OBS_VALUE'].astype(float)
        # Renaming columns
        df.rename(columns={value_col_name: new_col_name, geo_col_name: 'NUTS_ID'}, inplace=True)
        return df

    # Define file paths and column names
    file_info = [
        ("data/X_data(Nuts2)/Disposible_income(Nuts2)/tgs00026_linear.csv", 'DISP_INCOME_VAL'),
        ("data/X_data(Nuts2)/Primary_income(Nuts2)/tgs00036_linear.csv", 'PRIM_INCOME_VAL'),
        ("data/X_data(Nuts2)/Population_density(Nuts2)/tgs00024_linear.csv", 'POP_DENS_VAL'),
        ("data/X_data(Nuts2)/Population_jan1(Nuts2)/tgs00096_linear.csv", 'POP_COUNT_VAL'),
        ("data/X_data(Nuts2)/Total_death(Nuts2)/tgs00057_linear.csv", 'TOTAL_DEATH_VAL'),
        ("data/X_data(Nuts2)/Tourism_places(Nuts2)/tgs00112_linear.csv", 'TOUR_PLCS_VAL', {'accomunit': 'ESTBL'}),
        ("data/X_data(Nuts2)/Nights_spent_in_tourism_place(Nuts2)/tgs00111_linear.csv", 'TOUR_NIGHTS_VAL',
         {'c_resid': 'TOTAL'}),
        ("data/X_data(Nuts2)/Science_Tech(Nuts2)/Sci_tech_pop.csv", 'SCI_TECH_VAL'),
        ("data/X_data(Nuts2)/Teriary_edu(Nuts2)/tgs00109_linear.csv", 'EDU_VAL', {'sex': 'T'})
    ]

    # Process each file and store the resulting DataFrames in a list
    processed_dfs = []
    for file_path, value_col_name, *optional_filter in file_info:
        additional_filter = optional_filter[0] if optional_filter else None
        processed_df = process_df(file_path, value_col_name, time_period_option=time_period_option,
                                  additional_filter=additional_filter)
        processed_dfs.append(processed_df)

    # You can assign them to specific variable names if needed
    disp_income_df, prim_income_df, pop_density_df, pop_ct_df, death_all_df, tourism_places_df, tourism_nights_df, \
    sci_tech_df, edu_df = processed_dfs

    # Creating the explanatory variable dataframe
    # List of DataFrames to merge
    dfs_to_merge = [
        disp_income_df,
        prim_income_df,
        pop_density_df,
        pop_ct_df,
        death_all_df,
        tourism_places_df,
        tourism_nights_df,
        sci_tech_df,
        edu_df
    ]

    # Start with the first DataFrame
    X_df = dfs_to_merge[0]
    # Merge each subsequent DataFrame into the merged_df
    for df in dfs_to_merge[1:]:
        if time_period_option != 'all_years':
            X_df = pd.merge(X_df, df, on='NUTS_ID', how='inner')
        else:
            X_df = pd.merge(X_df, df, on=['NUTS_ID', 'TIME_PERIOD'], how='inner')

    # Dropping na values
    X_df = X_df.dropna()
    # Identify the explanatory variables
    if time_period_option != 'all_years':
        explanatory_vars = X_df.columns.difference(['NUTS_ID'])
    else:
        explanatory_vars = X_df.columns.difference(['NUTS_ID', 'TIME_PERIOD'])
    # Standardizing the explanatory variables
    if standardize:
        # Initialize the scaler
        scaler = StandardScaler()
        # Fit the scaler on the explanatory variables and transform them (standard normal scaling
        # to have mean of 0 and a standard deviation of 1.
        X_df[explanatory_vars] = scaler.fit_transform(X_df[explanatory_vars])

    return X_df, explanatory_vars

