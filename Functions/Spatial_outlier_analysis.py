# User-defined function imports
from Functions.spatial_functions import spatial_weight_matrix, global_moran_val, local_moran_val, lisa_df_update, \
    neighborhood_dict_creation, lisa_update_STAC
from Functions.spatiotemporal_autocorrelation import year_df_creation, region_df_creation, even_time_period_GDP, \
    global_STAC, local_STAC

# Other imports
import pandas as pd
import numpy as np
from esda.moran import Moran_Local
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


def explanatory_HL_LH_df_creation(exp_var_list, final_df):
    """
    :param exp_var_list: list of explanatory variables
    :param final_df: dataframe of explanatory variables merged with shapefile information
    :return: numerous dictionaries that contain information of high-low and low-high regions of explanatory variables,
    global and local moran values of each variable
    """
    # Initializing variables for results
    global_moran_dict = {}
    local_moran_percent_dict = {}
    local_moran_dict = {}
    LH_region_dict = {}
    HL_region_dict = {}
    HH_region_dict = {}
    LL_region_dict = {}

    for exp_var in exp_var_list:
        exp_var_df = final_df[['NUTS_ID', 'NAME_LATN', 'NUTS_NAME', 'geometry', exp_var]]
        # Calculating W, global and local moran values and percentage of positive lisa values
        w_adaptive_exp = spatial_weight_matrix(exp_var_df, 15)
        global_moran_exp = global_moran_val(exp_var_df, exp_var, w_adaptive_exp)
        # Adding the global moran value of explanatory variable to results
        global_moran_dict[exp_var] = (global_moran_exp.I, global_moran_exp.p_sim)
        lisa_exp = local_moran_val(exp_var_df, exp_var, w_adaptive_exp)
        # Adding lisa object of explanatory variable to results
        local_moran_dict[exp_var] = lisa_exp
        # Calculate percentage of lisa values above zero
        lisa_list = list(lisa_exp.Is)
        count_above_zero = sum(1 for value in lisa_list if value > 0)
        total_values = len(lisa_list)
        percentage = (count_above_zero / total_values) * 100
        # Adding positive lisa percentage of explanatory variable to result
        local_moran_percent_dict[exp_var] = percentage
        # Reset index to update dataframe with lisa values and quadrant info
        exp_var_df.reset_index(inplace=True)
        # Updating dataframe
        exp_var_df = lisa_df_update(exp_var_df, lisa_exp)
        # significant LISA regions
        exp_var_df = exp_var_df[exp_var_df['LISA_sig'] == 'Significant']
        # LH regions
        LH_exp_df = exp_var_df[exp_var_df['LISA_quadrant'] == 2]
        LH_exp_df = LH_exp_df.drop(columns=['LISA_sig', 'LISA_quadrant'])
        LH_region_dict[exp_var] = LH_exp_df
        # HL regions
        HL_exp_df = exp_var_df[exp_var_df['LISA_quadrant'] == 4]
        HL_exp_df = HL_exp_df.drop(columns=['LISA_sig', 'LISA_quadrant'])
        HL_region_dict[exp_var] = HL_exp_df
        # HH regions
        HH_exp_df = exp_var_df[exp_var_df['LISA_quadrant'] == 1]
        HH_exp_df = HH_exp_df.drop(columns=['LISA_sig', 'LISA_quadrant'])
        HH_region_dict[exp_var] = HH_exp_df
        # LL regions
        LL_exp_df = exp_var_df[exp_var_df['LISA_quadrant'] == 3]
        LL_exp_df = LL_exp_df.drop(columns=['LISA_sig', 'LISA_quadrant'])
        LL_region_dict[exp_var] = LL_exp_df

    # Returning all the results
    return global_moran_dict, local_moran_percent_dict, local_moran_dict, LH_region_dict, HL_region_dict, HH_region_dict \
        , LL_region_dict


def explanatory_STAC_HL_LH_df_creation(exp_var_list, final_df, region_df, neighbor_weights_dict, w_adaptive_exp):
    """
    :param w_adaptive_exp: spatial weight matrix defined for regions of explanatory data
    :param neighbor_weights_dict: dictionary of neighborhood weights for each region
    :param region_df: dataframe of unique regions with geometry information
    :param exp_var_list: list of explanatory variables
    :param final_df: dataframe of explanatory variables merged with shapefile information
    :return: numerous dictionaries that contain information of high-low and low-high regions of explanatory variables,
    global and local moran values of each variable
    """
    # Initializing variables for results
    global_moran_dict = {}
    local_moran_percent_dict = {}
    local_moran_dict = {}
    LH_region_dict = {}
    HL_region_dict = {}
    HH_region_dict = {}
    LL_region_dict = {}

    for exp_var in exp_var_list:
        exp_var_df = final_df[['NUTS_ID', 'NAME_LATN', 'NUTS_NAME', 'TIME_PERIOD', 'geometry', exp_var]]
        # Global STAC value calculation
        global_moran_exp = global_STAC(neighbor_weights_dict, exp_var_df, region_df, exp_var)
        # Adding the global moran value of explanatory variable to results
        global_moran_dict[exp_var] = global_moran_exp
        lISA_dict = local_STAC(neighbor_weights_dict, exp_var_df, region_df, exp_var)
        # Creating a numpy array of lisa values
        lisa_values = np.array(list(lISA_dict.values()))
        # Creating a lisa object using manually calculated LISA values
        lisa_exp = Moran_Local(lisa_values, w_adaptive_exp)  # type: ignore
        lisa_exp.Is = lisa_values
        # Adding lisa object of explanatory variable to results
        local_moran_dict[exp_var] = lisa_exp.Is
        # Calculate percentage of lisa values above zero
        lisa_list = list(lisa_exp.Is)
        count_above_zero = sum(1 for value in lisa_list if value > 0)
        total_values = len(lisa_list)
        percentage = (count_above_zero / total_values) * 100
        # Adding positive LISA percentage of explanatory variable to result
        local_moran_percent_dict[exp_var] = percentage
        # Extract only singular regions since STAC values calculated for each region
        exp_var_df = year_df_creation(exp_var_df, 2015)
        exp_var_df = exp_var_df.drop(columns=['TIME_PERIOD'])
        # Updating dataframe to include STAC LISA values for each region
        exp_var_df = lisa_update_STAC(exp_var_df, final_df, lISA_dict, w_adaptive_exp, exp_var)[0]
        # LH regions
        LH_exp_df = exp_var_df[exp_var_df['LISA_quadrant'] == 2]
        LH_exp_df = LH_exp_df.drop(columns=['Quadrant_name', 'LISA_quadrant'])
        LH_region_dict[exp_var] = LH_exp_df
        # HL regions
        HL_exp_df = exp_var_df[exp_var_df['LISA_quadrant'] == 4]
        HL_exp_df = HL_exp_df.drop(columns=['Quadrant_name', 'LISA_quadrant'])
        HL_region_dict[exp_var] = HL_exp_df
        # HH regions
        HH_exp_df = exp_var_df[exp_var_df['LISA_quadrant'] == 1]
        HH_exp_df = HH_exp_df.drop(columns=['Quadrant_name', 'LISA_quadrant'])
        HH_region_dict[exp_var] = HH_exp_df
        # LL regions
        LL_exp_df = exp_var_df[exp_var_df['LISA_quadrant'] == 3]
        LL_exp_df = LL_exp_df.drop(columns=['Quadrant_name', 'LISA_quadrant'])
        LL_region_dict[exp_var] = LL_exp_df

    # Returning all the results
    return global_moran_dict, local_moran_percent_dict, local_moran_dict, LH_region_dict, HL_region_dict, HH_region_dict \
        , LL_region_dict


def create_interactive_bar_graph(GDP_df, explanatory_vars_list, spatial_cluster_types_list, gdp_outlier_type):
    """
    Creates an interactive bar graph showing the percentage of regions explained by each explanatory variable
    for multiple spatial cluster types.

    Parameters:
    GDP_df (pd.DataFrame): DataFrame containing the regions of the dependent variable in 'NUTS_ID' column.
    explanatory_vars_list (list): List of dictionaries where each dictionary contains explanatory variables as keys
                                  and DataFrames containing the regions in 'NUTS_ID' column as values.
    spatial_cluster_types_list (list): List of descriptions for each spatial cluster type.
    gdp_cluster_type: The GDP outlier type that is explained by the variables.

    Returns:
    None: Displays the interactive bar graph.
    """
    # Ensure the input lists have the same length
    if len(explanatory_vars_list) != len(spatial_cluster_types_list):
        raise ValueError("The number of explanatory_vars dictionaries must match the number of spatial_cluster_types")

    # Initialize an empty list to store all the results
    all_results = []

    # Loop over each set of explanatory variables and corresponding cluster type
    for explanatory_vars, spatial_cluster_type in zip(explanatory_vars_list, spatial_cluster_types_list):
        # Initialize an empty list to store the results for the current cluster type
        current_results = []

        # Calculate the percentage of regions explained by each explanatory variable
        for var, df in explanatory_vars.items():
            common_regions = GDP_df['NUTS_ID'].isin(df['NUTS_ID']).sum()
            percentage_explained = round((common_regions / len(GDP_df)) * 100, 3)
            current_results.append({'Explanatory Variable': var, 'Percentage Explained': percentage_explained,
                                    'Cluster Type': spatial_cluster_type})

        # Add the current results to the all_results list
        all_results.extend(current_results)

    # Create a DataFrame from the results
    results_df = pd.DataFrame(all_results)

    # Filter out entries where the percentage explained is zero
    results_df = results_df[results_df['Percentage Explained'] > 0]

    # Create the interactive grouped bar graph
    fig = px.bar(results_df, x='Explanatory Variable', y='Percentage Explained', color='Cluster Type',
                 barmode='group', title='Percentage of '+gdp_outlier_type+' Regions Explained by Explanatory Variables',
                 labels={'Percentage Explained': 'Percentage of Regions', 'Cluster Type': 'Cluster Type'},
                 text='Percentage Explained')

    # Update the layout to ensure white background and black text
    fig.update_layout(
        xaxis_title='Explanatory Variable',
        yaxis_title='Percentage of Regions Explained',
        uniformtext_minsize=8, uniformtext_mode='hide',
        plot_bgcolor='white',  # Background color for the plot area
        paper_bgcolor='white',  # Background color for the entire figure
        font=dict(color='black')  # Font color for the text
    )

    # Show the interactive bar graph
    fig.show()


# High-low regions analysis
def create_categorical_heatmap_HL(GDP_df, explanatory_vars):
    """
    Creates a categorical heatmap showing which explanatory variables are able to explain which regions.

    Parameters:
    GDP_df (pd.DataFrame): DataFrame containing the regions of the dependent variable in 'NUTS_ID' column.
    explanatory_vars (dict): Dictionary where keys are explanatory variables and values are DataFrames
                             containing the regions in 'NUTS_ID' column.

    Returns:
    None: Displays the heatmap.
    """
    # Extract the list of regions
    regions = GDP_df['NUTS_ID'].tolist()

    # Initialize a dictionary to store the data for the heatmap
    matrix_data = {'Region': regions}
    for var in explanatory_vars.keys():
        matrix_data[var] = [0] * len(regions)

    # Populate the matrix data
    for var, df in explanatory_vars.items():
        explained_regions = df['NUTS_ID'].tolist()
        for i, region in enumerate(regions):
            if region in explained_regions:
                matrix_data[var][i] = 1

    # Create a DataFrame for the heatmap
    matrix_df = pd.DataFrame(matrix_data)
    matrix_df.set_index('Region', inplace=True)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns,
        y=matrix_df.index,
        colorscale=[[0, 'red'], [1, 'green']],
        showscale=False,
        xgap=2,  # Add gap between cells for x-axis
        ygap=2,  # Add gap between cells for y-axis
    ))

    # Add custom legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='red'),
        legendgroup='Not Explained',
        showlegend=True,
        name='Not Explained'
    ))

    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(size=10, color='green'),
        legendgroup='Explained',
        showlegend=True,
        name='Explained'
    ))

    # Update the layout to ensure white background and black text
    fig.update_layout(
        title={
            'text': 'Categorical Heatmap of High-Low Regions Explained by Explanatory Variables',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Explanatory Variable',
        yaxis_title='High-Low Region',
        yaxis=dict(categoryorder='total ascending'),
        plot_bgcolor='white',  # Background color for the plot area
        paper_bgcolor='white',  # Background color for the entire figure
        font=dict(color='black')  # Font color for the text
    )

    # Show the heatmap
    fig.show()


# Low-high region analysis
def country_code_analysis(regions_GDP, region_dict):
    """
    Creates a dataframe where Low-High regions are segregated into their country codes and then creates various plots
    to analyse how well explanatory variable can account for the GDP LL or HH regions.

    :param regions_GDP: dataframe containing GDP data for significant HH or LL regions.
    :param region_dict: dictionary containing dataframes for each explanatory variable with significant HH or LL regions.
    :return: new dataframe containing information about how much percentage of each LL/HH region is explained by each
    explanatory variable. Also returns the list of regions that cannot be explained by any variable.
    """

    # Function to extract the broader category from NUTS_ID
    def get_country_code(nuts_id):
        return nuts_id[:2]

    # Add a column for the broad category in GDP_df
    regions_GDP = regions_GDP.copy()
    regions_GDP['Broad_category'] = regions_GDP['NUTS_ID'].apply(get_country_code)

    # Initialize the final result dictionary
    exp_result = {}
    # Iterate through each unique broad category
    for var, df in region_dict.items():
        # Initializing empty result dictionary for each explanatory variable
        result = {}
        for category in regions_GDP['Broad_category'].unique():
            # Get all NUTS_IDs for the current category
            nuts_ids_lh = set(regions_GDP[regions_GDP['Broad_category'] == category]['NUTS_ID'])
            # Get the intersection of these NUTS_IDs with those in the explanatory_df
            common_nuts_ids = nuts_ids_lh.intersection(set(df['NUTS_ID']))
            # Calculate the percentage
            percentage = len(common_nuts_ids) / len(nuts_ids_lh) * 100
            # Store the result
            result[category] = round(percentage, 3)
        exp_result[var] = result

    # Indicating and removing all broad categories where NO explanatory variable is able to explain it!
    # Step 1: Identify all the broad categories that are not explained by ANY explanatory variable!
    unexplained_countries = set(exp_result[next(iter(exp_result))].keys())
    for outer_key in exp_result:
        unexplained_countries &= {k for k, v in exp_result[outer_key].items() if v == 0.0}
    # Step 2: Remove these keys from each inner dictionary
    for outer_key in exp_result:
        for key in unexplained_countries:
            exp_result[outer_key].pop(key, None)

    # Convert exp_result dictionary into a dataframe
    exp_df = pd.DataFrame(exp_result).transpose()

    # Return both the explanatory df and the unexplained regions list
    return exp_df, unexplained_countries


def create_heatmap(exp_df, cluster_type):
    """
    Creates a heatmap of how much percentage of each Low-High spatial cluster in GDP data is explained by each variable
    :param cluster_type: string variable expressing whether heat map is of low-low or high-high cluster analysis
    :param exp_df: dataframe with percentage information of explanatory power of each variable for each LH country code
    :return: None: Displays the heatmap.
    """
    df = exp_df.melt(id_vars='index', var_name='Explanatory Variable', value_name='Percentage')
    df.columns = ['Broad Category', 'Explanatory Variable', 'Percentage']

    # Create the interactive heatmap
    fig = px.density_heatmap(
        df,
        x='Broad Category',
        y='Explanatory Variable',
        z='Percentage',
        text_auto=True,
        color_continuous_scale='viridis'
    )

    # Update the layout to ensure white background and black text
    fig.update_layout(
        title={
            'text': 'Percentage of ' + cluster_type + ' Countries Explained by Explanatory Variables',
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Explanatory Variable',
        yaxis_title='Country Code',
        plot_bgcolor='white',  # Background color for the plot area
        paper_bgcolor='white',  # Background color for the entire figure
        font=dict(color='black')  # Font color for the text
    )

    # Show the heatmap
    fig.show()

