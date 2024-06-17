# Imports
import pandas as pd
import matplotlib.pyplot as plt
from pysal.lib import weights
from pysal.explore import esda
import seaborn
from splot import esda as esdaplot
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# User-defined functions
from Functions.spatiotemporal_autocorrelation import region_df_creation, corr_time_series, adjusted_time_series


# Spatial Weight Matrix calculation
def spatial_weight_matrix(gdf_lvl, k_val):
    # Building kernel weights
    # Distance weights: Build weights with adaptive bandwidth
    w_adaptive = weights.distance.Kernel.from_dataframe(gdf_lvl, fixed=False, k=k_val)
    # Row-standardization
    w_adaptive.transform = "R"

    return w_adaptive


def neighborhood_dict_creation(w_adaptive):
    """
    :param w_adaptive: weights object from esda package that has neighborhood information already
    :return: dictionary of neighborhood IDs of each region
    """
    neighbor_weights_dict = {}
    for region_id, neighbors in w_adaptive.neighbors.items():
        weights_for_region = [w_adaptive[region_id][neighbor_id] for neighbor_id in neighbors]
        neighbor_weights_dict[region_id] = {neighbor_id: weight for neighbor_id, weight in
                                            zip(neighbors, weights_for_region)}
    return neighbor_weights_dict


def global_moran_val(gdf_level, variable, w_adaptive):
    # Global SAC
    moran = esda.moran.Moran(gdf_level[variable], w_adaptive)
    return moran


def local_moran_val(gdf_level, variable, w_adaptive):
    # Local SAC
    # Creating the local moran object
    lisa = esda.moran.Moran_Local(gdf_level[variable], w_adaptive)
    return lisa


def lisa_df_update(gdf_lvl, lisa):
    # Copy of dataframe for Splicing (to avoid Splice ambiguity error)
    gdf_lvl_copy = gdf_lvl.copy()
    # Find out significant observations
    labels = pd.Series(
        1 * (lisa.p_sim < 0.05),  # Assign 1 if significant, 0 otherwise
        index=gdf_lvl.index  # Use the index in the original data
        # Recode 1 to "Significant and 0 to "Non-significant"
    ).map({1: "Significant", 0: "Non-Significant"})

    # Adding lisa and significance column to df
    numpy_df = pd.DataFrame(lisa.Is, columns=['LISA_VALUE'])
    # gdf_lvl['LISA_VALUE'] = numpy_df['LISA_VALUE']
    gdf_lvl_copy.loc[:, 'LISA_VALUE'] = numpy_df['LISA_VALUE'].values

    # Adding significance values
    series_df = labels.to_frame(name='LISA_sig')
    gdf_lvl_copy['LISA_sig'] = series_df['LISA_sig']

    # Get p-values from Local Moran's I
    p_values = lisa.p_sim
    # Create a DataFrame with the extracted p-values
    p_values_df = pd.DataFrame(p_values, columns=['LISA_p_values'])
    gdf_lvl_copy['LISA_p_values'] = p_values_df['LISA_p_values']

    # Get quadrant information from Local Moran's I
    qd_df = pd.DataFrame(lisa.q, columns=['LISA_quadrant'])
    gdf_lvl_copy['LISA_quadrant'] = qd_df['LISA_quadrant']

    return gdf_lvl_copy


def lisa_update_STAC(region_df, final_df, local_STAC_dict, w_adaptive, variable):
    """ Function to update the STAC dataframe so each region now has updated LISA information such as the LISA value and
    quadrant information (HL, HH, LH or LL quadrant)
    :param w_adaptive: spatial weight matrix that quantifies spatial proximity between regions in region_df
    :param local_STAC_dict: dictionary with already calculated LISA values for each region
    :param region_df: dataframe of region IDs and geo information which is to updated with LISA info and returned
    :param variable: string of variable of interest
    :param final_df: dataframe with entire data for each region across all time periods
    :return: result_df: dataframe of the region_df updated with LISA info and also quadrant_df: dataframe with quadrant
    information
    """
    # Initializing dataframes to store results in
    df_region_series = pd.DataFrame(columns=['NUTS_ID', 'Region_value'])
    df_mean_series = pd.DataFrame(columns=['NUTS_ID', 'Mean_value'])
    quadrant_df = pd.DataFrame()
    # Converting LISA dictionary into a df for easier modification of region_df
    LISA_df = pd.DataFrame(list(local_STAC_dict.items()), columns=['NUTS_ID', 'LISA_VALUE'])
    # Calculate mean GDP over all regions for each year
    # Group by TIME_PERIOD and calculate the mean GDP value summed over all regions for each year
    mean_gdp_sum_per_year = final_df.groupby('TIME_PERIOD')[variable].mean()
    # Convert the resulting series to a DataFrame
    mean_gdp_df = mean_gdp_sum_per_year.to_frame(name='Mean_variable')
    # ----- Quadrant calculation (LL, HH, HL and LH) ------#
    # Getting list of regions
    regions = list(LISA_df['NUTS_ID'])
    for i, region_i in enumerate(regions):
        # Iterate over all neighbor weights for the current region
        time_series_i = region_df_creation(final_df, region_i)
        corr_i = corr_time_series(time_series_i[variable], mean_gdp_df['Mean_variable'])
        region_i_val = adjusted_time_series(corr_i, time_series_i[variable])
        mean_i_val = adjusted_time_series(corr_i, mean_gdp_df['Mean_variable'])
        # Storing the results in dataframes
        df_region_series.loc[i] = [region_i, region_i_val]
        df_mean_series.loc[i] = [region_i, mean_i_val]
    # Quadrant calculation
    quadrant_df["NUTS_ID"] = df_region_series["NUTS_ID"]
    quadrant_df[variable] = df_region_series["Region_value"]
    quadrant_df[variable + "_mean"] = df_mean_series["Mean_value"]
    # Creating spatially lagged variable
    quadrant_df["w_" + variable] = weights.lag_spatial(w_adaptive, quadrant_df[variable])

    # Creating respective centered versions where mean is subtracted from each
    quadrant_df[variable + "_std"] = quadrant_df[variable] - quadrant_df[variable + "_mean"]
    quadrant_df["w_" + variable + "_std"] = weights.lag_spatial(w_adaptive, quadrant_df[variable + "_std"])

    # Calculating the quadrant information and adding new columns
    for index, row in quadrant_df.iterrows():
        if row[variable + "_std"] > 0.0 and row["w_" + variable + "_std"] > 0.0:
            quadrant_df.at[index, 'Quadrant_name'] = 'HH'
            quadrant_df.at[index, 'LISA_quadrant'] = 1
        elif row[variable + "_std"] < 0.0 and row["w_" + variable + "_std"] < 0.0:
            quadrant_df.at[index, 'Quadrant_name'] = 'LL'
            quadrant_df.at[index, 'LISA_quadrant'] = 3
        elif row[variable + "_std"] < 0.0 and row["w_" + variable + "_std"] > 0.0:
            quadrant_df.at[index, 'Quadrant_name'] = 'LH'
            quadrant_df.at[index, 'LISA_quadrant'] = 2
        else:
            quadrant_df.at[index, 'Quadrant_name'] = 'HL'
            quadrant_df.at[index, 'LISA_quadrant'] = 4

    # Updating the region_df with LISA_df information
    # Merging lisa_df with region_df
    merged_df1 = pd.merge(region_df, LISA_df[['NUTS_ID', 'LISA_VALUE']], on='NUTS_ID', how='left')
    # Merging quadrant_df with the result of the first merge
    result_df = pd.merge(merged_df1, quadrant_df[['NUTS_ID', 'LISA_quadrant', 'Quadrant_name']], on='NUTS_ID',
                         how='left')

    return result_df, quadrant_df


def quadrant_plot_STAC(quadrant_df, variable):
    """
    :param variable: variable of interest
    :param quadrant_df: dataframe with spatially lagged standard deviation and standard deviation of variable.
    :return: None, instead plot the scatter plot of spatially lagged standard deviation versus standard deviation to
    visualize how the quadrants are split.
    """
    # Define the color mapping for each quadrant
    color_mapping = {'HH': 'red', 'LH': 'lightblue', 'LL': 'darkblue', 'HL': 'orange'}
    # Set up the figure and axis
    plt.figure(figsize=(9, 9))
    # Plot values - scatter plot
    sns.scatterplot(
        x=variable + "_std",
        y="w_" + variable + "_std",
        data=quadrant_df,
        hue='Quadrant_name',  # Use 'Quadrant_name' for coloring
        palette=color_mapping,  # Apply the custom color mapping
        legend=True  # Disable the legend for simplicity
    )
    # Add the regression line using sns.regplot without the confidence interval
    sns.regplot(
        x=variable + "_std",
        y="w_" + variable + "_std",
        data=quadrant_df,
        scatter=False,  # Disable scatter plot from regplot
        color='black',  # Regression line color
        ci=None  # Remove the error bars
    )
    # Add vertical and horizontal lines
    plt.axvline(0, c='k', alpha=0.5)
    plt.axhline(0, c='k', alpha=0.5)
    # Set axis labels
    plt.xlabel(f'{variable}_std')
    plt.ylabel(f'w_{variable}_std')
    # Add a title
    plt.title(f'Relationship between {variable}_std and w_{variable}_std by Quadrant')

    # Display the plot
    plt.show()


# Plots for local moran values
def local_moran_density_plot(lisa):
    # Visualizing the density diagram of the local moran values (to compare between positive and negative SAC)
    # Draw KDE line
    ax = seaborn.kdeplot(lisa.Is)
    # Add one small bar (rug) for each observation
    # along horizontal axis
    seaborn.rugplot(lisa.Is, ax=ax)
    # Add x-axis label
    ax.set_xlabel('LISA values')
    # Add title
    ax.set_title('Density Diagram of LISA Values')


def local_moran_plots(gdf_lvl, lisa):
    # Set up figure and axes
    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    # Make the axes accessible with single indexing
    axs = axs.flatten()

    # Subplot 1 #
    # Choropleth of local statistics
    # Grab first axis in the figure
    ax = axs[0]
    # Assign new column with local statistics on-the-fly
    gdf_lvl.assign(
        Is=lisa.Is
        # Plot choropleth of local statistics
    ).plot(
        column="Is",
        cmap="plasma",
        scheme="quantiles",
        k=5,
        edgecolor="white",
        linewidth=0.1,
        alpha=0.75,
        legend=True,
        ax=ax,
    )

    # Subplot 2 #
    # Quadrant categories
    # Grab second axis of local statistics
    ax = axs[1]
    # Plot Quadrant colors (note to ensure all polygons are assigned a
    # quadrant, we "trick" the function by setting significance level to
    # 1 so all observations are treated as "significant" and thus assigned
    # a quadrant color
    esdaplot.lisa_cluster(lisa, gdf_lvl, p=1, ax=ax)

    # Subplot 3 #
    # Significance map
    # Grab third axis of local statistics
    ax = axs[2]
    #
    # Find out significant observations
    labels = pd.Series(
        1 * (lisa.p_sim < 0.05),  # Assign 1 if significant, 0 otherwise
        index=gdf_lvl.index  # Use the index in the original data
        # Recode 1 to "Significant and 0 to "Non-significant"
    ).map({1: "Significant", 0: "Non-Significant"})
    # Assign labels to `db` on the fly
    gdf_lvl.assign(
        cl=labels
        # Plot choropleth of (non-)significant areas
    ).plot(
        column="cl",
        categorical=True,
        k=2,
        cmap="Paired",
        linewidth=0.1,
        edgecolor="white",
        legend=True,
        ax=ax,
    )

    # Subplot 4 #
    # Cluster map
    # Grab second axis of local statistics
    ax = axs[3]
    # Plot Quadrant colors In this case, we use a 5% significance
    # level to select polygons as part of statistically significant
    # clusters
    esdaplot.lisa_cluster(lisa, gdf_lvl, p=0.05, ax=ax)

    # Figure styling #
    # Set title to each subplot
    for i, ax in enumerate(axs.flatten()):
        ax.set_axis_off()
        ax.set_title(
            [
                "Local Statistics",
                "Scatterplot Quadrant",
                "Statistical Significance",
                "Moran Cluster Map",
            ][i],
            y=0,
        )
    # Tight layout to minimize in-between white space
    f.tight_layout()

    # Display the figure
    plt.show()


def local_moran_plots_STAC(gdf_lvl, lisa):
    # Set up figure and axes
    f, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    # Make the axes accessible with single indexing
    axs = axs.flatten()

    # Subplot 1 #
    # Choropleth of local statistics
    ax = axs[0]
    gdf_lvl.assign(
        Is=lisa.Is
    ).plot(
        column="Is",
        cmap="plasma",
        scheme="quantiles",
        k=5,
        edgecolor="white",
        linewidth=0.1,
        alpha=0.75,
        legend=True,
        ax=ax,
    )
    ax.set_title("Local Statistics", y=0)

    # Subplot 2 #
    # Quadrant categories
    ax = axs[1]
    color_mapping = {
        'HH': 0,  # red
        'HL': 1,  # orange
        'LH': 2,  # blue
        'LL': 3  # light blue
    }
    gdf_lvl['color'] = gdf_lvl['Quadrant_name'].map(color_mapping)
    cmap = mcolors.ListedColormap(['#e41a1c', '#ff7f00', '#a6cee3', '#377eb8'])
    gdf_lvl.plot(
        ax=ax,
        column='color',
        categorical=True,
        k=4,
        cmap=cmap,
        edgecolor="white",
        linewidth=0.1,
        alpha=0.75,
        legend=False,
    )
    ax.set_axis_off()
    legend_handles = [Line2D([0], [0], marker='o', color='w', label=label, markersize=8, markerfacecolor=cmap(i))
                      for label, i in color_mapping.items()]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=8, frameon=True)
    ax.set_title("Scatterplot Quadrant", y=0)

    # Subplot 3 #
    # Significant vs Non-significant regions
    ax = axs[2]
    sig_color_mapping = {
        'Significant': 0,  # significant (brown)
        'Non-Significant': 1  # non-significant (light blue)
    }
    gdf_lvl['sig_color'] = gdf_lvl['LISA_sig'].map(sig_color_mapping)
    cmap_sig = mcolors.ListedColormap(['#d95f02', '#a6cee3'])
    gdf_lvl.plot(
        ax=ax,
        column='sig_color',
        categorical=True,
        k=2,
        cmap=cmap_sig,
        edgecolor="white",
        linewidth=0.1,
        alpha=0.75,
        legend=False,
    )
    ax.set_axis_off()
    sig_legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='Significant', markersize=8, markerfacecolor='#d95f02'),
        Line2D([0], [0], marker='o', color='w', label='Non-Significant', markersize=8, markerfacecolor='#a6cee3')
    ]
    ax.legend(handles=sig_legend_handles, loc='upper right', fontsize=8, frameon=True)
    ax.set_title("Statistical Significance", y=0)

    # Subplot 4 #
    # Significant Quadrant categories
    ax = axs[3]
    sig_color_mapping_with_grey = {
        'HH': 0,  # red
        'HL': 1,  # orange
        'LH': 2,  # blue
        'LL': 3,  # light blue
        'NS': 4  # grey for non-significant
    }
    gdf_lvl['sig_quad_color'] = gdf_lvl.apply(
        lambda row: sig_color_mapping_with_grey[row['Quadrant_name']] if row['LISA_sig'] == 'Significant' else 4,
        axis=1)
    cmap_quad = mcolors.ListedColormap(['#e41a1c', '#ff7f00', '#a6cee3', '#377eb8', '#7f7f7f'])
    gdf_lvl.plot(
        ax=ax,
        column='sig_quad_color',
        categorical=True,
        k=5,
        cmap=cmap_quad,
        edgecolor="white",
        linewidth=0.1,
        alpha=0.75,
        legend=False,
    )
    ax.set_axis_off()
    sig_quad_legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='HH', markersize=8, markerfacecolor=cmap_quad(0)),
        Line2D([0], [0], marker='o', color='w', label='HL', markersize=8, markerfacecolor=cmap_quad(1)),
        Line2D([0], [0], marker='o', color='w', label='LH', markersize=8, markerfacecolor=cmap_quad(2)),
        Line2D([0], [0], marker='o', color='w', label='LL', markersize=8, markerfacecolor=cmap_quad(3)),
        Line2D([0], [0], marker='o', color='w', label='NS', markersize=8, markerfacecolor=cmap_quad(4))
    ]
    ax.legend(handles=sig_quad_legend_handles, loc='upper right', fontsize=8, frameon=True)
    ax.set_title("Moran Cluster Map", y=0)

    # Figure styling #
    # Set title to each subplot
    for i, ax in enumerate(axs.flatten()):
        ax.set_axis_off()
        ax.set_title(
            [
                "Local Statistics",
                "Scatterplot Quadrant",
                "Statistical Significance",
                "Moran Cluster map"
            ][i],
            y=0,
        )
    # Tight layout to minimize in-between white space
    f.tight_layout()
    # Display the figure
    plt.show()


def lisa_cluster_map(lisa, gdf_lvl):
    # Plot Quadrant colors with interactivity
    fig, ax = plt.subplots(figsize=(20, 20))
    # Plot Quadrant colors In this case, we use a 5% significance
    # level to select polygons as part of statistically significant clusters
    esdaplot.lisa_cluster(lisa, gdf_lvl, p=0.05, ax=ax)
    ax.set_title("Moran Cluster Map")
    # Display the figure
    plt.show()


def lisa_cluster_map_STAC(gdf):
    # Define the color mapping for the LISA quadrants
    color_mapping = {
        'HH': '#e41a1c',  # red
        'HL': '#ff7f00',  # orange
        'LH': '#a6cee3',  # blue
        'LL': '#377eb8'  # light blue
    }
    # Map the LISA quadrants to their corresponding colors
    gdf['color'] = gdf['Quadrant_name'].map(color_mapping)
    # Create the plot with specified background color and border settings
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # Plot the geo dataframe
    gdf.plot(
        ax=ax,
        color=gdf['color'],
        edgecolor="white",  # White borders
        linewidth=0.1,  # Thin lines for borders
        alpha=0.75,  # Transparency level
    )
    # Remove axis for a cleaner look
    ax.set_axis_off()
    # Create custom legend with better placement and font color
    legend_labels = {
        'HH': 'HH',
        'HL': 'HL',
        'LH': 'LH',
        'LL': 'LL'
    }
    # Create legend handles with small circular markers
    # Create custom legend handles with larger circular markers
    legend_handles = [Line2D([0], [0], marker='o', color='w', label=label, markersize=8, markerfacecolor=color)
                      for label, color in color_mapping.items()]
    # Add the custom legend to the plot
    plt.legend(handles=legend_handles, loc='upper right', fontsize=8, frameon=True)
    # Set the title with white color for better visibility on dark background
    plt.title("LISA Quadrants Map", fontsize=9, color='black')
    # Setting title at bottom:
    # plt.figtext(0.5, 0.02, "LISA Quadrants Map", ha="center", fontsize=9, color='black')
    plt.show()


def lisa_cluster_map_STAC1(gdf_lvl):
    # Create the plot with specified background color and border settings
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sig_color_mapping_with_grey = {
        'HH': 0,  # red
        'HL': 1,  # orange
        'LH': 2,  # blue
        'LL': 3,  # light blue
        'NS': 4  # grey for non-significant
    }
    gdf_lvl['sig_quad_color'] = gdf_lvl.apply(
        lambda row: sig_color_mapping_with_grey[row['Quadrant_name']] if row['LISA_sig'] == 'Significant' else 4,
        axis=1)
    cmap_quad = mcolors.ListedColormap(['#e41a1c', '#ff7f00', '#a6cee3', '#377eb8', '#7f7f7f'])
    gdf_lvl.plot(
        ax=ax,
        column='sig_quad_color',
        categorical=True,
        k=5,
        cmap=cmap_quad,
        edgecolor="white",
        linewidth=0.1,
        alpha=0.75,
        legend=False,
    )
    ax.set_axis_off()
    sig_quad_legend_handles = [
        Line2D([0], [0], marker='o', color='w', label='HH', markersize=8, markerfacecolor=cmap_quad(0)),
        Line2D([0], [0], marker='o', color='w', label='HL', markersize=8, markerfacecolor=cmap_quad(1)),
        Line2D([0], [0], marker='o', color='w', label='LH', markersize=8, markerfacecolor=cmap_quad(2)),
        Line2D([0], [0], marker='o', color='w', label='LL', markersize=8, markerfacecolor=cmap_quad(3)),
        Line2D([0], [0], marker='o', color='w', label='NS', markersize=8, markerfacecolor=cmap_quad(4))
    ]
    ax.legend(handles=sig_quad_legend_handles, loc='upper right', fontsize=8, frameon=True)
    ax.set_title("Moran Cluster Map", y=0)

    plt.show()
