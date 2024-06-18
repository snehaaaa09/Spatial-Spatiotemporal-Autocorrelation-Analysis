# Spatial-Spatiotemporal-Autocorrelation-Analysis
## About the project
This research explores Spatial and Spatiotemporal Autocorrelation analysis of regional GDP data across the EU at NUTS (Nomenclature of Territorial Units for Statistics) levels 2 and 3. Detecting the presence of spatial autocorrelation is a vital part of spatial data analysis. A popular measure, Moran's I (Moran, 1950) is implemented to measure the spatial autocorrelation. An extension of Moran's I, which incorporates temporal variations to measure spatiotemporal autocorrelation is also implemented (Goa et al., 2019). Spatial modeling with spatially lagged Y models and spatially lagged error models are also implemented to test the relationship between some explanatory variables and regional GDP across the EU. 

## Data Downloading
 **Note**: Before running any files, it is important to download all the required data and upload them in a folder called 'data' inside the 'Functions' directory(folder). It is important to name the sub-folders as mentioned here to ensure the code runs smoothly. All the data (regional GDP and explanatory data) below is obtained from the Eurostat data browser: https://ec.europa.eu/eurostat. 

#### Regional GDP 
1) Regional GDP data (at NUTS 2): The data entry is called 'Gross domestic product (GDP) at current market prices by NUTS 2 regions (nama_10r_2gdp)' on the data browser. Download the 'csv' version and upload it to a subfolder named 'GDP(Nuts2)' under the 'data' directory (data/GDP(Nuts2)/...)
2) Regional GDP data (at NUTS 3): The data entry is called 'Gross domestic product (GDP) at current market prices by NUTS 3 regions (nama_10r_3gdp)' on the data browser. Download the 'csv' version and upload it to a subfolder named 'GDP(Nuts3)' under the 'data' directory (data/GDP(Nuts2)/...)
3) Shapefile data for the NUTS regions: Download the shapefile data for NUTS (levels 2 and 3 are included in the single file) from the following source: https://ec.europa.eu/eurostat/web/gisco/geodata/statistical-units/territorial-units-statistics. Set the file format to 'SHP', the geometry type to 'Polygon(RG)' and the coordinate reference system to 'ESPG: 3035' before downloading it. Store this folder under the 'data' folder. 

#### Explanatory variables
Create a sub-folder called 'X_data(Nuts2)' under the 'data' folder before downloading and uploading the following data. (All of the explanatory data downloaded is at the NUTS 2 level)
1) Disposible_income(Nuts2): The data entry is called 'Disposable income of private households by NUTS 2 regions' on the data browser. Download the full dataset as the 'csv' version and upload it to the 'X_data(Nuts2)' subfolder under the folder named 'Disposible_income(Nuts2)' (data/X_data(Nuts2)/Disposible_income(Nuts2)/...)
2) Primary_income(Nuts2):  The data entry is called 'Primary income of private households by NUTS 2 regions' on the data browser. Download the full dataset as the 'csv' version and upload it to the 'X_data(Nuts2)' subfolder under the folder named 'Primary_income(Nuts2)' (data/X_data(Nuts2)/Primary_income(Nuts2)/...)

For the following explanatory variables, follow the same pattern of downloading and creating a sub-folder with the name of the variable: 
4) Population_density(Nuts2)
5) Population_jan1(Nuts2)
6) Total_death(Nuts2)
7) Tourism_places(Nuts2)
8) Nights_spent_in_tourism_place(Nuts2)
9) Science_Tech(Nuts2)
10) Teriary_edu(Nuts2)

## Jyputer Notebooks Explanation
All the files below (except one) are important to visualize the results of the research.

### 1) SAC_NUTS_2.ipynb
Here, the spatial autocorrelation of regional GDP is measured using Moran's I at the NUTS 2 level. Spatial autocorrelation is also done for all the explanatory variables to build heat maps of the spatial clusters and outliers that are similar between regional GDP and each explanatory variable. 

### 3) SAC_NUTS_3.ipynb 
Here, the spatial autocorrelation of regional GDP is measured using Moran's I at the NUTS 3 level. Spatial autocorrelation is also done for all the explanatory variables to build heat maps of the spatial clusters and outliers that are similar between regional GDP and each explanatory variable. Also, comparisons are made between the spatial autocorrelation results at the NUTS 2 and 3 levels here. 

### 4) STAC_NUTS_2.ipynb
Here, the spatiotemporal autocorrelation of regional GDP is conducted using the extended version of Moran's I at the NUTS 2 level. Spatiotemporal autocorrelation is also done for all the explanatory variables to build heat maps of the spatial clusters and outliers that are similar between regional GDP and each explanatory variable. Also, comparisons are made between the spatial autocorrelation and spatiotemporal autocorrelation results here. 

### 5) STAC_Significance_testing.ipynb (optional)
**Note:** This notebook is optional and need not be run. The results of the statistical significance tests are already stored in the 'results' directory. It takes around **12 hours** to run the local significance testing and **25 minutes** to run the global significance testing due to the large number of permutations (999). 
Here, the simulated values of the spatiotemporal autocorrelation measure are created and statistical significance testing is conducted at both the local and the global levels.  

### 6) Spatial_modeling.ipynb
Here, multiple spatially lagged models are run to find the one that performs the best. The results of the final model are displayed. 

## Python File Explanation
All the files and their respective functions within are imported into the Jyputer notebooks. Therefore, these files need **not** be run separately. View this file if you wish to see how the spatial/spatiotemporal autocorrelation results are obtained and how the visualizations are created.

### 1) data_prep.py
This file contains all the functions required for downloading, cleaning, and preparing the regional GDP data or explanatory variables. **Note:** It is very important that all the data files mentioned above are downloaded and present in the correct directory for the functions in this file to run smoothly. 

### 2) spatial_functions.py 
This file contains all the functions required for creating the spatial weight matrix, calculating Moran's I, calculating the quadrant of each LISA value, and displaying the LISA cluster maps of both spatial and spatiotemporal autocorrelation. 

### 3) Spatial_outlier_analysis.py
This file contains all the functions required for calculating the similar spatial clusters and outliers between regional GDP and the explanatory variables for both spatial and spatiotemporal autocorrelation. It also contains functions required for creating and displaying the various heat maps of the explanatory power of the variables. 

### 4) spatiotemporal_autocorrelation.py
This file contains all the functions required for implementing the extended version of Moran's I (since no Python implementation is readily available, all the components of the formula are coded as separate functions). 

## References
Gao, Y., Cheng, J., Meng, H., & Liu, Y. (2019). Measuring spatio-temporal autocorrelation in time series data of collective human mobility. Geo-Spatial Information Science, 22(3), 166–173. https://doi.org/10.1080/10095020.2019.1643609. 

Moran, P. (1950) A Test for the Serial Independence of Residuals. Biometrika, 37, 178-181. http://dx.doi.org/10.1093/biomet/37.1-2.178
