# Imports
import folium


def NUTS_map(gdf_lvl, variable):
    # Center coordinates for the map
    eu_center = [50.8503, 4.3517]

    # Create a folium map centered on the EU center
    m = folium.Map(location=eu_center, zoom_start=4)

    # Add choropleth layer with GDP values
    folium.Choropleth(
        geo_data=gdf_lvl,
        data=gdf_lvl,
        columns=['NUTS_ID', variable],
        key_on='feature.properties.NUTS_ID',
        fill_color='OrRd',  # Color scale
        fill_opacity=0.7,
        line_opacity=0.1,
        legend_name=variable,
        highlight=True,
        name='GDP Choropleth',
    ).add_to(m)

    # Add GeoJSON layer with tooltip
    folium.GeoJson(
        gdf_lvl,
        name='geojson',
        tooltip=folium.features.GeoJsonTooltip(fields=['NUTS_ID', variable], labels=True, sticky=False)
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)
    m




