import streamlit as st
import folium
import json

# load GeoJSON data for Indonesia
with open("indonesia-prov.geojson") as f:
    indo_geojson = json.load(f)

# set initial view to Indonesia
m = folium.Map(location=[-2.5, 118], zoom_start=5)

# add GeoJSON data for each province
for feature in indo_geojson['features']:
    name = feature['properties']['name']
    geometry = feature['FeatureCollection']
    folium.GeoJson(
        data=geometry,
        name=name,
        tooltip=name
    ).add_to(m)

# display map using Streamlit
st.write(folium_static(m))
