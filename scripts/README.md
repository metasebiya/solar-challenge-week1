import streamlit as st
import pandas as pd
import os

from app.utils import load_data, plot_ghi_boxplot, get_top_regions

st.set_page_config(page_title="Solar GHI Dashboard", layout="wide")

st.title("🌞 Global Horizontal Irradiance (GHI) Dashboard")

# Load data (adjust path if needed)
data_path = os.path.join("data", "ghi_data.csv")
df = load_data(data_path)

# Country selection
countries = df["Country"].unique().tolist()
selected_countries = st.multiselect("Select countries", countries, default=countries)

# Filtered data
filtered_df = df[df["Country"].isin(selected_countries)]

# Boxplot
st.subheader("GHI Distribution by Country")
st.pyplot(plot_ghi_boxplot(filtered_df))

# Top regions
st.subheader("Top Regions by GHI")
top_regions = get_top_regions(filtered_df, top_n=10)
st.dataframe(top_regions)
