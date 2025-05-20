import streamlit as st
import pandas as pd
import plotly.express as px
from utils import fetch_cleaned_data, process_data, get_summary_stats

# Set page config
st.set_page_config(page_title="Solar Data Dashboard", layout="wide")

# Define available countries
COUNTRIES = ["benin", "sierraleone", "togo"]

# Cache the data fetching function to avoid redundant I/O
@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_fetch_cleaned_data(country):
    return fetch_cleaned_data(country)

# Cache the data processing function
@st.cache_data(ttl=3600)
def cached_process_data(selected_countries):
    return process_data(selected_countries)

# Cache the summary stats function
@st.cache_data(ttl=3600)
def cached_get_summary_stats(dfs, valid_countries):
    return get_summary_stats(dfs, valid_countries)

# Title
st.title("Solar Data Analysis Dashboard")

# Sidebar for country selection
st.sidebar.header("Filter Options")
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    options=[c.capitalize() for c in COUNTRIES],
    default=[c.capitalize() for c in COUNTRIES],
    key="country_selector"
)

# Initialize session state for lazy loading
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.dfs = []
    st.session_state.valid_countries = []

# Load data only when the "Load Data" button is clicked
if st.sidebar.button("Load Data"):
    with st.spinner("Loading and processing data..."):
        # Fetch and process data using cached functions
        st.session_state.dfs, st.session_state.valid_countries = cached_process_data(selected_countries)
        st.session_state.data_loaded = True

# Display content only if data is loaded and valid
if st.session_state.data_loaded and st.session_state.valid_countries:
    # GHI Boxplot
    st.subheader("GHI Distribution by Country")
    boxplot_data = pd.concat(
        [df[["GHI"]].assign(Country=country.capitalize()) for df, country in zip(st.session_state.dfs, st.session_state.valid_countries)],
        axis=0
    )
    fig = px.box(boxplot_data, x="Country", y="GHI", title="GHI Distribution (W/m²)")
    fig.update_layout(xaxis_title="Country", yaxis_title="GHI (W/m²)")
    st.plotly_chart(fig, use_container_width=True)

    # Summary Table
    st.subheader("Top Regions by GHI")
    summary_table = cached_get_summary_stats(st.session_state.dfs, st.session_state.valid_countries)
    st.table(summary_table)
elif st.session_state.data_loaded and not st.session_state.valid_countries:
    st.warning("No valid data available for selected countries. Please check data files or select different countries.")
else:
    st.info("Please select countries and click 'Load Data' to view the dashboard.")