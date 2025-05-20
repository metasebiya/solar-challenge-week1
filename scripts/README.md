# Streamlit Section: Solar Data Analysis Dashboard

## Overview
The **Solar Data Analysis Dashboard** is a Streamlit-based web application within the **Solar Farm Investment Analysis – Moonlight Solar Solutions** project. This dashboard enables interactive visualization and analysis of Global Horizontal Irradiance (GHI) data for Benin, Sierra Leone, and Togo, supporting the project’s goal of identifying high-potential regions for solar farm investment through data-driven insights.

## Role in the Parent Project
As part of the **Solar Farm Investment Analysis – Moonlight Solar Solutions** project, this Streamlit section complements the exploratory data analysis (EDA) in the `notebooks/` directory by providing an interactive interface for stakeholders. It uses cleaned datasets from the `data/` directory and utility functions from the `scripts/` directory to deliver dynamic GHI visualizations and summary statistics, facilitating strategic decision-making.

## Features
- **Country Selection**: Choose one or more countries (Benin, Sierra Leone, Togo) via a sidebar to analyze GHI data.
- **GHI Boxplot**: Visualize GHI distribution (in W/m²) across selected countries using an interactive Plotly boxplot.
- **Summary Statistics**: Display a table of top regions by GHI for the selected countries.
- **Performance Optimization**: Implements Streamlit’s `@st.cache_data` (1-hour cache) and lazy loading via a "Load Data" button to minimize redundant computations.
- **Responsive Design**: Utilizes Streamlit’s wide layout for an enhanced user experience.

## Prerequisites
- Python 3.8 or higher
- Dependencies listed in the parent project’s `requirements.txt`:
  - `streamlit==1.38.0`
  - `pandas==2.2.2`
  - `plotly==5.22.0`
- The `utils.py` script in `../scripts/`, containing `fetch_cleaned_data`, `process_data`, and `get_summary_stats` functions
- Cleaned datasets for Benin, Sierra Leone, and Togo in the `../data/` directory

## Installation
1. **Clone the Parent Repository** (if not already done):
   ```bash
   git clone https://github.com/metasebiya/solar-challenge-week1.git
   cd solar-challenge-week1
