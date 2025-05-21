# Streamlit Section: Solar Data Analysis Dashboard

## Overview
The **Solar Data Analysis Dashboard** is a Streamlit-based web application within the **Solar Farm Investment Analysis – Moonlight Solar Solutions** project. This dashboard enables interactive visualization and analysis of Global Horizontal Irradiance (GHI) data for Benin, Sierra Leone, and Togo, supporting the project’s goal of identifying high-potential regions for solar farm investment through data-driven insights.

## Role in the Parent Project
As part of the **Solar Farm Investment Analysis – Moonlight Solar Solutions** project, this Streamlit section complements the exploratory data analysis (EDA) in the `notebooks/` directory by providing an interactive interface for stakeholders. It uses cleaned datasets from the `data/` directory and utility functions from the `app/` directory to deliver dynamic GHI visualizations and summary statistics, facilitating strategic decision-making.

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
- The `utils.py` script in `../app/`, containing `fetch_cleaned_data`, `process_data`, and `get_summary_stats` functions
- Cleaned datasets for Benin, Sierra Leone, and Togo in the `../data/` directory

## Installation
1. **Clone the Parent Repository** (if not already done):
   ```bash
   git clone https://github.com/metasebiya/solar-challenge-week1.git
   cd solar-challenge-week1
2. **Setup virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install Dependencies**:
   ```bash
   streamlit==1.38.0
   pandas==2.2.2
   plotly==5.22.0
   numpy==1.26.4
   matplotlib==3.8.4
   seaborn==0.13.2
   scipy==1.13.0
   ```
   ***Install them from the project root***
   pip install -r requirements.txt
## Usage
1. **Navigate to the Streamlit Directory**
   ```bash
   cd app/
   ```
2. **Run the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```
3. **Interact with the Dashboard:**
   - Select one or more countries (Benin, Sierra Leone, Togo) in the sidebar.
   - Click the "Load Data" button to fetch and process data.
   - Explore the GHI boxplot and summary statistics table.
   - A warning message displays if no valid data is available for the selected countries.
## Notes
- Data Source: The dashboard relies on cleaned datasets in ../data/ and processing functions in ../scripts/utils.py.
- Caching: Data is cached for 1 hour using @st.cache_data to optimize performance.
- Lazy Loading: Data processing is triggered only when the "Load Data" button is clicked, reducing unnecessary computations.
- Error Handling: A warning is shown if no valid data is available for the selected countries.
## 👤 Author

- **Name**: Metasebiya Akililu
- **GitHub**: [@metasebiya](https://github.com/metasebiya)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions or feedback, please reach out via [GitHub Issues](https://github.com/metasebiya/solar-challenge-week1/issues).
  
   
