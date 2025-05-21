import pandas as pd
from pathlib import Path

# Define data path
DATA_PATH = Path("../data")

def fetch_cleaned_data(country: str) -> pd.DataFrame:
    """
    Fetch cleaned CSV data for a given country.
    
    Args:
        country (str): Name of the country (e.g., 'benin', 'sierraleone', 'togo').
    
    Returns:
        pd.DataFrame: Cleaned DataFrame if file exists, None otherwise.
    """
    file_path = DATA_PATH / f"{country.lower()}_clean.csv"
    try:
        if file_path.exists():
            return pd.read_csv(file_path)
        else:
            return None
    except Exception as e:
        print(f"Error loading data for {country}: {e}")
        return None

def process_data(countries: list) -> tuple[list[pd.DataFrame], list[str]]:
    """
    Dynamically fetch and process data for a list of countries.
    
    Args:
        countries (list): List of country names (e.g., ['Benin', 'SierraLeone', 'Togo']).
    
    Returns:
        tuple: List of DataFrames and list of valid country names.
    """
    dfs = []
    valid_countries = []
    
    for country in countries:
        df = fetch_cleaned_data(country.lower())
        if df is not None:
            # Example processing: Ensure 'GHI' column exists and drop rows with missing GHI
            if 'GHI' in df.columns:
                df = df.dropna(subset=['GHI'])
                dfs.append(df)
                valid_countries.append(country.lower())
    
    return dfs, valid_countries

def get_summary_stats(dfs: list[pd.DataFrame], countries: list[str]) -> pd.DataFrame:
    """
    Generate a summary table of GHI statistics for selected countries.
    
    Args:
        dfs (list): List of DataFrames for each country.
        countries (list): List of country names.
    
    Returns:
        pd.DataFrame: Summary table with mean, median, and std dev of GHI.
    """
    summary = []
    for country, df in zip(countries, dfs):
        summary.append({
            "Country": country.capitalize(),
            "Average GHI (W/m²)": round(df["GHI"].mean(), 2),
            "Median GHI (W/m²)": round(df["GHI"].median(), 2),
            "Std Dev GHI (W/m²)": round(df["GHI"].std(), 2)
        })
    return pd.DataFrame(summary)