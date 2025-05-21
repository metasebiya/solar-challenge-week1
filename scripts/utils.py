import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


def load_data(file_path):
    """Load CSV data into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file path is invalid.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {file_path}") from e


def convert_timestamp_to_index(df, timestamp_col='Timestamp'):
    """Convert a timestamp column to a datetime index.

    Args:
        df (pd.DataFrame): Input DataFrame with a timestamp column.
        timestamp_col (str, optional): Name of the timestamp column. Defaults to 'Timestamp'.

    Returns:
        pd.DataFrame: DataFrame with datetime index.
    """
    df_copy = df.copy()
    df_copy[timestamp_col] = pd.to_datetime(df_copy[timestamp_col])
    df_copy.set_index(timestamp_col, inplace=True)
    return df_copy



def report_negative_values(df, columns=None):
    """Report the number and percentage of negative values in specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list, optional): Columns to check for negative values.
            Defaults to ['GHI', 'DNI', 'DHI'].

    Returns:
        dict: Dictionary with column names as keys and a tuple of
              (negative_count, negative_percentage) as values.
    """
    if columns is None:
        columns = ['GHI', 'DNI', 'DHI']
    result = {}
    for col in columns:
        if col in df.columns:
            neg_count = len(df[df[col] < 0])
            neg_percentage = round(neg_count / len(df) * 100, 2)
            result[col] = (neg_count, neg_percentage)
    return result


def replace_negative_with_median(df, columns=None):
    """Replace negative values with the median of non-negative values in specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list, optional): Columns to process. Defaults to ['GHI', 'DNI', 'DHI'].

    Returns:
        pd.DataFrame: DataFrame with negative values replaced.
    """
    if columns is None:
        columns = ['GHI', 'DNI', 'DHI']
    df_cleaned = df.copy()
    for col in columns:
        if col in df_cleaned.columns:
            median_val = df_cleaned[df_cleaned[col] >= 0][col].median()
            df_cleaned[col] = df_cleaned[col].apply(
                lambda x: median_val if x < 0 else x
            )
    return df_cleaned


def clean_data(df, columns=None):
    """Clean DataFrame by replacing negative values in specified columns with zero.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list, optional): Columns to clean. Defaults to ['GHI', 'DNI', 'DHI'].

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if columns is None:
        columns = ['GHI', 'DNI', 'DHI']
    df_cleaned = df.copy()
    for col in columns:
        if col in df_cleaned.columns:
            df_cleaned[col] = df_cleaned[col].clip(lower=0)
    return df_cleaned


def detect_outliers(df, columns=None, threshold=3):
    """Detect outliers in specified columns using z-scores.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list, optional): Columns to check for outliers.
            Defaults to ['ModA', 'ModB', 'WS', 'WSgust'].
        threshold (float, optional): Z-score threshold for outliers. Defaults to 3.

    Returns:
        dict: Dictionary with column names as keys and a tuple of
              (outlier_count, outlier_percentage, outlier_indices) as values.
    """
    if columns is None:
        columns = ['ModA', 'ModB', 'WS', 'WSgust']
    result = {}
    z_scores_df = df[columns].apply(zscore)
    for col in columns:
        if col in df.columns:
            outlier_flags = z_scores_df[col].abs() > threshold
            outlier_indices = df.index[outlier_flags].tolist()
            outlier_count = len(outlier_indices)
            outlier_percentage = round(outlier_count / len(df) * 100, 2)
            result[col] = (outlier_count, outlier_percentage, outlier_indices)
    return result

def replace_outliers_with_mean(dataframe, columns, threshold=3):
    """Replace outliers in specified columns with the mean of non-outlier values using z-scores.

    Args:
        dataframe (pd.DataFrame): Input DataFrame containing the data.
        columns (list): List of column names to check for outliers.
        threshold (float, optional): Z-score threshold for identifying outliers. Defaults to 3.

    Returns:
        pd.DataFrame: DataFrame with outliers replaced by the mean of non-outlier values.
    """
    df = dataframe.copy()
    for col in columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in DataFrame. Skipping.")
            continue

        # Compute z-scores for the column
        z_scores = zscore(df[col].dropna())
        z_scores = pd.Series(z_scores, index=df[col].dropna().index)
        
        # Identify outliers (absolute z-score > threshold)
        outlier_flags = z_scores.abs() > threshold
        
        # Calculate mean of non-outlier values
        non_outlier_mean = df.loc[~outlier_flags, col].mean()
        
        # Replace outliers with the mean
        df.loc[outlier_flags, col] = non_outlier_mean
        print(f"Replaced {outlier_flags.sum()} outliers in column '{col}' with mean {non_outlier_mean:.2f}")
    
    return df


def plot_outliers(df, columns=None, threshold=3, save_path=None):
    """Create scatter plots highlighting outliers for specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list, optional): Columns to plot. Defaults to ['ModA', 'ModB', 'WS', 'WSgust'].
        threshold (float, optional): Z-score threshold for outliers. Defaults to 3.
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    if columns is None:
        columns = ['ModA', 'ModB', 'WS', 'WSgust']
    fig, axes = plt.subplots(
        nrows=1, ncols=len(columns), figsize=(5 * len(columns), 4), sharey=False
    )
    axes = [axes] if len(columns) == 1 else axes
    for i, col in enumerate(columns):
        if col in df.columns:
            z_scores = zscore(df[col])
            outliers = np.abs(z_scores) > threshold
            ax = axes[i]
            ax.scatter(df.index, df[col], label='All Data', alpha=0.7)
            ax.scatter(
                df.index[outliers], df.loc[outliers, col],
                color='r', label='Outliers (|Z| > 3)'
            )
            ax.set_title(f'{col}')
            ax.set_xlabel('Index')
            ax.set_ylabel(col)
            ax.legend(loc='upper right')
    plt.suptitle('Outlier Visualization (|Z| > 3)', fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def data_overview(df):
    """Generate a summary of the DataFrame including shape, dtypes, and missing values.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        dict: Dictionary containing shape, dtypes, summary statistics, and missing values.
    """
    return {
        'shape': df.shape,
        'dtypes': df.dtypes.to_dict(),
        'describe': df.describe().to_dict(),
        'missing_values': df.isna().sum().to_dict()
    }

def plot_time_series(df, columns=None, save_path=None):
    """Create line plots for specified columns over time.

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime index.
        columns (list, optional): Columns to plot. Defaults to ['GHI', 'DNI', 'DHI', 'Tamb'].
        save_path (str, optional): Path to save the plot. If None, display the plot.

    Raises:
        ValueError: If the DataFrame index is not a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    if columns is None:
        columns = ['GHI', 'DNI', 'DHI', 'Tamb']
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, len(columns), figsize=(20, 4), sharex=True)
    axes = [axes] if len(columns) == 1 else axes
    for ax, col in zip(axes, columns):
        if col in df.columns:
            ax.plot(df.index, df[col], color='tab:blue')
            ax.set_title(f'{col}', fontsize=14)
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_monthly_trends(df, variables=None, units=None, save_path=None):
    """Create bar charts for monthly trends of specified variables.

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime index.
        variables (list, optional): Variables to plot. Defaults to ['GHI', 'DNI', 'DHI', 'Tamb'].
        units (list, optional): Units for variables. Defaults to ['W/m²', 'W/m²', 'W/m²', '°C'].
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    if variables is None:
        variables = ['GHI', 'DNI', 'DHI', 'Tamb']
    if units is None:
        units = ['W/m²', 'W/m²', 'W/m²', '°C']
    monthly_avg = df.groupby(df.index.month)[variables].mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for i, (var, unit) in enumerate(zip(variables, units)):
        if var in df.columns:
            axes[i].bar(monthly_avg.index, monthly_avg[var])
            axes[i].set_title(f'Monthly Average {var}')
            axes[i].set_xlabel('Months')
            axes[i].set_ylabel(f'{var} ({unit})')
            axes[i].grid(True, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_daily_trends(df, variables=None, units=None, save_path=None):
    """Create line plots for daily trends of specified variables by hour.

    Args:
        df (pd.DataFrame): Input DataFrame with a datetime index.
        variables (list, optional): Variables to plot. Defaults to ['GHI', 'DNI', 'DHI', 'Tamb'].
        units (list, optional): Units for variables. Defaults to ['W/m²', 'W/m²', 'W/m²', '°C'].
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    if variables is None:
        variables = ['GHI', 'DNI', 'DHI', 'Tamb']
    if units is None:
        units = ['W/m²', 'W/m²', 'W/m²', '°C']
    hourly_avg = df.groupby(df.index.hour)[variables].mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()
    for i, (var, unit) in enumerate(zip(variables, units)):
        if var in df.columns:
            axes[i].plot(hourly_avg.index, hourly_avg[var], marker='o', color='blue')
            axes[i].set_title(f'Daily Average {var} by Hour')
            axes[i].set_xlabel('Hour of Day')
            axes[i].set_ylabel(f'{var} ({unit})')
            axes[i].grid(True)
            axes[i].set_xticks(range(0, 24, 2))
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_cleaning_impact(df, save_path=None):
    """Create bar chart for average ModA and ModB by Cleaning status.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Cleaning', 'ModA', and 'ModB' columns.
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    cleaning_impact = df.groupby('Cleaning')[['ModA', 'ModB']].mean()
    cleaning_impact.plot(kind='bar', figsize=(8, 6))
    plt.title('Average ModA and ModB Before and After Cleaning')
    plt.xlabel('Cleaning (0 = No, 1 = Yes)')
    plt.ylabel('Average Value (W/m²)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_correlation_heatmap(df, columns=None, save_path=None):
    """Create a correlation heatmap for specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (list, optional): Columns for correlation. Defaults to ['GHI', 'DNI', 'DHI', 'TModA', 'TModB'].
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    if columns is None:
        columns = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
    corr_matrix = df[columns].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



def plot_wind_vs_ghi(df, save_path=None):
    """Create regression plots for wind-related variables vs. GHI.

    Args:
        df (pd.DataFrame): Input DataFrame with 'WS', 'WSgust', 'WD', and 'GHI' columns.
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, var, label in zip(
        axes,
        ['WS', 'WSgust', 'WD'],
        ['Wind Speed (m/s)', 'Wind Gust Speed (m/s)', 'Wind Direction (°N)']
    ):
        sns.regplot(
            x=var, y='GHI', data=df, ax=ax,
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}
        )
        ax.set_title(f'{var} vs GHI')
        ax.set_xlabel(label)
        ax.set_ylabel('GHI (W/m²)')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_rh_vs_tamb(df, save_path=None):
    """Create regression plot for Relative Humidity vs. Ambient Temperature.

    Args:
        df (pd.DataFrame): Input DataFrame with 'RH' and 'Tamb' columns.
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.regplot(
        x='RH', y='Tamb', data=df,
        scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}
    )
    plt.title('Relative Humidity vs Ambient Temperature')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Ambient Temperature (°C)')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_rh_vs_dhi(df, save_path=None):
    """Create regression plot for Relative Humidity vs. Diffuse Horizontal Irradiance.

    Args:
        df (pd.DataFrame): Input DataFrame with 'RH' and 'DHI' columns.
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.regplot(
        x='RH', y='DHI', data=df,
        scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}
    )
    plt.title('Relative Humidity vs Diffuse Horizontal Irradiance')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('DHI (W/m²)')
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_wind_direction(df, save_path=None):
    """Create radial bar plot for average wind speed by wind direction.

    Args:
        df (pd.DataFrame): Input DataFrame with 'WD' and 'WS' columns.
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    bins = np.arange(0, 360 + 23, 23)
    df['WD_bin'] = pd.cut(df['WD'], bins=bins, include_lowest=True, labels=bins[:-1])
    wd_speed = df.groupby('WD_bin')['WS'].mean()
    angles = np.deg2rad(wd_speed.index.astype(float))
    width = np.deg2rad(22.5)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.bar(angles, wd_speed, width=width, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_theta_direction(-1)
    ax.set_theta_offset(np.pi / 2)
    ax.set_title('Average Wind Speed by Direction')
    ax.set_ylabel('Wind Speed (m/s)', labelpad=30)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_histograms(df, save_path=None):
    """Create histograms for GHI and WS.

    Args:
        df (pd.DataFrame): Input DataFrame with 'GHI' and 'WS' columns.
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data=df, x='GHI', bins=30, ax=axes[0], edgecolor='black')
    axes[0].set_title('Distribution of GHI')
    axes[0].set_xlabel('GHI (W/m²)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    sns.histplot(data=df, x='WS', bins=30, ax=axes[1], edgecolor='black', color='green')
    axes[1].set_title('Distribution of Wind Speed')
    axes[1].set_xlabel('Wind Speed (m/s)')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_rh_vs_variables(df, variables, units, save_path=None):
    """Create regression plots for RH vs. multiple variables.

    Args:
        df (pd.DataFrame): Input DataFrame with 'RH' and specified variable columns.
        variables (list): List of column names to plot against 'RH'.
        units (list): List of units for each variable.
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (var, unit) in enumerate(zip(variables, units)):
        sns.regplot(
            x='RH', y=var, data=df, ax=axes[i],
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}
        )
        axes[i].set_title(f'RH vs {var}')
        axes[i].set_xlabel('Relative Humidity (%)')
        axes[i].set_ylabel(f'{var} ({unit})')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_bubble_charts(df, save_path=None):
    """Create bubble charts for GHI vs. Tamb with bubble sizes based on RH and BP.

    Args:
        df (pd.DataFrame): Input DataFrame with 'GHI', 'Tamb', 'RH', and 'BP' columns.
        save_path (str, optional): Path to save the plot. If None, display the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    rh_sizes = [df['RH'].min(), df['RH'].median(), df['RH'].max()]
    rh_sizes = [round(x, 1) for x in rh_sizes]
    bp_sizes = [df['BP'].min(), df['BP'].median(), df['BP'].max()]
    bp_sizes = [round(x, 1) for x in bp_sizes]
    # GHI vs Tamb, bubble size = RH
    axes[0].scatter(df['Tamb'], df['GHI'], s=df['RH']*10, alpha=0.5, c='blue')
    axes[0].set_title('GHI vs Tamb (Bubble Size = RH)')
    axes[0].set_xlabel('Ambient Temperature (°C)')
    axes[0].set_ylabel('GHI (W/m²)')
    axes[0].grid(True, alpha=0.3)
    for size in rh_sizes:
        axes[0].scatter([], [], s=size*10, label=f'RH={size}%', c='blue', alpha=0.5)
    axes[0].legend(title='Bubble Size')
    # GHI vs Tamb, bubble size = BP
    bp_scaled = (df['BP'] - df['BP'].min()) / (df['BP'].max() - df['BP'].min()) * 1000
    axes[1].scatter(df['Tamb'], df['GHI'], s=bp_scaled, alpha=0.5, c='green')
    axes[1].set_title('GHI vs Tamb (Bubble Size = BP)')
    axes[1].set_xlabel('Ambient Temperature (°C)')
    axes[1].set_ylabel('GHI (W/m²)')
    axes[1].grid(True, alpha=0.3)
    for size in bp_sizes:
        scaled_size = (size - df['BP'].min()) / (df['BP'].max() - df['BP'].min()) * 1000
        axes[1].scatter([], [], s=scaled_size, label=f'BP={int(size)}mbar', c='green', alpha=0.5)
    axes[1].legend(title='Bubble Size')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()