import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Check if required columns exist
    required_cols = {'deltaX', 'deltaY'}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    # Check if columns are numeric
    for col in required_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column {col} must be numeric")

    # Create filtering condition (using bitwise operator &)
    condition = (
            (df['deltaX'].abs() <= 100) &
            (df['deltaY'].abs() <= 100)
    )

    # Perform filtering and generate a new DataFrame
    filtered_df = df.loc[condition].copy()

    # Statistics of filtering results
    original_count = len(df)
    filtered_count = len(filtered_df)
    print(f"Filtering complete: {original_count} rows originally → {filtered_count} rows retained "
          f"({original_count - filtered_count} rows removed)")

    return filtered_df

def z_score_normalize(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")

    # Handle Empty Data
    if series.empty:
        return series

    # Copy Data to Avoid Modifying Original Series
    clean_series = series.dropna().copy()

    # Calculate Statistics
    mean = clean_series.mean()
    std = clean_series.std(ddof=0)  # Use population standard deviation

    # Handle Special Cases (Constant Data)
    if std == 0:
        # Return 0 or maintain original values if mean is non-zero
        return pd.Series(0 if mean != 0 else clean_series.values,
                         index=clean_series.index)

    # Perform Standardization
    normalized = (clean_series - mean) / std

    # Rebuild Original Index Structure
    return series.map(lambda x: (x - mean) / std if pd.notnull(x) else np.nan)


def min_max_normalize(series: pd.Series, feature_range: tuple = (0, 1)) -> pd.Series:
    if len(feature_range) != 2:
        raise ValueError(" feature_range must contain two elements")

    # Automatically Sort Range Parameters
    min_bound, max_bound = sorted(feature_range)

    # Handle Empty Data
    if series.empty:
        return series

    clean_series = series.dropna().copy()

    # Calculate Extremes
    data_min = clean_series.min()
    data_max = clean_series.max()
    data_range = data_max - data_min

    # Handle Constant Data (Zero Range)
    if data_range == 0:
        return pd.Series(min_bound, index=clean_series.index)

    # Perform Normalization
    scaled = (clean_series - data_min) / data_range
    normalized = scaled * (max_bound - min_bound) + min_bound

    # Rebuild Complete Index
    return series.map(lambda x: ((x - data_min) / data_range) * (max_bound - min_bound) + min_bound
    if pd.notnull(x) else np.nan)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame !")

    # Create a copy to avoid modifying the input DataFrame
    normalized_df = df.copy()

    # Apply min-max normalization to 'X' and 'Y' columns
    for column in ['X1', 'Y1', 'Frequency_ID', 'X2', 'Y2']:
        if column in df.columns:
            normalized_df[column] = min_max_normalize(df[column])
        else:
            raise KeyError(f"The column '{column}' does not exist in the DataFrame !")

    # # Apply z-score normalization to 'deltaX' and 'deltaY' columns
    # for column in ['deltaX', 'deltaY']:
    #     if column in df.columns:
    #         normalized_df[column] = z_score_normalize(df[column])
    #     else:
    #         raise KeyError(f"The column '{column}' does not exist in the DataFrame !")

    return normalized_df


def print_stats(df: pd.DataFrame):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame !")

    col_means = df.mean()
    col_stds = df.std()
    col_mins = df.min()
    col_maxs = df.max()

    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.titlesize': 14
    })

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"\nField: {col}")
            print(f"Mean: {col_means[col]:.2f}")
            print(f"Standard deviation: {col_stds[col]:.2f}")
            print(f"Minimum: {col_mins[col]:.4f}")
            print(f"Maximum: {col_maxs[col]:.4f}")
            print("-" * 50)

            plt.figure()
            df[col].hist(grid=False, edgecolor='black', bins=100)

            # 添加标注
            plt.axvline(col_means[col], color='red', linestyle='dashed', linewidth=2,
                        label=f'Mean: {col_means[col]:.2f}')
            plt.axvline(col_means[col] - col_stds[col], color='orange', linestyle='dotted',
                        linewidth=2, label=f'±1 STD')
            plt.axvline(col_means[col] + col_stds[col], color='orange', linestyle='dotted',
                        linewidth=2)

            plt.title(f'Distribution of {col}', pad=20)
            plt.xlabel(col, labelpad=15)
            plt.ylabel('Frequency', labelpad=15)
            plt.legend()
            plt.tight_layout()
            plt.show()

        else:
            print(f"\nField: {col} [Non-numeric column skipped]")
            print("-" * 50)

def split_dataset(df):
    # Randomly split the data set with fixed random seed
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save the split data set
    train_df.to_csv("./Data/train_data.csv", index=False)
    test_df.to_csv("./Data/test_data.csv", index=False)

    print("The data set has been successfully split and saved！")

if __name__ == "__main__":
    # Read the dataset
    data_path = "./Data/full_data_1.csv"
    df = pd.read_csv(data_path)

    # filtered_df = filter_outliers(df)

    normalize_df = normalize(df)
    # normalize_df.to_csv("./Data/normalized_data.csv", index=False)
    print_stats(normalize_df)

    split_dataset(normalize_df)

    # split_dataset(df)