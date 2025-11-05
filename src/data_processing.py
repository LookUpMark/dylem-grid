"""
Data loading, preprocessing, and transformation functions for gesture recognition
Handles CSV loading, data cleaning, normalization, PCA transformation, and tensor preparation
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
import torch


def data_loader(data_path, data_type):
    """
    Load CSV files from the DYLEM-GRID dataset
    
    Args:
        data_path: Base path to the dataset
        data_type: Type of data ('Raw' or 'Cleaned')
    
    Returns:
        lst: List of DataFrames containing the data
        labels: List of corresponding labels
    """
    lst = []
    labels = []
    data_path = os.path.join(data_path, 'DYLEM-GRID_' + data_type)
    
    # First collect all CSV file paths
    file_paths = []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for ges_folder in os.listdir(folder_path):
            gesfolder_path = os.path.join(folder_path, ges_folder)
            if not os.path.isdir(gesfolder_path):
                continue
            for file in os.listdir(gesfolder_path):
                if file.endswith('.csv'):
                    file_paths.append((os.path.join(gesfolder_path, file), ges_folder))

    # Load all CSV files
    for file_path, label in file_paths:
        try:
            df = pd.read_csv(file_path)
            lst.append(df)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return lst, labels


def data_preprocess(data, labels):
    """
    Preprocess the data:
    1. Backward fill missing values
    2. Remove duplicate columns per sample
    3. Concatenate all samples
    4. Remove low-variance columns globally
    5. Remove outliers per class
    6. Normalize globally with MinMaxScaler
    7. Split back into individual samples
    
    Args:
        data: List of DataFrames
        labels: List of corresponding labels
    
    Returns:
        processed_data: List of processed DataFrames
        labels: Shuffled list of labels (synchronized with data)
    """
    # Step 1: Handle missing values per sample
    for df in data:
        df.bfill(inplace=True)

    # Step 2: Drop duplicate columns on each sample
    cleaned_data = []
    for df in data:
        df_clean = df.loc[:, ~df.T.duplicated(keep='first')]
        cleaned_data.append(df_clean)
    data = cleaned_data
    
    # Step 3: Concatenate all dataframes
    lengths = [len(df) for df in data]
    concatenated = pd.concat(data, ignore_index=True)
    
    # Create a label column to track which class each row belongs to
    label_column = []
    for label, length in zip(labels, lengths):
        label_column.extend([label] * length)
    concatenated['_class_label'] = label_column
    
    # Step 4: Drop columns where a single value occupies more than 90% of rows
    if not concatenated.empty:
        cols_to_check = [col for col in concatenated.columns if col != '_class_label']
        max_frac = concatenated[cols_to_check].apply(
            lambda col: col.value_counts(normalize=True).max() if len(col) > 0 else 0
        )
        cols_to_keep = max_frac[max_frac < 0.9].index.tolist() + ['_class_label']
        concatenated = concatenated[cols_to_keep]
    
    # Step 5: Find and remove outliers per class
    for label in concatenated['_class_label'].unique():
        class_mask = concatenated['_class_label'] == label
        for column in concatenated.columns:
            if column == '_class_label':
                continue
            if pd.api.types.is_numeric_dtype(concatenated[column]):
                # Convert to float if it's an integer type
                if pd.api.types.is_integer_dtype(concatenated[column]):
                    concatenated[column] = concatenated[column].astype(np.float64)
                
                # Calculate mean and std for this class only
                class_data = concatenated.loc[class_mask, column]
                mean = class_data.mean()
                std = class_data.std()
                
                # Skip if std is 0 or NaN (constant column for this class)
                if pd.isna(std) or std == 0:
                    continue
                    
                threshold = 3 * std
                # Find outliers within this class
                outliers = class_mask & ((concatenated[column] - mean).abs() > threshold)
                concatenated.loc[outliers, column] = mean

    # Step 6: Normalize the concatenated data (excluding label column)
    scaler = MinMaxScaler()
    numeric_cols = [col for col in concatenated.select_dtypes(include=[np.number]).columns 
                    if col != '_class_label']
    if len(numeric_cols) > 0 and not concatenated.empty:
        concatenated[numeric_cols] = scaler.fit_transform(concatenated[numeric_cols])
    
    # Remove the label column before splitting
    concatenated = concatenated.drop('_class_label', axis=1)
    
    # Step 7: Split back into individual dataframes
    processed_data = []
    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length
        processed_data.append(concatenated.iloc[start_idx:end_idx].copy())
        start_idx = end_idx
    
    # Shuffle the data and labels together
    combined = list(zip(processed_data, labels))
    random.shuffle(combined)
    processed_data, labels = zip(*combined)
    
    return list(processed_data), list(labels)


def apply_pca(data, labels, variance_threshold=0.95):
    """
    Apply PCA to reduce dimensionality while retaining specified variance
    
    Args:
        data: List of DataFrames
        labels: List of corresponding labels
        variance_threshold: Fraction of variance to retain (default: 0.95)
    
    Returns:
        transformed_data: List of DataFrames with PCA-transformed features
        labels: Unchanged list of labels
    """
    # Keep track of original lengths to split later
    lengths = [len(df) for df in data]
    concatenated = pd.concat(data, ignore_index=True)
    
    # Get numeric columns
    numeric_cols = concatenated.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0 or concatenated.empty:
        return data, labels
    
    pca = PCA(n_components=variance_threshold)
    transformed = pca.fit_transform(concatenated[numeric_cols])
    
    # Create DataFrame with transformed data
    transformed_df = pd.DataFrame(
        transformed, 
        columns=[f'PC{i+1}' for i in range(transformed.shape[1])]
    )
    
    # Split back into individual dataframes
    transformed_data = []
    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length
        transformed_data.append(transformed_df.iloc[start_idx:end_idx].copy())
        start_idx = end_idx
    
    return transformed_data, labels


def prepare_data(data, labels):
    """
    Convert list of dataframes to padded tensors
    
    Args:
        data: List of DataFrames
        labels: List of corresponding labels
    
    Returns:
        X: Tensor of shape (n_samples, max_seq_len, n_features)
        y: Tensor of encoded labels
        label_encoder: Fitted LabelEncoder
    """
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Find max sequence length
    max_len = max(len(df) for df in data)
    input_size = data[0].shape[1]
    
    # Pad sequences
    padded_data = []
    for df in data:
        arr = df.values
        if len(arr) < max_len:
            # Pad with zeros
            padding = np.zeros((max_len - len(arr), input_size))
            arr = np.vstack([arr, padding])
        padded_data.append(arr)
    
    X = np.array(padded_data, dtype=np.float32)
    y = np.array(encoded_labels, dtype=np.int64)
    
    return torch.FloatTensor(X), torch.LongTensor(y), label_encoder
