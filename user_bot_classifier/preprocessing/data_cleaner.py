# preprocessing/data_cleaner.py
import pandas as pd
import logging
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataCleaner:
    """Cleans the raw data."""

    def __init__(self):
        """Initializes the DataCleaner."""
        self.numeric_imputer = None
        self.categorical_imputer = None
        logging.info("DataCleaner initialized.")

    def identify_column_types(self, df, target_column):
        """
        Identifies numeric and categorical columns automatically.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_column (str): Name of the target variable column.

        Returns:
            tuple: Lists of numeric and categorical column names.
        """
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Remove target column from feature lists
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

        logging.info(f"Identified {len(numeric_cols)} numeric columns.")
        logging.info(f"Identified {len(categorical_cols)} categorical columns.")
        return numeric_cols, categorical_cols

    def fit_imputers(self, df, numeric_cols, categorical_cols, num_strategy='median', cat_strategy='constant', fill_value='Missing'):
        """
        Fits imputers for numeric and categorical columns.

        Args:
            df (pd.DataFrame): DataFrame to fit imputers on (usually training data).
            numeric_cols (list): List of numeric column names.
            categorical_cols (list): List of categorical column names.
            num_strategy (str): Imputation strategy for numeric columns ('mean', 'median', etc.).
            cat_strategy (str): Imputation strategy for categorical columns ('constant', 'most_frequent').
            fill_value (str): Value to use for constant categorical imputation.
        """
        if numeric_cols:
            self.numeric_imputer = SimpleImputer(strategy=num_strategy)
            self.numeric_imputer.fit(df[numeric_cols])
            logging.info(f"Numeric imputer fitted with strategy: {num_strategy}")
        else:
             logging.warning("No numeric columns provided for numeric imputer fitting.")

        if categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy=cat_strategy, fill_value=fill_value)
            self.categorical_imputer.fit(df[categorical_cols])
            logging.info(f"Categorical imputer fitted with strategy: {cat_strategy}, fill_value: {fill_value}")
        else:
            logging.warning("No categorical columns provided for categorical imputer fitting.")


    def impute_missing(self, df, numeric_cols, categorical_cols):
        """
        Imputes missing values using pre-fitted imputers.

        Args:
            df (pd.DataFrame): DataFrame to impute.
            numeric_cols (list): List of numeric column names.
            categorical_cols (list): List of categorical column names.

        Returns:
            pd.DataFrame: DataFrame with missing values imputed.
        """
        df_imputed = df.copy()
        if self.numeric_imputer and numeric_cols:
            # Only impute columns present in the current dataframe
            cols_to_impute_num = [col for col in numeric_cols if col in df_imputed.columns]
            if cols_to_impute_num:
                df_imputed[cols_to_impute_num] = self.numeric_imputer.transform(df_imputed[cols_to_impute_num])
                logging.info(f"Imputed missing values in {len(cols_to_impute_num)} numeric columns.")
            else:
                logging.warning("Numeric columns specified for imputation not found in the dataframe.")
        elif numeric_cols:
             logging.warning("Numeric imputer not fitted, skipping numeric imputation.")

        if self.categorical_imputer and categorical_cols:
             # Only impute columns present in the current dataframe
            cols_to_impute_cat = [col for col in categorical_cols if col in df_imputed.columns]
            if cols_to_impute_cat:
                df_imputed[cols_to_impute_cat] = self.categorical_imputer.transform(df_imputed[cols_to_impute_cat])
                logging.info(f"Imputed missing values in {len(cols_to_impute_cat)} categorical columns.")
            else:
                 logging.warning("Categorical columns specified for imputation not found in the dataframe.")
        elif categorical_cols:
            logging.warning("Categorical imputer not fitted, skipping categorical imputation.")

        return df_imputed


    def drop_duplicates(self, df):
        """
        Removes duplicate rows from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with duplicates removed.
        """
        initial_rows = len(df)
        df_cleaned = df.drop_duplicates()
        rows_removed = initial_rows - len(df_cleaned)
        if rows_removed > 0:
            logging.info(f"Removed {rows_removed} duplicate rows.")
        else:
            logging.info("No duplicate rows found.")
        return df_cleaned

    def clean_data(self, df, target_column):
        """
        Applies cleaning steps: identifies types, fits imputers (call separately on train),
        imputes missing values, and drops duplicates.

        Args:
            df (pd.DataFrame): Input DataFrame.
            target_column (str): Name of the target variable column.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        logging.info("Starting data cleaning process...")
        # Drop duplicates first
        df_cleaned = self.drop_duplicates(df)

        # Identify types (needed for imputation)
        numeric_cols, categorical_cols = self.identify_column_types(df_cleaned, target_column)

        # Impute missing values (Imputers must be fitted first on training data)
        if self.numeric_imputer or self.categorical_imputer:
             df_cleaned = self.impute_missing(df_cleaned, numeric_cols, categorical_cols)
        else:
             logging.warning("Imputers not fitted. Call fit_imputers() on training data first before cleaning test/new data.")


        logging.info("Data cleaning process finished.")
        return df_cleaned