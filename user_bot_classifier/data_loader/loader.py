# data_loader/loader.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """Handles loading data from a CSV file."""

    def __init__(self, file_path):
        """
        Initializes the DataLoader.

        Args:
            file_path (str): The path to the CSV file.
        """
        self.file_path = file_path
        logging.info(f"DataLoader initialized with file path: {self.file_path}")

    # Убираем параметр skip_rows из определения метода, если он не нужен для других целей
    def load_data(self, na_values=['Unknown', '']):
        """
        Loads data from the specified CSV file. Assumes header is on the first line.

        Args:
            na_values (list): List of strings to recognize as NaN/missing values.

        Returns:
            pd.DataFrame: Loaded data as a pandas DataFrame, or None if loading fails.
        """
        try:
            # *** ИЗМЕНЕНИЕ ЗДЕСЬ ***
            # Указываем, что заголовок находится в первой строке (индекс 0)
            # Убираем skiprows, так как первая строка - это нужный заголовок
            df = pd.read_csv(self.file_path, header=0, na_values=na_values)
            logging.info(f"Successfully loaded data from {self.file_path}. Shape: {df.shape}")

            # Удаляем 'Unnamed:' столбцы, если они появляются
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            logging.info(f"Removed unnamed columns. Shape after removal: {df.shape}")
            return df
        except FileNotFoundError:
            logging.error(f"Error: File not found at {self.file_path}")
            return None
        except Exception as e:
            logging.error(f"An error occurred during data loading: {e}")
            return None