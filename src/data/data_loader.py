"""
Data loading module for the Box Office Revenue Prediction project.
"""

import os
import logging
import pandas as pd
from typing import Optional, Union, List

logger = logging.getLogger(__name__)


class DataLoader:
    """Class to load and prepare movie data for analysis."""

    def __init__(self, data_path: str):
        """
        Initialize the DataLoader with the path to the data file.

        Args:
            data_path (str): Path to the raw data file.
        """
        self.data_path = data_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load the movie data from the specified path.

        Returns:
            pd.DataFrame: The loaded movie data.
        """
        logger.info(f"Loading data from {self.data_path}")

        # Check if file exists
        if not os.path.exists(self.data_path):
            logger.warning(
                f"Data file {self.data_path} does not exist. Using sample data."
            )
            return self._create_sample_data()

        # Determine file type and load accordingly
        file_ext = os.path.splitext(self.data_path)[1].lower()
        try:
            if file_ext == ".csv":
                self.data = pd.read_csv(self.data_path)
            elif file_ext == ".json":
                self.data = pd.read_json(self.data_path)
            elif file_ext in [".xls", ".xlsx"]:
                self.data = pd.read_excel(self.data_path)
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                raise ValueError(f"Unsupported file format: {file_ext}")

            logger.info(f"Successfully loaded data with shape {self.data.shape}")
            return self.data

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _create_sample_data(self) -> pd.DataFrame:
        """
        Create a small sample dataset for testing purposes.

        Returns:
            pd.DataFrame: A sample movie dataset.
        """
        logger.info("Creating sample movie dataset")

        # Create a sample dataset with key features
        sample_data = {
            "id": list(range(1, 101)),
            "title": [f"Movie {i}" for i in range(1, 101)],
            "release_date": pd.date_range(start="2010-01-01", periods=100, freq="M"),
            "budget": [1000000 * (i % 10 + 1) for i in range(1, 101)],
            "runtime": [90 + (i % 60) for i in range(1, 101)],
            "genres": [
                "Action|Adventure",
                "Comedy|Romance",
                "Drama|Thriller",
                "Horror|Mystery",
                "Animation|Family",
                "Sci-Fi|Fantasy",
                "Documentary",
                "Crime|Drama",
                "Action|Comedy",
                "Romance|Drama",
            ]
            * 10,
            "original_language": ["en"] * 70 + ["es"] * 10 + ["fr"] * 10 + ["ja"] * 10,
            "popularity": [50 + (i % 50) for i in range(1, 101)],
            "vote_average": [(i % 10) / 2 + 5 for i in range(1, 101)],
            "vote_count": [100 * (i % 20 + 1) for i in range(1, 101)],
            "revenue": [5000000 * (i % 20 + 1) for i in range(1, 101)],
        }

        sample_df = pd.DataFrame(sample_data)
        logger.info(f"Created sample dataset with shape {sample_df.shape}")

        # Save the sample data
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        sample_df.to_csv(self.data_path, index=False)
        logger.info(f"Saved sample dataset to {self.data_path}")

        return sample_df

    def merge_additional_data(
        self, additional_data_path: str, on: str = "id"
    ) -> pd.DataFrame:
        """
        Merge additional data with the main dataset.

        Args:
            additional_data_path (str): Path to the additional data file.
            on (str): Column to join on.

        Returns:
            pd.DataFrame: The merged dataset.
        """
        if self.data is None:
            self.load_data()

        logger.info(f"Merging additional data from {additional_data_path}")

        try:
            # Load additional data
            file_ext = os.path.splitext(additional_data_path)[1].lower()
            if file_ext == ".csv":
                additional_data = pd.read_csv(additional_data_path)
            elif file_ext == ".json":
                additional_data = pd.read_json(additional_data_path)
            elif file_ext in [".xls", ".xlsx"]:
                additional_data = pd.read_excel(additional_data_path)
            else:
                logger.error(f"Unsupported file format: {file_ext}")
                raise ValueError(f"Unsupported file format: {file_ext}")

            # Merge data
            self.data = pd.merge(self.data, additional_data, on=on, how="left")
            logger.info(f"Successfully merged data. New shape: {self.data.shape}")

            return self.data

        except Exception as e:
            logger.error(f"Error merging additional data: {str(e)}")
            raise
