"""
Data preprocessing module for the Box Office Revenue Prediction project.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Preprocessor:
    """Class to preprocess movie data for model training."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the Preprocessor.

        Args:
            random_state (int): Random state for reproducibility.
        """
        self.random_state = random_state
        self.numerical_features = None
        self.categorical_features = None
        self.date_features = None
        self.text_features = None
        self.target = "revenue"
        self.preprocessor = None
        self.scaler = StandardScaler()

    def process(
        self, data: pd.DataFrame, test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Preprocess the data and split into training and test sets.

        Args:
            data (pd.DataFrame): The raw movie data.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Processed training and test data and targets.
        """
        logger.info("Starting data preprocessing")

        # Make a copy to avoid modifying the original data
        df = data.copy()

        # Identify feature types
        self._identify_feature_types(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        # Feature transformation
        df = self._transform_features(df)

        # Handle outliers
        df = self._handle_outliers(df)

        # Split data into features and target
        X = df.drop(columns=[self.target])
        y = df[self.target]

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        logger.info(
            f"Data split into train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets"
        )

        # Save processed data
        self._save_processed_data(X_train, X_test, y_train, y_test)

        return X_train, X_test, y_train, y_test

    def _identify_feature_types(self, df: pd.DataFrame) -> None:
        """
        Identify the types of features in the dataset.

        Args:
            df (pd.DataFrame): The movie data.
        """
        # Identify numerical features (excluding the target)
        self.numerical_features = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        if self.target in self.numerical_features:
            self.numerical_features.remove(self.target)

        # Identify categorical features
        self.categorical_features = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Identify date features
        self.date_features = [
            col for col in df.columns if "date" in col.lower() or "time" in col.lower()
        ]

        # Identify text features (assuming longer string columns are text)
        self.text_features = [
            col
            for col in self.categorical_features
            if df[col].astype(str).str.len().mean() > 20
        ]

        # Remove text features from categorical features
        self.categorical_features = [
            col for col in self.categorical_features if col not in self.text_features
        ]

        logger.info(
            f"Identified feature types: {len(self.numerical_features)} numerical, "
            f"{len(self.categorical_features)} categorical, {len(self.date_features)} date, "
            f"{len(self.text_features)} text"
        )

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df (pd.DataFrame): The movie data.

        Returns:
            pd.DataFrame: Data with handled missing values.
        """
        logger.info("Handling missing values")

        # Check for missing values
        missing_values = df.isnull().sum()
        missing_cols = missing_values[missing_values > 0].index.tolist()

        if missing_cols:
            logger.info(f"Found missing values in {len(missing_cols)} columns")

            # For numerical features, use KNN imputation
            num_missing = [
                col for col in missing_cols if col in self.numerical_features
            ]
            if num_missing:
                logger.info(
                    f"Imputing missing numerical values with KNN for {len(num_missing)} columns"
                )
                imputer = KNNImputer(n_neighbors=5)
                df[num_missing] = imputer.fit_transform(df[num_missing])

            # For categorical features, fill with mode
            cat_missing = [
                col for col in missing_cols if col in self.categorical_features
            ]
            for col in cat_missing:
                logger.info(f"Filling missing values in {col} with mode")
                df[col] = df[col].fillna(df[col].mode()[0])

            # For date features, fill with median date
            date_missing = [col for col in missing_cols if col in self.date_features]
            for col in date_missing:
                logger.info(f"Filling missing values in {col} with median date")
                df[col] = pd.to_datetime(df[col])
                median_date = df[col].dropna().median()
                df[col] = df[col].fillna(median_date)

            # For text features, fill with empty string
            text_missing = [col for col in missing_cols if col in self.text_features]
            for col in text_missing:
                logger.info(f"Filling missing values in {col} with empty string")
                df[col] = df[col].fillna("")

        return df

    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features for model training.

        Args:
            df (pd.DataFrame): The movie data.

        Returns:
            pd.DataFrame: Data with transformed features.
        """
        logger.info("Transforming features")

        # Process date features
        for col in self.date_features:
            if col in df.columns:
                logger.info(f"Extracting date components from {col}")
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[f"{col}_year"] = df[col].dt.year
                df[f"{col}_month"] = df[col].dt.month
                df[f"{col}_day"] = df[col].dt.day
                df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                df[f"{col}_quarter"] = df[col].dt.quarter
                # Drop original date column
                df = df.drop(columns=[col])

        # Process categorical features
        for col in self.categorical_features:
            if (
                col in df.columns and df[col].nunique() < 10
            ):  # One-hot encode only if few categories
                logger.info(f"One-hot encoding {col}")
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(columns=[col])
            elif col in df.columns:
                logger.info(f"Label encoding {col}")
                df[col] = df[col].astype("category").cat.codes

        # Process text features - simple bag of words for demonstration
        # In a real project, you might use TF-IDF or embeddings
        for col in self.text_features:
            if col in df.columns:
                logger.info(f"Extracting basic text features from {col}")
                df[f"{col}_length"] = df[col].astype(str).apply(len)
                df[f"{col}_word_count"] = (
                    df[col].astype(str).apply(lambda x: len(x.split()))
                )
                # Drop original text column
                df = df.drop(columns=[col])

        # Update numerical features list after transformations
        self.numerical_features = df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        if self.target in self.numerical_features:
            self.numerical_features.remove(self.target)

        return df

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in numerical features.

        Args:
            df (pd.DataFrame): The movie data.

        Returns:
            pd.DataFrame: Data with handled outliers.
        """
        logger.info("Handling outliers")

        for col in self.numerical_features:
            if col in df.columns:
                # Calculate IQR
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                # Define bounds
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Count outliers
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

                if outliers > 0:
                    logger.info(f"Found {outliers} outliers in {col}")

                    # Cap outliers instead of removing
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    logger.info(f"Capped outliers in {col}")

        return df

    def _save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """
        Save the processed data to disk.

        Args:
            X_train (pd.DataFrame): Training features.
            X_test (pd.DataFrame): Test features.
            y_train (pd.Series): Training target.
            y_test (pd.Series): Test target.
        """
        logger.info("Saving processed data")

        # Create processed data directory
        import os

        os.makedirs("data/processed", exist_ok=True)

        # Save data
        X_train.to_csv("data/processed/X_train.csv", index=False)
        X_test.to_csv("data/processed/X_test.csv", index=False)
        y_train.to_csv("data/processed/y_train.csv", index=False)
        y_test.to_csv("data/processed/y_test.csv", index=False)

        logger.info("Processed data saved successfully")
