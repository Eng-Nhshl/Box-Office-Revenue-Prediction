"""
Feature engineering module for the Box Office Revenue Prediction project.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Class to perform feature engineering on movie data."""

    def __init__(
        self,
        create_interactions: bool = True,
        use_pca: bool = False,
        feature_selection: bool = True,
        n_components: int = 10,
        k_best: int = 20,
    ):
        """
        Initialize the FeatureEngineer.

        Args:
            create_interactions (bool): Whether to create interaction features.
            use_pca (bool): Whether to use PCA for dimensionality reduction.
            feature_selection (bool): Whether to use feature selection.
            n_components (int): Number of PCA components.
            k_best (int): Number of best features to select.
        """
        self.create_interactions = create_interactions
        self.use_pca = use_pca
        self.feature_selection = feature_selection
        self.n_components = n_components
        self.k_best = k_best
        self.poly = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )
        self.pca = PCA(n_components=n_components)
        self.selector = SelectKBest(f_regression, k=k_best)
        self.interaction_features = []
        self.selected_features = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """
        Fit the feature engineering transformations.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.

        Returns:
            FeatureEngineer: The fitted instance.
        """
        logger.info("Fitting feature engineering transformations")

        # Store original feature names
        self.original_features = X.columns.tolist()

        # Create interaction features
        if self.create_interactions:
            logger.info("Creating interaction features")
            # Select only numerical columns for interactions
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns

            # If there are too many numerical columns, select only the most important ones
            if len(num_cols) > 10:
                logger.info(
                    "Too many numerical columns, selecting top 10 for interactions"
                )
                if y is not None:
                    # Use correlation with target to select top features
                    correlations = (
                        X[num_cols].corrwith(y).abs().sort_values(ascending=False)
                    )
                    num_cols = correlations.head(10).index.tolist()
                else:
                    # If no target is provided, just take the first 10
                    num_cols = num_cols[:10]

            # Fit polynomial features
            self.poly.fit(X[num_cols])

            # Get interaction feature names
            interaction_names = []
            for i, name1 in enumerate(num_cols):
                for name2 in num_cols[i + 1 :]:
                    interaction_names.append(f"{name1}_{name2}")

            self.interaction_features = interaction_names

        # Apply PCA
        if self.use_pca:
            logger.info(f"Applying PCA with {self.n_components} components")
            # Select only numerical columns for PCA
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns
            self.pca.fit(X[num_cols])
            logger.info(
                f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}"
            )

        # Apply feature selection
        if self.feature_selection and y is not None:
            logger.info(
                f"Applying feature selection to select top {self.k_best} features"
            )
            self.selector.fit(X, y)

            # Get selected feature names
            mask = self.selector.get_support()
            self.selected_features = X.columns[mask].tolist()
            logger.info(f"Selected features: {self.selected_features}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the feature engineering transformations.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.DataFrame: The transformed features.
        """
        logger.info("Applying feature engineering transformations")

        # Make a copy to avoid modifying the original data
        X_transformed = X.copy()

        # Create domain-specific features
        X_transformed = self._create_domain_features(X_transformed)

        # Create interaction features
        if self.create_interactions:
            logger.info("Adding interaction features")
            # Select only numerical columns for interactions
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns

            # If there are too many numerical columns, select only the ones used during fit
            if len(num_cols) > 10:
                common_cols = [col for col in self.original_features if col in num_cols]
                num_cols = common_cols[:10]

            # Transform data
            if len(num_cols) >= 2:  # Need at least 2 columns for interactions
                # Check if poly is fitted, if not, fit it first
                if not self.is_fitted:
                    logger.info("Fitting polynomial features transformer")
                    self.poly.fit(X[num_cols])

                interactions = self.poly.transform(X[num_cols])

                # Extract only the interaction terms (not the original features or squared terms)
                n_original = len(num_cols)
                interaction_terms = interactions[:, n_original:]

                # Create a DataFrame with interaction features
                if not self.is_fitted:
                    # Generate interaction feature names if not fitted yet
                    interaction_names = []
                    for i, name1 in enumerate(num_cols):
                        for name2 in num_cols[i + 1 :]:
                            interaction_names.append(f"{name1}_{name2}")
                    self.interaction_features = interaction_names

                # Add interaction features to the transformed data
                interaction_df = pd.DataFrame(
                    interaction_terms,
                    columns=self.interaction_features[: interaction_terms.shape[1]],
                    index=X.index,
                )
                X_transformed = pd.concat([X_transformed, interaction_df], axis=1)

        # Apply PCA
        if self.use_pca:
            logger.info("Adding PCA components")
            # Select only numerical columns for PCA
            num_cols = X.select_dtypes(include=["int64", "float64"]).columns

            # Check if PCA is fitted, if not, fit it first
            if not self.is_fitted:
                logger.info("Fitting PCA transformer")
                self.pca.fit(X[num_cols])

            pca_components = self.pca.transform(X[num_cols])

            # Create a DataFrame with PCA components
            pca_df = pd.DataFrame(
                pca_components,
                columns=[f"pca_{i+1}" for i in range(self.n_components)],
                index=X.index,
            )
            X_transformed = pd.concat([X_transformed, pca_df], axis=1)

        # Apply feature selection
        if self.feature_selection and self.is_fitted:
            logger.info("Selecting best features")
            if self.selected_features:
                # Keep only selected features
                common_features = [
                    col
                    for col in self.selected_features
                    if col in X_transformed.columns
                ]
                if common_features:
                    X_transformed = X_transformed[common_features]
                    logger.info(f"Selected {len(common_features)} features")
                else:
                    logger.warning(
                        "No common features found after selection. Keeping all features."
                    )

        return X_transformed

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Fit and apply the feature engineering transformations.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.Series, optional): The target variable.

        Returns:
            pd.DataFrame: The transformed features.
        """
        return self.fit(X, y).transform(X)

    def _create_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for movie revenue prediction.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            pd.DataFrame: The data with additional domain-specific features.
        """
        logger.info("Creating domain-specific features")

        # Make a copy to avoid modifying the original data
        X_new = X.copy()

        # Create budget-related features
        if "budget" in X_new.columns:
            logger.info("Creating budget-related features")

            # Log transform of budget (common in revenue prediction)
            X_new["log_budget"] = np.log1p(X_new["budget"])

            # Budget per runtime minute (production efficiency)
            if "runtime" in X_new.columns:
                X_new["budget_per_minute"] = X_new["budget"] / X_new["runtime"].replace(
                    0, 1
                )

        # Create popularity-related features
        if "popularity" in X_new.columns:
            logger.info("Creating popularity-related features")

            # Log transform of popularity
            X_new["log_popularity"] = np.log1p(X_new["popularity"])

            # Popularity to vote count ratio (engagement metric)
            if "vote_count" in X_new.columns:
                X_new["popularity_to_votes"] = X_new["popularity"] / X_new[
                    "vote_count"
                ].replace(0, 1)

        # Create vote-related features
        if "vote_average" in X_new.columns and "vote_count" in X_new.columns:
            logger.info("Creating vote-related features")

            # Weighted rating (IMDB formula)
            # WR = (v/(v+m)) * R + (m/(v+m)) * C
            # where:
            # R = average rating for the movie
            # v = number of votes for the movie
            # m = minimum votes required (using median)
            # C = mean vote across the whole dataset
            m = X_new["vote_count"].median()
            C = X_new["vote_average"].mean()
            X_new["weighted_rating"] = (
                X_new["vote_count"] / (X_new["vote_count"] + m) * X_new["vote_average"]
            ) + (m / (X_new["vote_count"] + m) * C)

        # Create release date-related features
        release_date_cols = [col for col in X_new.columns if "release_date" in col]
        if release_date_cols:
            logger.info("Creating release date-related features")

            # Season features (summer blockbusters, holiday releases)
            if "release_date_month" in X_new.columns:
                # Summer months (May-August)
                X_new["is_summer_release"] = (
                    X_new["release_date_month"].isin([5, 6, 7, 8]).astype(int)
                )

                # Holiday months (November-December)
                X_new["is_holiday_release"] = (
                    X_new["release_date_month"].isin([11, 12]).astype(int)
                )

                # Weekend release (Friday-Sunday)
                if "release_date_dayofweek" in X_new.columns:
                    X_new["is_weekend_release"] = (
                        X_new["release_date_dayofweek"].isin([4, 5, 6]).astype(int)
                    )

        # Create genre-related features
        genre_cols = [col for col in X_new.columns if "genre" in col.lower()]
        if genre_cols:
            logger.info("Creating genre-related features")

            # Count number of genres (movies with multiple genres might perform differently)
            for col in genre_cols:
                if X_new[col].dtype == "object":
                    X_new[f"{col}_count"] = X_new[col].str.count("\|") + 1

        return X_new
