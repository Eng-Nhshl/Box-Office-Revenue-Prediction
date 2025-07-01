"""
Model evaluation module for the Box Office Revenue Prediction project.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Class to evaluate machine learning models for box office revenue prediction."""

    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.metrics = {}
        self.feature_importance = None
        self.shap_values = None

    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate a model on the given data.

        Args:
            model (Any): The trained model.
            X (pd.DataFrame): Test features.
            y (pd.Series): Test target.

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        logger.info("Evaluating model")

        # Make predictions
        try:
            if hasattr(model, "predict"):
                y_pred = model.predict(X)
            else:
                # For PyTorch models
                import torch

                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X.values)
                    y_pred = model(X_tensor).numpy().flatten()
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y - y_pred) / y)) * 100

        # Store metrics
        self.metrics = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

        # Log metrics
        logger.info(f"Model evaluation metrics:")
        for metric, value in self.metrics.items():
            logger.info(f"{metric}: {value:.4f}")

        # Save metrics
        self._save_metrics()

        return self.metrics

    def get_feature_importance(
        self, model: Any, feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Get feature importance from the model.

        Args:
            model (Any): The trained model.
            feature_names (List[str]): List of feature names.

        Returns:
            pd.DataFrame: DataFrame with feature importance.
        """
        logger.info("Getting feature importance")

        try:
            # Different models have different ways to get feature importance
            if hasattr(model, "feature_importances_"):
                # For tree-based models
                importance = model.feature_importances_

            elif hasattr(model, "coef_"):
                # For linear models
                importance = np.abs(model.coef_)
                if importance.ndim > 1:
                    importance = importance.mean(axis=0)

            elif hasattr(model, "get_booster"):
                # For XGBoost
                importance = model.get_booster().get_score(importance_type="gain")
                # Convert to array with proper ordering
                imp_array = np.zeros(len(feature_names))
                for key, value in importance.items():
                    try:
                        idx = int(key.replace("f", ""))
                        if idx < len(imp_array):
                            imp_array[idx] = value
                    except:
                        continue
                importance = imp_array

            elif hasattr(model, "feature_importance"):
                # For LightGBM
                importance = model.feature_importance()

            elif hasattr(model, "get_feature_importance"):
                # For CatBoost
                importance = model.get_feature_importance()

            else:
                # For models without built-in feature importance
                logger.warning(
                    "Model doesn't have built-in feature importance. Using SHAP values."
                )
                return self._get_shap_importance(model, feature_names)

            # Create DataFrame
            if len(importance) != len(feature_names):
                logger.warning(
                    f"Feature importance length ({len(importance)}) doesn't match feature names length ({len(feature_names)}). Using indices as names."
                )
                feature_names = [f"Feature_{i}" for i in range(len(importance))]

            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importance}
            )

            # Sort by importance
            importance_df = importance_df.sort_values(
                "Importance", ascending=False
            ).reset_index(drop=True)

            # Store feature importance
            self.feature_importance = importance_df

            # Save feature importance
            self._save_feature_importance()

            return importance_df

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            logger.warning("Falling back to SHAP values for feature importance.")
            return self._get_shap_importance(model, feature_names)

    def _get_shap_importance(
        self, model: Any, feature_names: List[str], sample_size: int = 100
    ) -> pd.DataFrame:
        """
        Get feature importance using SHAP values.

        Args:
            model (Any): The trained model.
            feature_names (List[str]): List of feature names.
            sample_size (int): Number of samples to use for SHAP calculation.

        Returns:
            pd.DataFrame: DataFrame with feature importance based on SHAP values.
        """
        logger.info("Calculating SHAP values for feature importance")

        try:
            # Create a SHAP explainer
            if hasattr(model, "predict"):
                # For most models
                explainer = shap.Explainer(model)
            else:
                # For PyTorch models
                def predict_fn(X):
                    import torch

                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        return model(X_tensor).numpy()

                explainer = shap.Explainer(predict_fn)

            # Sample data for SHAP values calculation
            X_sample = pd.DataFrame(
                np.random.randn(sample_size, len(feature_names)), columns=feature_names
            )

            # Calculate SHAP values
            shap_values = explainer(X_sample)

            # Store SHAP values
            self.shap_values = shap_values

            # Get mean absolute SHAP values for each feature
            importance = np.abs(shap_values.values).mean(axis=0)

            # Create DataFrame
            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": importance}
            )

            # Sort by importance
            importance_df = importance_df.sort_values(
                "Importance", ascending=False
            ).reset_index(drop=True)

            # Store feature importance
            self.feature_importance = importance_df

            # Save feature importance
            self._save_feature_importance()

            return importance_df

        except Exception as e:
            logger.error(f"Error calculating SHAP values: {str(e)}")

            # Create a dummy importance DataFrame
            importance_df = pd.DataFrame(
                {"Feature": feature_names, "Importance": np.ones(len(feature_names))}
            )

            return importance_df

    def plot_residuals(self, model: Any, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Plot residuals to analyze model errors.

        Args:
            model (Any): The trained model.
            X (pd.DataFrame): Test features.
            y (pd.Series): Test target.
        """
        logger.info("Plotting residuals")

        try:
            # Make predictions
            if hasattr(model, "predict"):
                y_pred = model.predict(X)
            else:
                # For PyTorch models
                import torch

                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X.values)
                    y_pred = model(X_tensor).numpy().flatten()

            # Calculate residuals
            residuals = y - y_pred

            # Create figure
            plt.figure(figsize=(12, 10))

            # Plot residuals vs. predicted values
            plt.subplot(2, 2, 1)
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color="r", linestyle="-")
            plt.xlabel("Predicted Values")
            plt.ylabel("Residuals")
            plt.title("Residuals vs. Predicted Values")

            # Plot residuals distribution
            plt.subplot(2, 2, 2)
            sns.histplot(residuals, kde=True)
            plt.xlabel("Residuals")
            plt.title("Residuals Distribution")

            # Plot predicted vs. actual values
            plt.subplot(2, 2, 3)
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
            plt.xlabel("Actual Values")
            plt.ylabel("Predicted Values")
            plt.title("Predicted vs. Actual Values")

            # Plot residuals Q-Q plot
            plt.subplot(2, 2, 4)
            import scipy.stats as stats

            stats.probplot(residuals, plot=plt)
            plt.title("Residuals Q-Q Plot")

            plt.tight_layout()

            # Save figure
            os.makedirs("reports/figures", exist_ok=True)
            plt.savefig("reports/figures/residuals_analysis.png")
            plt.close()

            logger.info(
                "Residuals plot saved to reports/figures/residuals_analysis.png"
            )

        except Exception as e:
            logger.error(f"Error plotting residuals: {str(e)}")

    def _save_metrics(self) -> None:
        """Save evaluation metrics to disk."""
        logger.info("Saving evaluation metrics")

        # Create reports directory
        os.makedirs("reports", exist_ok=True)

        # Save metrics
        metrics_df = pd.DataFrame([self.metrics])
        metrics_df.to_csv("reports/evaluation_metrics.csv", index=False)

        logger.info("Evaluation metrics saved to reports/evaluation_metrics.csv")

    def _save_feature_importance(self) -> None:
        """Save feature importance to disk."""
        if self.feature_importance is not None:
            logger.info("Saving feature importance")

            # Create reports directory
            os.makedirs("reports", exist_ok=True)

            # Save feature importance
            self.feature_importance.to_csv(
                "reports/feature_importance.csv", index=False
            )

            logger.info("Feature importance saved to reports/feature_importance.csv")

            # Create a bar plot of feature importance
            plt.figure(figsize=(12, 8))

            # Plot top 20 features or all if less than 20
            n_features = min(20, len(self.feature_importance))
            top_features = self.feature_importance.head(n_features)

            sns.barplot(x="Importance", y="Feature", data=top_features)
            plt.title(f"Top {n_features} Feature Importance")
            plt.tight_layout()

            # Save figure
            os.makedirs("reports/figures", exist_ok=True)
            plt.savefig("reports/figures/feature_importance.png")
            plt.close()

            logger.info(
                "Feature importance plot saved to reports/figures/feature_importance.png"
            )
