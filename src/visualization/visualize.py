"""
Visualization module for the Box Office Revenue Prediction project.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class Visualizer:
    """Class to create visualizations for the Box Office Revenue Prediction project."""

    def __init__(self, save_dir: str = "reports/figures"):
        """
        Initialize the Visualizer.

        Args:
            save_dir (str): Directory to save visualizations.
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_predictions(self, model: Any, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Plot predicted vs. actual values.

        Args:
            model (Any): The trained model.
            X (pd.DataFrame): Test features.
            y (pd.Series): Test target.
        """
        logger.info("Plotting predicted vs. actual values")

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

            # Create figure
            plt.figure(figsize=(10, 8))

            # Plot predicted vs. actual values
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
            plt.xlabel("Actual Revenue")
            plt.ylabel("Predicted Revenue")
            plt.title("Predicted vs. Actual Revenue")

            # Add metrics to the plot
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            plt.annotate(
                f"RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.2f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            plt.tight_layout()

            # Save figure
            plt.savefig(os.path.join(self.save_dir, "predictions.png"))
            plt.close()

            # Create interactive Plotly version
            fig = px.scatter(
                x=y,
                y=y_pred,
                opacity=0.6,
                labels={"x": "Actual Revenue", "y": "Predicted Revenue"},
                title="Predicted vs. Actual Revenue",
            )

            # Add identity line
            fig.add_trace(
                go.Scatter(
                    x=[y.min(), y.max()],
                    y=[y.min(), y.max()],
                    mode="lines",
                    name="Identity Line",
                    line=dict(color="red", dash="dash"),
                )
            )

            # Add metrics annotation
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"RMSE: {rmse:.2f}<br>MAE: {mae:.2f}<br>R²: {r2:.2f}",
                showarrow=False,
                font=dict(size=14),
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
            )

            # Save interactive figure
            fig.write_html(os.path.join(self.save_dir, "predictions_interactive.html"))

            logger.info(
                "Predictions plot saved to reports/figures/predictions.png and predictions_interactive.html"
            )

        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")

    def plot_feature_importance(
        self, feature_importance: pd.DataFrame, top_n: int = 20
    ) -> None:
        """
        Plot feature importance.

        Args:
            feature_importance (pd.DataFrame): DataFrame with feature importance.
            top_n (int): Number of top features to plot.
        """
        logger.info(f"Plotting top {top_n} feature importance")

        try:
            # Create figure
            plt.figure(figsize=(12, 10))

            # Get top N features
            n_features = min(top_n, len(feature_importance))
            top_features = feature_importance.head(n_features).copy()

            # Sort for better visualization
            top_features = top_features.sort_values("Importance")

            # Create horizontal bar plot
            sns.barplot(x="Importance", y="Feature", data=top_features)
            plt.title(f"Top {n_features} Feature Importance")
            plt.tight_layout()

            # Save figure
            plt.savefig(os.path.join(self.save_dir, "feature_importance.png"))
            plt.close()

            # Create interactive Plotly version
            fig = px.bar(
                top_features,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Top {n_features} Feature Importance",
            )

            # Save interactive figure
            fig.write_html(
                os.path.join(self.save_dir, "feature_importance_interactive.html")
            )

            logger.info(
                "Feature importance plot saved to reports/figures/feature_importance.png and feature_importance_interactive.html"
            )

        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")

    def plot_error_distribution(
        self, model: Any, X: pd.DataFrame, y: pd.Series
    ) -> None:
        """
        Plot error distribution.

        Args:
            model (Any): The trained model.
            X (pd.DataFrame): Test features.
            y (pd.Series): Test target.
        """
        logger.info("Plotting error distribution")

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

            # Calculate errors
            errors = y - y_pred
            abs_errors = np.abs(errors)

            # Create figure
            plt.figure(figsize=(15, 10))

            # Plot error distribution
            plt.subplot(2, 2, 1)
            sns.histplot(errors, kde=True)
            plt.xlabel("Error")
            plt.title("Error Distribution")

            # Plot absolute error distribution
            plt.subplot(2, 2, 2)
            sns.histplot(abs_errors, kde=True)
            plt.xlabel("Absolute Error")
            plt.title("Absolute Error Distribution")

            # Plot error vs. predicted values
            plt.subplot(2, 2, 3)
            plt.scatter(y_pred, errors, alpha=0.5)
            plt.axhline(y=0, color="r", linestyle="-")
            plt.xlabel("Predicted Revenue")
            plt.ylabel("Error")
            plt.title("Error vs. Predicted Revenue")

            # Plot error vs. actual values
            plt.subplot(2, 2, 4)
            plt.scatter(y, errors, alpha=0.5)
            plt.axhline(y=0, color="r", linestyle="-")
            plt.xlabel("Actual Revenue")
            plt.ylabel("Error")
            plt.title("Error vs. Actual Revenue")

            plt.tight_layout()

            # Save figure
            plt.savefig(os.path.join(self.save_dir, "error_distribution.png"))
            plt.close()

            # Create interactive Plotly version for error distribution
            fig = px.histogram(
                errors,
                nbins=50,
                marginal="box",
                title="Error Distribution",
                labels={"value": "Error"},
            )

            # Save interactive figure
            fig.write_html(
                os.path.join(self.save_dir, "error_distribution_interactive.html")
            )

            logger.info(
                "Error distribution plot saved to reports/figures/error_distribution.png and error_distribution_interactive.html"
            )

        except Exception as e:
            logger.error(f"Error plotting error distribution: {str(e)}")

    def plot_correlation_matrix(
        self, data: pd.DataFrame, target_col: str = "revenue"
    ) -> None:
        """
        Plot correlation matrix.

        Args:
            data (pd.DataFrame): The dataset.
            target_col (str): The target column.
        """
        logger.info("Plotting correlation matrix")

        try:
            # Select only numerical columns
            num_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

            # If there are too many columns, select only the most correlated with the target
            if len(num_cols) > 20:
                logger.info(
                    "Too many numerical columns, selecting top 20 most correlated with target"
                )

                # Calculate correlation with target
                if target_col in num_cols:
                    target_corr = (
                        data[num_cols]
                        .corrwith(data[target_col])
                        .abs()
                        .sort_values(ascending=False)
                    )
                    # Select top 20 columns (including target)
                    selected_cols = target_corr.head(20).index.tolist()

                    # Make sure target is included
                    if target_col not in selected_cols:
                        selected_cols[-1] = target_col
                else:
                    # If target is not in numerical columns, just take the first 20
                    selected_cols = num_cols[:19] + [target_col]

                # Filter data
                corr_data = data[selected_cols]
            else:
                # Use all numerical columns
                corr_data = data[num_cols]

            # Calculate correlation matrix
            corr_matrix = corr_data.corr()

            # Create figure
            plt.figure(figsize=(14, 12))

            # Plot heatmap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                linewidths=0.5,
                vmin=-1,
                vmax=1,
            )
            plt.title("Correlation Matrix")
            plt.tight_layout()

            # Save figure
            plt.savefig(os.path.join(self.save_dir, "correlation_matrix.png"))
            plt.close()

            # Create interactive Plotly version
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale="RdBu_r",
                range_color=[-1, 1],
            )

            # Save interactive figure
            fig.write_html(
                os.path.join(self.save_dir, "correlation_matrix_interactive.html")
            )

            logger.info(
                "Correlation matrix plot saved to reports/figures/correlation_matrix.png and correlation_matrix_interactive.html"
            )

        except Exception as e:
            logger.error(f"Error plotting correlation matrix: {str(e)}")

    def plot_feature_distributions(
        self, data: pd.DataFrame, target_col: str = "revenue", top_n: int = 5
    ) -> None:
        """
        Plot distributions of top features.

        Args:
            data (pd.DataFrame): The dataset.
            target_col (str): The target column.
            top_n (int): Number of top features to plot.
        """
        logger.info(f"Plotting distributions of top {top_n} features")

        try:
            # Select only numerical columns
            num_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()

            # Remove target from features
            if target_col in num_cols:
                num_cols.remove(target_col)

            # Calculate correlation with target
            target_corr = (
                data[num_cols]
                .corrwith(data[target_col])
                .abs()
                .sort_values(ascending=False)
            )

            # Select top N features
            top_features = target_corr.head(top_n).index.tolist()

            # Create figure
            plt.figure(figsize=(15, 4 * top_n))

            # Plot distributions
            for i, feature in enumerate(top_features):
                plt.subplot(top_n, 2, 2 * i + 1)
                sns.histplot(data[feature], kde=True)
                plt.title(f"Distribution of {feature}")

                plt.subplot(top_n, 2, 2 * i + 2)
                sns.scatterplot(x=feature, y=target_col, data=data, alpha=0.5)
                plt.title(f"{feature} vs. {target_col}")

            plt.tight_layout()

            # Save figure
            plt.savefig(os.path.join(self.save_dir, "feature_distributions.png"))
            plt.close()

            # Create interactive Plotly version for each feature
            for feature in top_features:
                # Distribution
                fig_dist = px.histogram(
                    data, x=feature, marginal="box", title=f"Distribution of {feature}"
                )
                fig_dist.write_html(
                    os.path.join(self.save_dir, f"distribution_{feature}.html")
                )

                # Scatter plot
                fig_scatter = px.scatter(
                    data,
                    x=feature,
                    y=target_col,
                    opacity=0.6,
                    title=f"{feature} vs. {target_col}",
                    trendline="ols",
                )
                fig_scatter.write_html(
                    os.path.join(self.save_dir, f"scatter_{feature}_{target_col}.html")
                )

            logger.info(
                f"Feature distributions plot saved to reports/figures/feature_distributions.png and individual HTML files"
            )

        except Exception as e:
            logger.error(f"Error plotting feature distributions: {str(e)}")

    def create_eda_report(
        self, data: pd.DataFrame, target_col: str = "revenue"
    ) -> None:
        """
        Create a comprehensive EDA report.

        Args:
            data (pd.DataFrame): The dataset.
            target_col (str): The target column.
        """
        logger.info("Creating EDA report")

        try:
            # Plot correlation matrix
            self.plot_correlation_matrix(data, target_col)

            # Plot feature distributions
            self.plot_feature_distributions(data, target_col)

            # Plot target distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(data[target_col], kde=True)
            plt.title(f"Distribution of {target_col}")
            plt.savefig(os.path.join(self.save_dir, "target_distribution.png"))
            plt.close()

            # Create interactive Plotly version for target distribution
            fig = px.histogram(
                data,
                x=target_col,
                marginal="box",
                title=f"Distribution of {target_col}",
            )
            fig.write_html(
                os.path.join(self.save_dir, "target_distribution_interactive.html")
            )

            # Plot log-transformed target distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(np.log1p(data[target_col]), kde=True)
            plt.title(f"Distribution of Log({target_col})")
            plt.savefig(os.path.join(self.save_dir, "log_target_distribution.png"))
            plt.close()

            # Create interactive Plotly version for log-transformed target distribution
            fig = px.histogram(
                np.log1p(data[target_col]),
                marginal="box",
                title=f"Distribution of Log({target_col})",
            )
            fig.write_html(
                os.path.join(self.save_dir, "log_target_distribution_interactive.html")
            )

            logger.info("EDA report created successfully")

        except Exception as e:
            logger.error(f"Error creating EDA report: {str(e)}")

    def plot_learning_curves(
        self,
        train_sizes: List[float],
        train_scores: List[float],
        test_scores: List[float],
    ) -> None:
        """
        Plot learning curves.

        Args:
            train_sizes (List[float]): Training set sizes.
            train_scores (List[float]): Training scores.
            test_scores (List[float]): Test scores.
        """
        logger.info("Plotting learning curves")

        try:
            # Create figure
            plt.figure(figsize=(10, 6))

            # Calculate mean and std for train scores
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)

            # Calculate mean and std for test scores
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)

            # Plot learning curves
            plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
            plt.plot(
                train_sizes, test_mean, "o-", color="g", label="Cross-validation score"
            )

            # Plot standard deviation bands
            plt.fill_between(
                train_sizes,
                train_mean - train_std,
                train_mean + train_std,
                alpha=0.1,
                color="r",
            )
            plt.fill_between(
                train_sizes,
                test_mean - test_std,
                test_mean + test_std,
                alpha=0.1,
                color="g",
            )

            # Add labels and title
            plt.xlabel("Training Set Size")
            plt.ylabel("Score")
            plt.title("Learning Curves")
            plt.legend(loc="best")
            plt.grid(True)

            # Save figure
            plt.savefig(os.path.join(self.save_dir, "learning_curves.png"))
            plt.close()

            # Create interactive Plotly version
            fig = go.Figure()

            # Add training score
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=train_mean,
                    mode="lines+markers",
                    name="Training Score",
                    line=dict(color="red"),
                    error_y=dict(type="data", array=train_std, visible=True),
                )
            )

            # Add cross-validation score
            fig.add_trace(
                go.Scatter(
                    x=train_sizes,
                    y=test_mean,
                    mode="lines+markers",
                    name="Cross-validation Score",
                    line=dict(color="green"),
                    error_y=dict(type="data", array=test_std, visible=True),
                )
            )

            # Update layout
            fig.update_layout(
                title="Learning Curves",
                xaxis_title="Training Set Size",
                yaxis_title="Score",
                legend=dict(x=0.01, y=0.99),
                template="plotly_white",
            )

            # Save interactive figure
            fig.write_html(
                os.path.join(self.save_dir, "learning_curves_interactive.html")
            )

            logger.info(
                "Learning curves plot saved to reports/figures/learning_curves.png and learning_curves_interactive.html"
            )

        except Exception as e:
            logger.error(f"Error plotting learning curves: {str(e)}")
