"""
Model training module for the Box Office Revenue Prediction project.
"""

import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Any, Optional, Union, Tuple

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna

logger = logging.getLogger(__name__)


class PyTorchNN(nn.Module):
    """PyTorch Neural Network for regression."""

    def __init__(
        self, input_dim: int, hidden_dims: list = [128, 64], dropout: float = 0.2
    ):
        """
        Initialize the neural network.

        Args:
            input_dim (int): Number of input features.
            hidden_dims (list): List of hidden layer dimensions.
            dropout (float): Dropout rate.
        """
        super(PyTorchNN, self).__init__()

        layers = []
        prev_dim = input_dim

        # Create hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.model(x)


class ModelTrainer:
    """Class to train machine learning models for box office revenue prediction."""

    def __init__(
        self,
        model_type: str = "ensemble",
        optimize: bool = False,
        random_state: int = 42,
    ):
        """
        Initialize the ModelTrainer.

        Args:
            model_type (str): Type of model to train ('linear', 'tree', 'ensemble', or 'neural_network').
            optimize (bool): Whether to perform hyperparameter optimization.
            random_state (int): Random state for reproducibility.
        """
        self.model_type = model_type
        self.optimize = optimize
        self.random_state = random_state
        self.model = None
        self.best_params = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train a model on the given data.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.

        Returns:
            Any: The trained model.
        """
        logger.info(f"Training {self.model_type} model")

        # Select model based on type
        if self.model_type == "linear":
            self.model = self._train_linear_model(X, y)
        elif self.model_type == "tree":
            self.model = self._train_tree_model(X, y)
        elif self.model_type == "ensemble":
            self.model = self._train_ensemble_model(X, y)
        elif self.model_type == "neural_network":
            self.model = self._train_neural_network(X, y)
        else:
            logger.error(f"Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Save the trained model
        self._save_model()

        return self.model

    def _train_linear_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train a linear model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.

        Returns:
            Any: The trained linear model.
        """
        logger.info("Training linear model")

        if self.optimize:
            logger.info("Optimizing hyperparameters for linear model")

            # Define the objective function for Optuna
            def objective(trial):
                # Define hyperparameters to optimize
                alpha = trial.suggest_float("alpha", 1e-5, 100, log=True)
                l1_ratio = trial.suggest_float("l1_ratio", 0, 1)

                # Create and train model
                model = ElasticNet(
                    alpha=alpha, l1_ratio=l1_ratio, random_state=self.random_state
                )
                model.fit(X, y)

                # Calculate negative mean squared error
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)

                return mse

            # Create a study object and optimize
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=50)

            # Get best parameters
            self.best_params = study.best_params
            logger.info(f"Best parameters: {self.best_params}")

            # Train model with best parameters
            model = ElasticNet(
                alpha=self.best_params["alpha"],
                l1_ratio=self.best_params["l1_ratio"],
                random_state=self.random_state,
            )
        else:
            # Use default parameters
            model = ElasticNet(random_state=self.random_state)

        # Train the model
        model.fit(X, y)
        logger.info("Linear model trained successfully")

        return model

    def _train_tree_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train a tree-based model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.

        Returns:
            Any: The trained tree-based model.
        """
        logger.info("Training tree-based model")

        if self.optimize:
            logger.info("Optimizing hyperparameters for tree model")

            # Define the objective function for Optuna
            def objective(trial):
                # Define hyperparameters to optimize
                n_estimators = trial.suggest_int("n_estimators", 50, 500)
                max_depth = trial.suggest_int("max_depth", 3, 20)
                min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
                min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

                # Create and train model
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=self.random_state,
                )
                model.fit(X, y)

                # Calculate negative mean squared error
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)

                return mse

            # Create a study object and optimize
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=50)

            # Get best parameters
            self.best_params = study.best_params
            logger.info(f"Best parameters: {self.best_params}")

            # Train model with best parameters
            model = RandomForestRegressor(
                n_estimators=self.best_params["n_estimators"],
                max_depth=self.best_params["max_depth"],
                min_samples_split=self.best_params["min_samples_split"],
                min_samples_leaf=self.best_params["min_samples_leaf"],
                random_state=self.random_state,
            )
        else:
            # Use default parameters
            model = RandomForestRegressor(random_state=self.random_state)

        # Train the model
        model.fit(X, y)
        logger.info("Tree-based model trained successfully")

        return model

    def _train_ensemble_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train an ensemble model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.

        Returns:
            Any: The trained ensemble model.
        """
        logger.info("Training ensemble model")

        if self.optimize:
            logger.info("Optimizing hyperparameters for ensemble model")

            # Define the objective function for Optuna
            def objective(trial):
                # Choose an algorithm
                algorithm = trial.suggest_categorical(
                    "algorithm", ["xgboost", "lightgbm", "catboost"]
                )

                if algorithm == "xgboost":
                    # Define hyperparameters to optimize for XGBoost
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float(
                            "learning_rate", 0.01, 0.3, log=True
                        ),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float(
                            "colsample_bytree", 0.6, 1.0
                        ),
                        "min_child_weight": trial.suggest_int(
                            "min_child_weight", 1, 10
                        ),
                        "reg_alpha": trial.suggest_float(
                            "reg_alpha", 1e-8, 1.0, log=True
                        ),
                        "reg_lambda": trial.suggest_float(
                            "reg_lambda", 1e-8, 1.0, log=True
                        ),
                        "random_state": self.random_state,
                    }
                    model = xgb.XGBRegressor(**params)

                elif algorithm == "lightgbm":
                    # Define hyperparameters to optimize for LightGBM
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                        "max_depth": trial.suggest_int("max_depth", 3, 10),
                        "learning_rate": trial.suggest_float(
                            "learning_rate", 0.01, 0.3, log=True
                        ),
                        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                        "colsample_bytree": trial.suggest_float(
                            "colsample_bytree", 0.6, 1.0
                        ),
                        "min_child_samples": trial.suggest_int(
                            "min_child_samples", 5, 100
                        ),
                        "reg_alpha": trial.suggest_float(
                            "reg_alpha", 1e-8, 1.0, log=True
                        ),
                        "reg_lambda": trial.suggest_float(
                            "reg_lambda", 1e-8, 1.0, log=True
                        ),
                        "random_state": self.random_state,
                    }
                    model = lgb.LGBMRegressor(**params)

                else:  # catboost
                    # Define hyperparameters to optimize for CatBoost
                    params = {
                        "iterations": trial.suggest_int("iterations", 50, 500),
                        "depth": trial.suggest_int("depth", 3, 10),
                        "learning_rate": trial.suggest_float(
                            "learning_rate", 0.01, 0.3, log=True
                        ),
                        "l2_leaf_reg": trial.suggest_float(
                            "l2_leaf_reg", 1e-8, 10.0, log=True
                        ),
                        "random_strength": trial.suggest_float(
                            "random_strength", 1e-8, 10.0, log=True
                        ),
                        "bagging_temperature": trial.suggest_float(
                            "bagging_temperature", 0, 10.0
                        ),
                        "random_seed": self.random_state,
                    }
                    model = cb.CatBoostRegressor(**params, verbose=0)

                # Train the model
                model.fit(X, y)

                # Calculate negative mean squared error
                y_pred = model.predict(X)
                mse = np.mean((y - y_pred) ** 2)

                return mse

            # Create a study object and optimize
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=50)

            # Get best parameters and algorithm
            self.best_params = study.best_params
            algorithm = self.best_params.pop("algorithm", "xgboost")
            logger.info(f"Best algorithm: {algorithm}")
            logger.info(f"Best parameters: {self.best_params}")

            # Train model with best parameters
            if algorithm == "xgboost":
                model = xgb.XGBRegressor(**self.best_params)
            elif algorithm == "lightgbm":
                model = lgb.LGBMRegressor(**self.best_params)
            else:  # catboost
                model = cb.CatBoostRegressor(**self.best_params, verbose=0)
        else:
            # Use default XGBoost model
            model = xgb.XGBRegressor(random_state=self.random_state)

        # Train the model
        model.fit(X, y)
        logger.info("Ensemble model trained successfully")

        return model

    def _train_neural_network(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """
        Train a neural network model.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Training target.

        Returns:
            Any: The trained neural network model.
        """
        logger.info("Training neural network model")

        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X.values)
        y_tensor = torch.FloatTensor(y.values).view(-1, 1)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        if self.optimize:
            logger.info("Optimizing hyperparameters for neural network")

            # Define the objective function for Optuna
            def objective(trial):
                # Define hyperparameters to optimize
                n_layers = trial.suggest_int("n_layers", 1, 3)
                hidden_dims = [
                    trial.suggest_int(f"hidden_dim_{i}", 32, 256)
                    for i in range(n_layers)
                ]
                dropout = trial.suggest_float("dropout", 0.1, 0.5)
                learning_rate = trial.suggest_float(
                    "learning_rate", 1e-4, 1e-2, log=True
                )
                weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

                # Create model
                input_dim = X.shape[1]
                model = PyTorchNN(
                    input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
                )

                # Define loss function and optimizer
                criterion = nn.MSELoss()
                optimizer = optim.Adam(
                    model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )

                # Train the model
                model.train()
                for epoch in range(50):  # Train for 50 epochs
                    epoch_loss = 0
                    for batch_X, batch_y in dataloader:
                        # Forward pass
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                        # Backward pass and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()

                    # Early stopping
                    if epoch_loss / len(dataloader) < 0.001:
                        break

                # Evaluate the model
                model.eval()
                with torch.no_grad():
                    y_pred = model(X_tensor)
                    mse = criterion(y_pred, y_tensor).item()

                return mse

            # Create a study object and optimize
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=30)

            # Get best parameters
            self.best_params = study.best_params
            logger.info(f"Best parameters: {self.best_params}")

            # Extract parameters
            n_layers = self.best_params["n_layers"]
            hidden_dims = [self.best_params[f"hidden_dim_{i}"] for i in range(n_layers)]
            dropout = self.best_params["dropout"]
            learning_rate = self.best_params["learning_rate"]
            weight_decay = self.best_params["weight_decay"]

            # Create and train model with best parameters
            input_dim = X.shape[1]
            model = PyTorchNN(
                input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout
            )
            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:
            # Use default parameters
            input_dim = X.shape[1]
            model = PyTorchNN(input_dim=input_dim)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Train the model
        model.train()
        for epoch in range(100):  # Train for 100 epochs
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/100, Loss: {epoch_loss/len(dataloader):.4f}"
                )

            # Early stopping
            if epoch_loss / len(dataloader) < 0.001:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        logger.info("Neural network model trained successfully")

        return model

    def _save_model(self) -> None:
        """Save the trained model to disk."""
        logger.info("Saving model")

        # Create models directory
        os.makedirs("models", exist_ok=True)

        # Save model
        if self.model_type == "neural_network":
            # Save PyTorch model
            model_path = f"models/{self.model_type}_model.pt"
            torch.save(self.model.state_dict(), model_path)
        else:
            # Save scikit-learn or other model
            model_path = f"models/{self.model_type}_model.joblib"
            joblib.dump(self.model, model_path)

        # Save best parameters if available
        if self.best_params:
            params_path = f"models/{self.model_type}_params.joblib"
            joblib.dump(self.best_params, params_path)

        logger.info(f"Model saved to {model_path}")
