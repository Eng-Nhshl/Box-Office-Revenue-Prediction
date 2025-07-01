"""
Utility functions for the Box Office Revenue Prediction project.
"""

import os
import logging
import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
from sklearn.model_selection import learning_curve

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration.

    Args:
        log_dir (str): Directory to save log files.
        log_level (int): Logging level.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename with timestamp
    log_filename = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logger.info(f"Logging configured. Log file: {log_path}")


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data as JSON file.

    Args:
        data (Dict[str, Any]): Data to save.
        filepath (str): Path to save the JSON file.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save data as JSON
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON data: {str(e)}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Loaded data.
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"File {filepath} does not exist")
            return {}

        # Load data from JSON
        with open(filepath, "r") as f:
            data = json.load(f)

        logger.info(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON data: {str(e)}")
        return {}


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data as pickle file.

    Args:
        data (Any): Data to save.
        filepath (str): Path to save the pickle file.
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save data as pickle
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving pickle data: {str(e)}")


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.

    Args:
        filepath (str): Path to the pickle file.

    Returns:
        Any: Loaded data.
    """
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            logger.error(f"File {filepath} does not exist")
            return None

        # Load data from pickle
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        logger.info(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading pickle data: {str(e)}")
        return None


def compute_learning_curves(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    train_sizes: Optional[List[float]] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Compute learning curves for a model.

    Args:
        model (Any): The model to evaluate.
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        cv (int): Number of cross-validation folds.
        train_sizes (List[float], optional): Training set sizes to evaluate.

    Returns:
        Tuple[List[float], List[float], List[float]]: Train sizes, train scores, and test scores.
    """
    try:
        logger.info("Computing learning curves")

        # Set default train sizes if not provided
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        # Compute learning curves
        train_sizes, train_scores, test_scores = learning_curve(
            model,
            X,
            y,
            cv=cv,
            train_sizes=train_sizes,
            scoring="neg_mean_squared_error",
        )

        # Convert negative MSE to positive
        train_scores = -train_scores
        test_scores = -test_scores

        logger.info("Learning curves computed successfully")

        return train_sizes, train_scores, test_scores
    except Exception as e:
        logger.error(f"Error computing learning curves: {str(e)}")
        return [], [], []


def format_runtime(seconds: float) -> str:
    """
    Format runtime in seconds to a human-readable string.

    Args:
        seconds (float): Runtime in seconds.

    Returns:
        str: Formatted runtime string.
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def log_memory_usage() -> None:
    """Log memory usage of the current process."""
    try:
        import psutil
        import os

        # Get current process
        process = psutil.Process(os.getpid())

        # Get memory info
        memory_info = process.memory_info()

        # Log memory usage
        logger.info(f"Memory usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    except ImportError:
        logger.warning("psutil not installed. Cannot log memory usage.")
    except Exception as e:
        logger.error(f"Error logging memory usage: {str(e)}")


def create_experiment_name(model_type: str, timestamp: Optional[str] = None) -> str:
    """
    Create a unique experiment name.

    Args:
        model_type (str): Type of model.
        timestamp (str, optional): Timestamp to use. If None, current time is used.

    Returns:
        str: Experiment name.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{model_type}_{timestamp}"


def log_experiment_params(
    params: Dict[str, Any], experiment_name: str, log_dir: str = "logs"
) -> None:
    """
    Log experiment parameters.

    Args:
        params (Dict[str, Any]): Experiment parameters.
        experiment_name (str): Name of the experiment.
        log_dir (str): Directory to save log files.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename
    log_path = os.path.join(log_dir, f"{experiment_name}_params.json")

    # Save parameters
    save_json(params, log_path)

    logger.info(f"Experiment parameters logged to {log_path}")


def log_experiment_results(
    results: Dict[str, Any], experiment_name: str, log_dir: str = "logs"
) -> None:
    """
    Log experiment results.

    Args:
        results (Dict[str, Any]): Experiment results.
        experiment_name (str): Name of the experiment.
        log_dir (str): Directory to save log files.
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create log filename
    log_path = os.path.join(log_dir, f"{experiment_name}_results.json")

    # Save results
    save_json(results, log_path)

    logger.info(f"Experiment results logged to {log_path}")


def calculate_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate confidence interval for the mean of a dataset.

    Args:
        data (np.ndarray): Data to calculate confidence interval for.
        confidence (float): Confidence level.

    Returns:
        Tuple[float, float]: Lower and upper bounds of the confidence interval.
    """
    try:
        import scipy.stats as stats

        # Calculate mean and standard error
        mean = np.mean(data)
        sem = stats.sem(data)

        # Calculate confidence interval
        interval = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)

        return mean - interval, mean + interval
    except ImportError:
        logger.warning("scipy not installed. Cannot calculate confidence interval.")
        return np.nan, np.nan
    except Exception as e:
        logger.error(f"Error calculating confidence interval: {str(e)}")
        return np.nan, np.nan
