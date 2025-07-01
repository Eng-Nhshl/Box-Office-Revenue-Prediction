import os
import logging
import argparse
from datetime import datetime

from src.data.data_loader import DataLoader
from src.data.preprocessor import Preprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.visualization.visualize import Visualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        "logs",
        "data/raw",
        "data/processed",
        "models",
        "reports",
        "reports/figures",
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory {dir_path} created or already exists.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Box Office Revenue Prediction Pipeline"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/movies_data.csv",
        help="Path to the raw data file",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ensemble",
        choices=["linear", "tree", "ensemble", "neural_network"],
        help="Type of model to train",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Whether to perform hyperparameter optimization",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2, help="Size of the test set"
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility"
    )
    return parser.parse_args()


def main():
    """Run the complete pipeline."""
    # Create necessary directories
    create_directories()

    # Parse arguments
    args = parse_arguments()

    logger.info("Starting Box Office Revenue Prediction pipeline")

    # Load data
    logger.info("Loading data")
    data_loader = DataLoader(args.data_path)
    data = data_loader.load_data()

    # Preprocess data
    logger.info("Preprocessing data")
    preprocessor = Preprocessor(random_state=args.random_state)
    X_train, X_test, y_train, y_test = preprocessor.process(
        data, test_size=args.test_size
    )

    # Feature engineering
    logger.info("Performing feature engineering")
    feature_engineer = FeatureEngineer()
    X_train = feature_engineer.transform(X_train)
    X_test = feature_engineer.transform(X_test)

    # Train model
    logger.info(f"Training {args.model_type} model")
    model_trainer = ModelTrainer(
        model_type=args.model_type,
        optimize=args.optimize,
        random_state=args.random_state,
    )
    model = model_trainer.train(X_train, y_train)

    # Evaluate model
    logger.info("Evaluating model")
    model_evaluator = ModelEvaluator()
    metrics = model_evaluator.evaluate(model, X_test, y_test)
    feature_importance = model_evaluator.get_feature_importance(model, X_train.columns)

    # Visualize results
    logger.info("Generating visualizations")
    visualizer = Visualizer()
    visualizer.plot_predictions(model, X_test, y_test)
    visualizer.plot_feature_importance(feature_importance)
    visualizer.plot_error_distribution(model, X_test, y_test)

    logger.info("Pipeline completed successfully")


if __name__ == "__main__":
    main()
