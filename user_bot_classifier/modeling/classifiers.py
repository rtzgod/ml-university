# modeling/classifiers.py
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline # Import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelTrainer:
    """Trains specified classification models."""

    def __init__(self, models_to_train=['logistic_regression', 'random_forest', 'gradient_boosting'], random_state=42):
        """
        Initializes the ModelTrainer.

        Args:
            models_to_train (list): List of model names to train.
                                    Options: 'logistic_regression', 'random_forest', 'gradient_boosting'.
            random_state (int): Random state for reproducibility.
        """
        self.models_to_train = models_to_train
        self.random_state = random_state
        self.pipelines = {} # Store pipelines instead of just models
        logging.info(f"ModelTrainer initialized for models: {self.models_to_train}")

    def _get_model(self, model_name):
        """Helper function to instantiate a model."""
        if model_name == 'logistic_regression':
            # Increase max_iter for convergence with scaled data
            return LogisticRegression(random_state=self.random_state, max_iter=1000, class_weight='balanced')
        elif model_name == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, class_weight='balanced')
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=self.random_state)
        else:
            logging.warning(f"Model '{model_name}' not recognized. Skipping.")
            return None

    def train(self, preprocessor, X_train, y_train):
        """
        Trains the specified models using pipelines that include the preprocessor.

        Args:
            preprocessor (ColumnTransformer): The fitted preprocessor/vectorizer.
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target labels.

        Returns:
            dict: Dictionary containing trained model pipelines {model_name: pipeline}.
        """
        self.pipelines = {}
        for model_name in self.models_to_train:
            model = self._get_model(model_name)
            # *** CHANGE THIS LINE ***
            # Explicitly check if the model object exists (is not None)
            if model is not None:
            # *** END CHANGE ***
                # Create a pipeline with preprocessing and the classifier
                pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier', model)])
                try:
                    logging.info(f"Training {model_name}...")
                    pipeline.fit(X_train, y_train)
                    self.pipelines[model_name] = pipeline
                    logging.info(f"{model_name} trained successfully.")
                except Exception as e:
                    logging.error(f"Error training {model_name}: {e}")
        return self.pipelines

    def get_trained_pipelines(self):
        """Returns the dictionary of trained pipelines."""
        return self.pipelines