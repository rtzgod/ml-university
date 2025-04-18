# preprocessing/vectorizer.py
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureVectorizer:
    """Transforms features into a numerical format suitable for ML models."""

    def __init__(self, numeric_features, categorical_features, num_impute_strategy='median', cat_impute_strategy='constant', cat_fill_value='Missing'):
        """
        Initializes the FeatureVectorizer.

        Args:
            numeric_features (list): List of names of numeric features.
            categorical_features (list): List of names of categorical features.
            num_impute_strategy (str): Imputation strategy for numeric cols.
            cat_impute_strategy (str): Imputation strategy for categorical cols.
            cat_fill_value (str): Fill value for constant categorical imputation.
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.transformer = None
        self.fitted_feature_names = None # To store feature names after transformation

        # Define preprocessing pipelines for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=num_impute_strategy)),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=cat_impute_strategy, fill_value=cat_fill_value)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Use sparse=False for easier handling later
        ])

        # Create the ColumnTransformer
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough' # Keep other columns if any (shouldn't be if lists are correct)
        )
        logging.info("FeatureVectorizer initialized with ColumnTransformer.")

    def fit(self, X):
        """
        Fits the ColumnTransformer to the data.

        Args:
            X (pd.DataFrame): Training data features.
        """
        try:
            # Ensure only columns present in X are used for fitting
            actual_numeric = [col for col in self.numeric_features if col in X.columns]
            actual_categorical = [col for col in self.categorical_features if col in X.columns]

            # Update the transformer list within the preprocessor
            self.preprocessor.transformers_ = [
                ('num', self.preprocessor.transformers[0][1], actual_numeric),
                ('cat', self.preprocessor.transformers[1][1], actual_categorical)
            ]

            self.preprocessor.fit(X)
            # Get feature names after transformation
            self.fitted_feature_names = self.preprocessor.get_feature_names_out()
            logging.info("FeatureVectorizer fitted to the data.")
        except Exception as e:
            logging.error(f"Error fitting FeatureVectorizer: {e}")
            raise

    def transform(self, X):
        """
        Transforms the data using the fitted ColumnTransformer.

        Args:
            X (pd.DataFrame): Data features to transform.

        Returns:
            np.ndarray: Transformed numerical data. Returns None if not fitted.
        """
        if self.preprocessor is None:
            logging.error("Vectorizer has not been fitted yet. Call fit() first.")
            return None
        try:
            X_transformed = self.preprocessor.transform(X)
            logging.info(f"Data transformed. Output shape: {X_transformed.shape}")
            return X_transformed
        except Exception as e:
            logging.error(f"Error transforming data: {e}")
            return None

    def get_feature_names(self):
         """Returns the feature names after transformation."""
         if self.fitted_feature_names is None:
              logging.warning("Vectorizer not fitted or feature names not generated.")
              return None
         return self.fitted_feature_names