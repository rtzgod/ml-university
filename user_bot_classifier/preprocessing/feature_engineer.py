# preprocessing/feature_engineer.py
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    """Creates new features from existing ones."""

    def __init__(self):
        """Initializes the FeatureEngineer."""
        logging.info("FeatureEngineer initialized.")
        self.profile_bool_cols = [ # Example list, adjust based on actual boolean cols
            'has_domain', 'has_birth_date', 'has_photo', 'can_post_on_wall',
            'can_send_message', 'has_website', 'has_short_name', 'has_first_name',
            'has_last_name', 'access_to_closed_profile', 'is_profile_closed',
            'has_nickname', 'has_maiden_name', 'has_mobile', 'all_posts_visible',
            'audio_available', 'has_interests', 'has_books', 'has_tv', 'has_quotes',
            'has_about', 'has_games', 'has_movies', 'has_activities', 'has_music',
            'can_add_as_friend', 'can_invite_to_group', 'is_blacklisted',
            'has_career', 'has_military_service', 'has_hometown',
            'has_universities', 'has_schools', 'has_relatives', 'is_verified',
            'is_confirmed', 'has_status', 'has_occupation',
            'occupation_type_university', 'occupation_type_work', 'has_personal_data'
        ]

    def add_profile_completeness(self, df):
        """
        Calculates a profile completeness score based on boolean-like features.
        Assumes 1.0 means 'has' and 0.0 means 'does not have'. Handles potential NaNs.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with 'profile_completeness' column added.
        """
        df_eng = df.copy()
        # Ensure only columns present in the dataframe are used
        relevant_cols = [col for col in self.profile_bool_cols if col in df_eng.columns]

        if not relevant_cols:
            logging.warning("No relevant boolean columns found for profile completeness calculation.")
            df_eng['profile_completeness'] = 0.0 # Or handle as appropriate
            return df_eng

        # Convert relevant columns to numeric, coercing errors and filling NaNs with 0
        for col in relevant_cols:
             # Check if column exists before trying conversion
            if col in df_eng.columns:
                 df_eng[col] = pd.to_numeric(df_eng[col], errors='coerce').fillna(0)
            else:
                 logging.warning(f"Column {col} not found in DataFrame for completeness calculation.")


        # Calculate completeness score
        df_eng['profile_completeness'] = df_eng[relevant_cols].sum(axis=1) / len(relevant_cols)
        logging.info("Added 'profile_completeness' feature.")
        return df_eng

    def engineer_features(self, df):
        """
        Applies all feature engineering steps.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new features added.
        """
        logging.info("Starting feature engineering...")
        df_engineered = self.add_profile_completeness(df)
        # Add calls to other feature engineering methods here if implemented
        # e.g., df_engineered = self.calculate_post_frequency(df_engineered)
        logging.info("Feature engineering finished.")
        return df_engineered