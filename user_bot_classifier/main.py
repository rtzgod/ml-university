# main.py
import numpy as np
import pandas as pd
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report # For detailed report printing

# Import custom modules
from data_loader.loader import DataLoader
from preprocessing.data_cleaner import DataCleaner
from preprocessing.feature_engineer import FeatureEngineer
from preprocessing.vectorizer import FeatureVectorizer
from modeling.classifiers import ModelTrainer
from evaluation.metrics import ModelEvaluator
# from visualization.plot_results import plot_confusion_matrix # If using separate plotting

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
FILE_PATH = 'bots_vs_users.csv' # Make sure this file is in the same directory or provide full path
TARGET_COLUMN = 'target'
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODELS_TO_TRAIN = ['logistic_regression', 'random_forest', 'gradient_boosting']
VISUALIZATION_DIR = 'visualizations'

# --- Main Workflow ---
if __name__ == "__main__":
    logging.info("Starting User vs Bot Classification Pipeline...")

    # 1. Load Data
    logging.info("--- Step 1: Loading Data ---")
    data_loader = DataLoader(FILE_PATH)
    # Adjust skiprows if your header isn't exactly on the second line (index 1)
    df = data_loader.load_data(na_values=['Unknown', ''])

    if df is None:
        logging.error("Failed to load data. Exiting.")
        exit()

    # Добавьте эту строку для проверки столбцов ПОСЛЕ исправления
    logging.info(f"Columns loaded: {df.columns.tolist()}")

    # Убедитесь, что целевая колонка существует и обрабатывается
    if TARGET_COLUMN not in df.columns:
         logging.error(f"Target column '{TARGET_COLUMN}' not found in the dataset AFTER fix. Columns are: {df.columns.tolist()}. Exiting.")
         exit()
    try:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
        df.dropna(subset=[TARGET_COLUMN], inplace=True) # Remove rows where target is NaN
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
        logging.info(f"Target column '{TARGET_COLUMN}' processed. Value counts:\n{df[TARGET_COLUMN].value_counts()}")
    except Exception as e:
        logging.error(f"Error processing target column '{TARGET_COLUMN}': {e}. Exiting.")
        exit()

    # Separate features and target
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # 2. Feature Engineering (Applied before split for simplicity here)
    logging.info("--- Step 2: Feature Engineering ---")
    feature_engineer = FeatureEngineer()
    X = feature_engineer.engineer_features(X) # Add completeness score

    # 3. Split Data
    logging.info("--- Step 3: Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y # Stratify for imbalanced classes
    )
    logging.info(f"Data split into training ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets.")

    # 4. Preprocessing & Vectorization (using Vectorizer which includes cleaning steps)
    logging.info("--- Step 4: Preprocessing and Vectorization ---")
    # Identify column types from the engineered training data
    data_cleaner_temp = DataCleaner() # Use cleaner just for type identification
    numeric_features, categorical_features = data_cleaner_temp.identify_column_types(X_train, TARGET_COLUMN) # Pass None as target here

    vectorizer = FeatureVectorizer(numeric_features=numeric_features, categorical_features=categorical_features)
    vectorizer.fit(X_train) # Fit vectorizer (including imputers, scaler, encoder) ONLY on training data

    # The actual transformation will happen inside the pipeline during training/prediction

    # 5. Train Models (using Pipelines)
    logging.info("--- Step 5: Training Models ---")
    trainer = ModelTrainer(models_to_train=MODELS_TO_TRAIN, random_state=RANDOM_STATE)
    # Pass the *fitted* vectorizer's preprocessor part to the trainer
    trained_pipelines = trainer.train(vectorizer.preprocessor, X_train, y_train)

    if not trained_pipelines:
        logging.error("No models were trained successfully. Exiting.")
        exit()

    # 6. Evaluate Models
    logging.info("--- Step 6: Evaluating Models ---")
    evaluator = ModelEvaluator(trained_pipelines)
    results = evaluator.evaluate(X_test, y_test) # Evaluator uses the pipelines, which handle preprocessing
    evaluator.print_results()

    # 7. Visualize Results
    logging.info("--- Step 7: Visualizing Results ---")
    if not os.path.exists(VISUALIZATION_DIR):
        os.makedirs(VISUALIZATION_DIR)
        logging.info(f"Created directory: {VISUALIZATION_DIR}")

    evaluator.plot_roc_curves(y_test, save_dir=VISUALIZATION_DIR)
    try:
        # Используем имена признаков из X_train, так как они те же, что и в X_test
        # feature_names = vectorizer.get_feature_names() # Это имена ПОСЛЕ трансформации, не подходят для исходных данных
        # Вместо этого, используем numeric_features, полученные ранее
        evaluator.plot_feature_importances(save_dir=VISUALIZATION_DIR, top_n=25) # Использует имена после трансформации
    except Exception as e:
         logging.error(f"Error getting feature names or plotting importance: {e}")

    # *** ДОБАВЛЕННЫЙ КОД ДЛЯ ВИЗУАЛИЗАЦИИ КЛАССИФИКАЦИИ ***
    logging.info("--- Plotting Classification Point Results ---")
    # Выбираем признаки для визуализации (можно задать вручную или оставить None для автовыбора)
    # Например, используем первые два числовых признака, которые мы определили ранее
    feature_x = numeric_features[0] if len(numeric_features) > 0 else None
    feature_y = numeric_features[1] if len(numeric_features) > 1 else None

    if feature_x and feature_y:
        for model_name in trained_pipelines.keys():
            evaluator.plot_classification_results(
                X_test, y_test, model_name,
                feature1=feature_x,
                feature2=feature_y,
                save_dir=VISUALIZATION_DIR
            )
    else:
        logging.warning("Недостаточно числовых признаков для построения графика результатов классификации.")


    # Print detailed classification reports
    print("\n--- Detailed Classification Reports ---")
    for model_name, metrics in results.items():
        if 'classification_report' in metrics and isinstance(metrics['classification_report'], dict):
             print(f"\nModel: {model_name}")
             # Need y_pred from the results dictionary if not recalculating
             y_pred_model = metrics.get('y_pred')
             if y_pred_model is not None:
                  print(classification_report(y_test, y_pred_model, zero_division=0))
             else:
                  # Fallback if y_pred wasn't stored or evaluation failed
                  try:
                       y_pred_recalc = trained_pipelines[model_name].predict(X_test)
                       print(classification_report(y_test, y_pred_recalc, zero_division=0))
                  except Exception as e:
                       print(f"  Could not generate report: {e}")

        elif 'error' in metrics:
             print(f"\nModel: {model_name}\n  Error: {metrics['error']}")
    print("-------------------------------------")

    # --- Step 8: Example Prediction (Выполняем ДО выводов) ---
    logging.info("--- Step 8: Example Prediction ---")
    best_model_name = None
    best_f1_score = -1
    example_predictions = None  # Переменная для хранения предсказаний примера
    example_probabilities = None  # Переменная для хранения вероятностей примера

    # Сначала найдем лучшую модель
    if results:
        for name, metrics in results.items():
            if 'error' not in metrics and 'f1_score' in metrics:
                current_f1 = metrics['f1_score']
                if current_f1 > best_f1_score:
                    best_f1_score = current_f1
                    best_model_name = name

    # Теперь выполняем предсказание, если лучшая модель найдена
    if best_model_name and best_model_name in trained_pipelines:
        best_pipeline = trained_pipelines[best_model_name]
        try:
            # Используем первые 2 строки тестового набора как пример
            sample_new_data = X_test.iloc[:2].copy()
            logging.info(f"Predicting on {len(sample_new_data)} new samples using best model: {best_model_name}...")

            example_predictions = best_pipeline.predict(sample_new_data)  # Сохраняем в новую переменную
            if hasattr(best_pipeline, "predict_proba"):
                example_probabilities = best_pipeline.predict_proba(sample_new_data)  # Сохраняем в новую переменную

            logging.info(f"Example Predictions: {example_predictions}")
            if example_probabilities is not None:
                logging.info(f"Example Probabilities (0, 1): {example_probabilities}")

        except Exception as e:
            logging.error(f"Error during example prediction: {e}")
            example_predictions = None  # Сбрасываем в случае ошибки
            example_probabilities = None
    else:
        logging.warning(
            f"Could not perform example prediction as best model '{best_model_name}' was not found or not trained.")

    # --- Step 9: Interpretation and Conclusion ---
    logging.info("--- Step 9: Interpretation and Conclusion ---")
    print("\n======================================")
    print(" ИТОГИ И ВЫВОДЫ")
    print("======================================")

    print("\nЗадача:")
    # ... (текст задачи) ...

    if not results:
        # ... (обработка отсутствия результатов) ...
        exit()

    print("\nРезультаты оценки моделей (на тестовой выборке):")
    # Best model name is already determined above
    # ... (цикл вывода метрик для всех моделей) ...

    print("\n--------------------------------------")
    if best_model_name:
        # ... (вывод лучшей модели, интерпретация метрик, важность признаков) ...

        print("\nВывод:")
        # ... (текст вывода) ...

        # --- Отображение Примера Предсказания ---
        print("\nПример использования для предсказания:")
        # Проверяем, существуют ли переменные с результатами примера
        if 'sample_new_data' in locals() and example_predictions is not None:
            print("  Входные данные (первые 2 строки тестового набора):")
            # print(sample_new_data) # Можно раскомментировать, если нужно видеть сами данные
            print(f"  Предсказанные метки (0=пользователь, 1=бот) моделью '{best_model_name}':",
                  example_predictions)  # Используем сохраненную переменную
            if example_probabilities is not None:
                print(
                    f"  Вероятности принадлежности к классу 'бот' (1): {[f'{p:.2f}' for p in example_probabilities[:, 1]]}")  # Используем сохраненную переменную
        else:
            print("  (Пример предсказания не был выполнен или завершился с ошибкой)")
        # --- Конец Отображения Примера ---

    else:
        print("\nНе удалось определить лучшую модель для выводов.")

    print("\n======================================")
    logging.info("Pipeline finished.")