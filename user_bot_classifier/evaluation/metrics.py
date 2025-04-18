# evaluation/metrics.py
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd # Import pandas
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix,
                             classification_report)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelEvaluator:
    """Evaluates trained classification models."""

    def __init__(self, pipelines):
        """
        Initializes the ModelEvaluator.

        Args:
            pipelines (dict): Dictionary of trained model pipelines {model_name: pipeline}.
        """
        self.pipelines = pipelines
        self.results = {}
        logging.info("ModelEvaluator initialized.")

    def evaluate(self, X_test, y_test):
        """
        Evaluates all models on the test set.

        Args:
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True test labels.

        Returns:
            dict: Dictionary containing evaluation metrics for each model.
        """
        self.results = {}
        if not self.pipelines:
            logging.warning("No pipelines available for evaluation.")
            return self.results

        for name, pipeline in self.pipelines.items():
            logging.info(f"Evaluating model: {name}...")
            try:
                y_pred = pipeline.predict(X_test)
                # predict_proba might not be available for all models/pipelines
                y_proba = None
                if hasattr(pipeline, "predict_proba"):
                    y_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of positive class

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
                cm = confusion_matrix(y_test, y_pred)
                report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)


                self.results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'confusion_matrix': cm,
                    'classification_report': report,
                    'y_pred': y_pred, # Store predictions if needed later
                    'y_proba': y_proba # Store probabilities if needed later
                }
                logging.info(f"Evaluation complete for {name}.")
            except Exception as e:
                logging.error(f"Error evaluating model {name}: {e}")
                self.results[name] = {'error': str(e)} # Store error message

        return self.results

    def print_results(self):
        """Prints the evaluation results in a formatted way."""
        if not self.results:
            print("No evaluation results available.")
            return

        print("\n--- Model Evaluation Results ---")
        for name, metrics in self.results.items():
            if 'error' in metrics:
                print(f"\nModel: {name}\n  Error during evaluation: {metrics['error']}")
                continue

            print(f"\nModel: {name}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            if metrics['roc_auc'] is not None:
                print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
            else:
                 print("  ROC AUC:   N/A")
            print("  Confusion Matrix:")
            print(metrics['confusion_matrix'])
            # print("\n  Classification Report:")
            # print(classification_report(y_test, metrics['y_pred'], zero_division=0)) # Or use stored report
        print("------------------------------")

    def plot_roc_curves(self, y_test, save_dir='visualization'):
        """
        Plots ROC curves for all evaluated models.

        Args:
            y_test (pd.Series): True test labels.
            save_dir (str): Directory to save the plot.
        """
        if not self.results:
            logging.warning("No results to plot ROC curves.")
            return

        plt.figure(figsize=(10, 8))
        for name, metrics in self.results.items():
             if 'error' in metrics or metrics.get('y_proba') is None:
                logging.warning(f"Skipping ROC curve for {name} due to error or missing probabilities.")
                continue
             fpr, tpr, _ = roc_curve(y_test, metrics['y_proba'])
             roc_auc = metrics['roc_auc']
             plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='lower right')
        plt.grid(True)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, 'roc_curves.png')
        try:
            plt.savefig(save_path)
            logging.info(f"ROC curves plot saved to {save_path}")
        except Exception as e:
            logging.error(f"Failed to save ROC curve plot: {e}")
        # plt.show() # Optionally display plot immediately
        plt.close() # Close plot to free memory

    def plot_feature_importances(self, save_dir='visualization', top_n=20):
        """
        Plots feature importances for tree-based models (Random Forest, Gradient Boosting).

        Args:
            save_dir (str): Directory to save the plots.
            top_n (int): Number of top features to display.
        """
        if not self.pipelines:
            logging.warning("No pipelines available for feature importance.")
            return

        for name, pipeline in self.pipelines.items():
            # Check if the final step is a classifier with feature_importances_
            if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                logging.info(f"Plotting feature importances for {name}...")
                try:
                    # Get feature names from the preprocessor step
                    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
                    importances = pipeline.named_steps['classifier'].feature_importances_

                    indices = np.argsort(importances)[::-1]
                    top_indices = indices[:top_n]

                    plt.figure(figsize=(12, max(6, top_n // 2))) # Adjust size based on N
                    plt.title(f"Feature Importances - {name} (Top {top_n})")
                    plt.barh(range(len(top_indices)), importances[top_indices], align='center')
                    plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
                    plt.xlabel('Relative Importance')
                    plt.gca().invert_yaxis() # Display most important at the top

                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, f'feature_importances_{name}.png')
                    plt.tight_layout()
                    plt.savefig(save_path)
                    logging.info(f"Feature importance plot for {name} saved to {save_path}")
                    # plt.show()
                    plt.close()

                except Exception as e:
                    logging.error(f"Error plotting feature importances for {name}: {e}")
            else:
                logging.info(f"Feature importances not available for model type: {name}")

    def plot_classification_results(self, X_test, y_test, model_name,
                                    feature1=None, feature2=None, save_dir='visualization'):
        """
        Визуализирует результаты классификации на тестовом наборе,
        показывая правильные и неправильные предсказания на основе двух признаков.

        Args:
            X_test (pd.DataFrame): Тестовые признаки (исходные, до векторизации).
            y_test (pd.Series): Истинные метки тестового набора.
            model_name (str): Имя модели для визуализации.
            feature1 (str, optional): Имя первого признака для оси X.
                                       Если None, будет выбрана первая числовая колонка.
            feature2 (str, optional): Имя второго признака для оси Y.
                                       Если None, будет выбрана вторая числовая колонка.
            save_dir (str): Директория для сохранения графика.
        """
        if model_name not in self.results or 'error' in self.results[model_name]:
            logging.warning(f"Нет результатов или произошла ошибка для модели {model_name}. Пропуск визуализации классификации.")
            return

        logging.info(f"Построение графика результатов классификации для {model_name}...")

        y_pred = self.results[model_name].get('y_pred')
        if y_pred is None:
            logging.warning(f"Предсказания (y_pred) не найдены для модели {model_name}. Пропуск визуализации.")
            return

        # --- Выбор признаков для осей ---
        numeric_cols = X_test.select_dtypes(include=np.number).columns.tolist()

        if feature1 is None:
            if len(numeric_cols) > 0:
                feature1 = numeric_cols[0]
            else:
                logging.warning(f"Не удалось автоматически выбрать feature1 для {model_name} (нет числовых колонок). Пропуск.")
                return
        elif feature1 not in X_test.columns:
             logging.warning(f"Признак '{feature1}' не найден в X_test для {model_name}. Пропуск.")
             return

        if feature2 is None:
            if len(numeric_cols) > 1:
                feature2 = numeric_cols[1]
            elif len(numeric_cols) == 1 and feature1 != numeric_cols[0]: # Если есть только одна и она не feature1
                 feature2 = numeric_cols[0]
            else:
                logging.warning(f"Не удалось автоматически выбрать feature2 для {model_name} (недостаточно числовых колонок). Пропуск.")
                return
        elif feature2 not in X_test.columns:
             logging.warning(f"Признак '{feature2}' не найден в X_test для {model_name}. Пропуск.")
             return

        if feature1 == feature2:
             logging.warning(f"feature1 и feature2 одинаковы ('{feature1}') для {model_name}. Визуализация будет неинформативной. Пропуск.")
             return

        logging.info(f"Используются признаки: X='{feature1}', Y='{feature2}'")
        # --- Создание временного DataFrame для удобства ---
        plot_df = pd.DataFrame({
            feature1: X_test[feature1],
            feature2: X_test[feature2],
            'True Label': y_test,
            'Predicted Label': y_pred
        })
        plot_df['Correct Prediction'] = (plot_df['True Label'] == plot_df['Predicted Label'])

        # --- Построение графика ---
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            data=plot_df,
            x=feature1,
            y=feature2,
            hue='True Label',  # Цвет по истинной метке
            style='Correct Prediction',  # Стиль маркера по правильности предсказания
            s=80,  # Размер маркера
            alpha=0.7 # Прозрачность
        )

        plt.title(f'Результаты классификации: {model_name}\n(Ось X: {feature1}, Ось Y: {feature2})')
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.legend(title='Легенда', loc='best', labels=['Истина: 0 (Неправильно)', 'Истина: 0 (Правильно)', 'Истина: 1 (Неправильно)', 'Истина: 1 (Правильно)']) # Уточнить легенду вручную
        plt.grid(True, linestyle='--', alpha=0.6)

        # --- Сохранение графика ---
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f'classification_results_{model_name}.png')
        try:
            plt.savefig(save_path)
            logging.info(f"График результатов классификации для {model_name} сохранен в {save_path}")
        except Exception as e:
            logging.error(f"Не удалось сохранить график результатов классификации для {model_name}: {e}")
        plt.close() # Закрыть график, чтобы освободить память