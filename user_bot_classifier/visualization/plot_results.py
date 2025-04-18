# visualization/plot_results.py
# (Potentially empty or contains utility functions if needed)
# Plotting functions moved to evaluation/metrics.py for better cohesion.
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_confusion_matrix(cm, classes, model_name, filename='confusion_matrix.png'):
    """Plots a confusion matrix."""
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(filename)
        logging.info(f"Confusion matrix for {model_name} saved to {filename}")
        # plt.show()
        plt.close()
    except Exception as e:
        logging.error(f"Error plotting confusion matrix for {model_name}: {e}")

# Add other general plotting functions here if desired