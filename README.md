
# Model Selection and Comparison

## Objective

The goal of this experiment is to classify hand gestures for maze navigation using various machine learning models. We evaluated six commonly used classifiers to determine the most suitable model for this task based on multiple performance metrics.

## Evaluated Models

- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- AdaBoost

Each model was trained on a normalized dataset of hand landmark coordinates extracted using MediaPipe. The performance of the models was evaluated on a validation set using four metrics:
- Accuracy
- Precision
- Recall
- F1 Score

## Performance Summary

The metrics were tracked using MLflow and visualized on a Grafana dashboard. The following summarizes the model performance:

| Model              | Accuracy | Precision | Recall | F1 Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.86     | 0.86      | 0.86   | 0.86     |
| Decision Tree       | 0.94     | 0.94      | 0.94   | 0.94     |
| SVM                 | 0.92     | 0.92      | 0.92   | 0.92     |
| Random Forest       | **0.98** | **0.98**  | **0.98**| **0.98** |
| Gradient Boosting   | 0.97     | 0.97      | 0.97   | 0.97     |
| AdaBoost            | 0.31     | 0.25      | 0.31   | 0.21     |

## Model Choice Justification

Among all the models, **Random Forest** consistently achieved the highest scores across all evaluation metrics:
- **Accuracy:** 0.98
- **Precision:** 0.98
- **Recall:** 0.98
- **F1 Score:** 0.98

These results indicate that the Random Forest model generalizes well to unseen data while maintaining strong classification performance. Additionally, it offers:
- Robustness to noise and overfitting.
- Interpretability of feature importance.
- Fast training with default hyperparameters, which fits the requirements of this project timeline.

Models like AdaBoost and Logistic Regression performed significantly worse in comparison, especially in F1 Score and Recall. Although Gradient Boosting also performed well, it did not outperform Random Forest and required longer training time.

## Conclusion

Based on the validation results and performance consistency, the **Random Forest classifier** was selected as the final model for deployment in the gesture-based maze navigation system.
