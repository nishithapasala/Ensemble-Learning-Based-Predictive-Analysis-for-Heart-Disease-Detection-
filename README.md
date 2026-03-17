# Ensemble-Learning-Based-Predictive-Analysis-for-Heart-Disease-Detection


This project presents a machine learning approach for **early detection of heart disease** using ensemble learning techniques.  
The work was developed as part of a research paper by students and faculty from the **Institute of Aeronautical Engineering, Hyderabad**.

## Objective
Build a reliable prediction system that helps identify heart disease risk early by combining multiple classifiers (ensemble methods) instead of relying on a single model.

## Key Highlights
- Ensemble focus: Bagging + Boosting + Stacking
- Best model: **Optimized Random Forest**
- Highest performance: **80.04% accuracy** + strong recall (important for not missing cases)
- Used two popular datasets:
  - Large dataset (~70,000 records) → better generalization
  - UCI Heart Disease dataset (303 records) → standard benchmark

## Models Compared

| Model                    | Best Accuracy | Notes                              |
|--------------------------|---------------|------------------------------------|
| Optimized Random Forest  | 80.04%        | Highest accuracy & good recall     |
| Stacking Classifier      | ~76–73%       | Solid but slightly lower           |
| Logistic Regression      | ~75–69%       | Simple baseline                    |
| Bagging Meta Estimator   | ~75–71%       | Reduces overfitting                |
| Gaussian Naïve Bayes     | ~74–57%       | Weak on larger/unbalanced data     |

## Main Findings
- Ensemble models clearly outperform single classifiers
- Recall is prioritized (avoid missing true heart disease cases)
- Larger dataset → better and more stable results
- Careful preprocessing (missing values, encoding, feature selection) was very important


## Future Ideas 
- Try XGBoost, LightGBM or deep learning
- Test on real hospital data
- Add more diseases (multi-disease prediction)
- Build a clinical decision support tool


