# Heart-Disease-Prediction
The Digital Stethoscope: A Comparative Analysis of Machine Learning Algorithms for Early Heart Disease Detection

A machine learning pipeline for predicting the presence of heart disease from clinical patient data, built for high recall (sensitivity) and interpretability, and deployed as an interactive clinical decision-support web app.
________________________________________
Overview
Cardiovascular disease is the leading cause of death globally, and early detection meaningfully improves outcomes. This project builds, compares, and deploys several supervised learning models on the UCI Heart Disease dataset to flag high-risk patients from routinely collected clinical attributes, optimizing for recall so that at-risk patients are less likely to be missed.
Objectives
•	Perform exploratory data analysis (EDA) to identify key clinical risk factors
•	Preprocess and engineer features to improve predictive power
•	Train and compare seven classification algorithms
•	Tune hyperparameters to maximize diagnostic recall
•	Explain model predictions with SHAP / LIME so clinicians can see why a prediction was made
•	Deploy the best model as a real-time, interactive web app
Dataset
•	Source: UCI Machine Learning Repository — Heart Disease dataset (Cleveland database)
•	Size: 303 instances, 14 clinical attributes
•	Target: presence (1) or absence (0) of heart disease
Attribute	Description
age, sex	Demographics
cp	Chest pain type
trestbps	Resting blood pressure
chol	Serum cholesterol
fbs	Fasting blood sugar > 120 mg/dl
restecg	Resting ECG results
thalach	Maximum heart rate achieved
exang	Exercise-induced angina
oldpeak, slope	ST depression / slope of peak exercise ST segment
ca	Number of major vessels colored by fluoroscopy
thal	Thalassemia status
Models Compared
Model	 and their Purpose
Support Vector Machine (SVM):	Optimal-hyperplane classification
Naive Bayes:	Fast probabilistic baseline
Random Forest	:Ensemble of trees, reduces overfitting
Gradient Boosting (XGBoost)	:Boosted ensemble, primary performance benchmark
Methodology
1.	Data acquisition — load the dataset from a public CSV endpoint for reproducibility
2.	Cleaning & preprocessing — median/mode imputation, one-hot encoding of categoricals, standard scaling of continuous features
3.	EDA — distribution plots, outlier detection, correlation heatmap (matplotlib, seaborn)
4.	Feature engineering — age banding, derived risk ratios; importance ranked via Random Forest importances / RFE
5.	Model training — 80/20 train/test split, all models via scikit-learn
6.	Hyperparameter tuning — GridSearchCV + k-fold cross-validation
7.	Evaluation — accuracy, precision, recall, F1, ROC-AUC, PR-AUC, calibration curves, and McNemar's/paired t-test for model comparison; recall is the primary optimization target, with the decision threshold tuned accordingly
8.	Interpretability — SHAP values (global + per-prediction) and LIME as a model-agnostic cross-check
9.	Deployment — best model serialized (pickle) and served via a Streamlit app that returns a prediction, a probability, and a SHAP-based explanation
Tech Stack
•	Language: Python 3
•	ML/Data: scikit-learn, pandas, numpy, xgboost
•	Interpretability: shap, lime
•	Visualization: matplotlib, seaborn
•	Deployment: Streamlit
•	Model persistence: pickle
Project Structure
├── data/                  # raw and processed dataset
├── notebooks/             # EDA and experimentation notebooks
├── src/
│   ├── preprocessing.py   # cleaning, encoding, scaling
│   ├── features.py        # feature engineering
│   ├── train.py           # model training and tuning
│   ├── evaluate.py        # metrics, calibration, significance tests
│   └── explain.py         # SHAP / LIME interpretability
├── app/
│   └── streamlit_app.py   # deployed decision-support interface
├── models/                # serialized trained models
├── requirements.txt
└── README.md
Evaluation Metrics
Primary metric: Recall (Sensitivity) — minimizing false negatives is prioritized given the clinical cost of a missed diagnosis. Reported alongside: Accuracy, Precision, F1-Score, ROC-AUC, PR-AUC, and calibration curves. Statistical significance between models is checked with McNemar's test / paired t-test across CV folds before any model is declared best.
Limitations
•	Dataset is limited to the Cleveland cohort (303 instances); results may not generalize across broader or more diverse populations
•	Not validated on an external/held-out clinical population
•	Intended for screening/decision support only — not a diagnostic substitute
Ethical Considerations
•	Privacy: dataset is fully anonymized and publicly available; no patient consent required for this research use
•	Disclaimer: the app does not provide a definitive diagnosis and directs users to consult a licensed healthcare professional
•	Bias: performance will be reported by subgroup (e.g., age, sex) given the dataset's demographic limitations, and the model should not be applied uncritically to underrepresented populations

