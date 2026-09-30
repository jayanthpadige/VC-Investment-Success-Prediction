
# VC Investment Success Prediction

A machine learning application that predicts the potential success of startup investments using startup characteristics, funding information, milestones, and relationships.

## Project Overview

The project uses historical startup data to train a machine learning model that predicts whether a startup is likely to be successful or unsuccessful.

The application provides a user-friendly interface built with Streamlit, allowing users to enter startup details and view a prediction with estimated probabilities.

## Objectives

- Analyze historical startup investment data.
- Preprocess and engineer relevant features.
- Train and evaluate multiple machine learning models.
- Predict startup investment outcomes.
- Display model performance and feature importance.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Matplotlib / Plotly (if used in visualizations)

## Dataset

- Dataset: Startup Success Prediction
- Total records: 923
- Input features: 20
- Target: Success

The dataset includes startup location, category, funding rounds, total funding, milestones, relationships, and other investment-related characteristics.

## Machine Learning Workflow

1. Data collection
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Model training
6. Model evaluation
7. Model selection
8. Deployment using Streamlit Cloud

## Models Evaluated

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Model Performance

The Gradient Boosting model was selected for the deployed application.

| Metric | Score |
|---|---:|
| Accuracy | 83.24% |
| Precision | 83.46% |
| Recall | 92.50% |
| F1-Score | 87.75% |
| ROC-AUC | 86.57% |

These results are based on the project's held-out test dataset. They do not guarantee the outcome of a real-world investment.

## Application Features

- Startup success prediction
- Estimated outcome probabilities
- Model performance information
- Feature importance visualization
- Project overview

## Project Structure

```text
VC-Investment-Success-Prediction/
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   └── raw/
├── models/
│   └── vc_success_prediction_model.pkl
├── notebooks/
├── src/
├── visualizations/
├── .gitignore
├── requirements.txt
└── README.md
```

## Run Locally

Clone the repository:

```bash
git clone https://github.com/jayanthpadige/VC-Investment-Success-Prediction.git
cd VC-Investment-Success-Prediction
```

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install requirements and start the app:

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```

## Live Application

[Open the VC Investment Success Predictor](https://vc-investment-success-prediction-ystkyuixrckej4jmmugbw.streamlit.app/)

## Author

**Padige Jayanth**  
Computer Science and Engineering  
Marwadi University