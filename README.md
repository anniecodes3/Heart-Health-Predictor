Heart Health Predictor : 
A machine learning-powered web application that predicts the likelihood of heart disease based on a user’s health parameters. This project combines simplicity, accuracy, and explainability — making AI-based heart health prediction accessible to everyone.

Project Overview : 
Heart disease remains one of the leading causes of death globally. Early risk prediction is crucial 

— and this app helps you do just that using:
- Logistic Regression Model for prediction  
- Streamlit App for interactive user input  
- SHAP (SHapley Additive Explanations) for interpretability  
- Visuals to explain which features influenced the result

Features : 
- Clean UI with health parameter inputs
- Instant prediction of heart disease risk (Yes/No)
- Dynamic SHAP plot showing feature impact
- Easy-to-run and open-source

Project Structure :
├── app.py                -Streamlit frontend interface
├── heart_health.py       -ML model prediction logic
├── Heart Health.ipynb    -EDA and model building notebook
├── heart_model.joblib    -Trained Logistic Regression model
├── requirements.txt      -Python dependencies
├── README.md             -Project instructions & overview
└── .devcontainer/        -VS Code Dev Container setup

Technologies Used : 
-Python 3.8+
-Pandas, NumPy, Matplotlib
-Scikit-learn – Machine Learning
-Joblib – Model serialization
-Streamlit – Web app interface
-Jupyter Notebook – Exploratory data analysis & model training

