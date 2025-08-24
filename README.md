# Sales Prediction with GenAI
This project demonstrates a powerful and interactive sales prediction tool that combines a traditional Machine Learning model with the natural language capabilities of a Generative AI model. The system allows a sales manager to ask questions about future sales in plain English and receive a precise, data-driven prediction along with an explanation of the key influencing factors.

## Project Overview
The application is built to provide an intuitive interface for business users to interact with a complex predictive model. Instead of requiring a data scientist to manually input features for a prediction, the application leverages Generative AI to translate a human question into a structured input that the ML model can understand. This makes the power of machine learning accessible to everyone.

## Features
Natural Language Interaction: Ask questions in everyday language, like "What are the predicted sales for Nike sneakers in Bangalore with a price of $2500?".

Hybrid AI Approach: This project showcases a powerful synergy. The Machine Learning (RandomForestRegressor) model is responsible for the core predictive intelligence, analyzing historical data and making an accurate, data-driven forecast. The Generative AI (Gemini API) acts as an intelligent layer on top, translating complex human queries into a format the ML model can use and then turning the numerical prediction back into a conversational, easy-to-understand response. This combination ensures both accuracy and accessibility.

Predictive Insights: The response not only gives a prediction but also identifies the top three most influential factors, giving users a clear understanding of what drives the forecast.

Interactive Web App: A user-friendly interface built with Streamlit for easy deployment and use.

## Getting Started
Prerequisites
Make sure you have Python installed on your system. It's recommended to create a virtual environment to manage dependencies.

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Nirupam-Naragund/Traditional-ML-powered-with-GenAI---Sales-prediction.git
cd Traditional-ML-powered-with-GenAI---Sales-prediction
```
### 2. Create a virtual environment (Python 3.11 recommended)
```bash
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Linux/Mac
```
### 3. Install dependencies

```bash
pip install pandas scikit-learn joblib streamlit requests numpy
```

### 5. Train the model

```bash
python train.py
```

### 6. Predict

```bash
streamlit run app.py
```

