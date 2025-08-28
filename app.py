import streamlit as st
import joblib
import pandas as pd
import json
import requests
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_PATH = 'sales_prediction_model3.joblib'

@st.cache_resource
def load_model():
    try:
        artifacts = joblib.load(MODEL_PATH)
        return artifacts["model_pipeline"], artifacts["numeric_medians"]
    except FileNotFoundError:
        st.error("Error: The model file 'sales_prediction_model.joblib' was not found. Please run `train.py` first.")
        return None, None

def predict_sales(query, model_pipeline, numeric_medians):
    if model_pipeline is None:
        return "Model not loaded. Please train the model first."

    api_key = GEMINI_API_KEY
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    # Step 1: Extract features using GenAI
    extract_prompt = f"""
    Extract sales features as JSON. If any feature is missing, skip it (do NOT put null).
    Required keys: category, brand, region, loyalty_tier, price, discount, qty_sold, ad_spend, channel, competitor_price, stock, age, day_of_year, product_id, customer_id.
    Manager query: "{query}"
    """

    payload_extract = {"contents": [{"parts": [{"text": extract_prompt}]}], "generationConfig": {"responseMimeType": "application/json"}}

    try:
        with st.spinner("Analyzing your query with GenAI..."):
            response_extract = requests.post(api_url, json=payload_extract)
            response_extract.raise_for_status()
            extracted_data = json.loads(response_extract.json()['candidates'][0]['content']['parts'][0]['text'])
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return "Couldn't understand your query. Please try again."

    df_pred = pd.DataFrame([extracted_data])

    # Step 2: Handle missing values
    cat_features = ['category', 'brand', 'region', 'loyalty_tier', 'channel', 'product_id', 'customer_id']
    num_features = ['age', 'price', 'discount', 'qty_sold', 'ad_spend', 'competitor_price', 'stock', 'day_of_year']
    all_features = cat_features + num_features

    for col in all_features:
        if col not in df_pred.columns:
            if col in num_features:
                df_pred[col] = numeric_medians.get(col, 0)
            else:
                df_pred[col] = "Unknown"

    df_pred = df_pred[all_features]

    # Step 3: Prediction
    try:
        with st.spinner("Predicting sales..."):
            prediction = model_pipeline.predict(df_pred)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Could not make a prediction due to missing or invalid data."

    # Step 4: Get top contributing factors
    feature_importances = model_pipeline.named_steps['regressor'].feature_importances_
    one_hot_features = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
    feature_names = np.concatenate([one_hot_features, num_features])

    sorted_idx = np.argsort(feature_importances)[::-1]
    top_features = [feature_names[i].split('__')[-1] for i in sorted_idx[:3]]
    top_features_string = ', '.join(top_features)

    # Step 5: Generate AI response
    respond_prompt = f"""
    Original query: "{query}"
    Predicted sales revenue: ${prediction:,.2f}
    Top contributing factors to the prediction: {top_features_string}

    Your response should be direct, polite, and professional. 
    Use the predicted sales value to answer the question clearly.
    Also, summarize briefly how the top 3 factors influenced this prediction.
    """

    payload_respond = {"contents": [{"parts": [{"text": respond_prompt}]}]}

    try:
        with st.spinner("Generating response..."):
            response_respond = requests.post(api_url, json=payload_respond)
            response_respond.raise_for_status()
            final_response = response_respond.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        final_response = f"Predicted sales revenue: ${prediction:,.2f}. Top factors: {top_features_string}"

    return final_response


# ---- Streamlit UI ----
st.set_page_config(layout="wide")
st.title("🛍️ Sales Prediction Dashboard")
st.write("Ask a natural language question about future sales, and I will predict the revenue.")

model, numeric_medians = load_model()

if model is not None:
    st.markdown("---")
    user_query = st.text_input("**Your Question:**",
                               "What are the predicted sales for Nike sneakers in Bangalore with a price of $2500 and an ad spend of $1500?")

    if st.button("Predict Sales"):
        if user_query:
            result = predict_sales(user_query, model, numeric_medians)
            st.markdown("---")
            st.subheader("Predicted Revenue")
            st.success(result)
        else:
            st.warning("Please enter a question to get a prediction.")
