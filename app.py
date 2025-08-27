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



MODEL_PATH = 'sales_prediction_model.joblib'


@st.cache_resource
def load_model():
    """
    Loads the trained model pipeline from the file.
    Uses st.cache_resource to ensure the model is loaded only once.
    """
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        return model_pipeline
    except FileNotFoundError:
        st.error("Error: The model file 'sales_prediction_model.joblib' was not found.")
        st.info("Please run `model_trainer.py` in your terminal to train and save the model.")
        return None


def predict_sales(query, model_pipeline):
    """
    Takes a natural language query, uses GenAI to extract features,
    makes a prediction with the ML model, and then uses GenAI again
    to formulate a human-readable response.
    """
    if model_pipeline is None:
        return "Model not loaded. Please ensure the model file exists."

    
    api_key = GEMINI_API_KEY 
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={api_key}"

    
    extract_prompt = f"""
    You are an expert data analyst. A manager has asked a question about predicting sales.
    Your task is to extract the key features from the manager's query and format them into a JSON object.
    You must only provide the JSON object in your response. Do not include any other text.
    The required features are: category, brand, region, loyalty_tier, price, discount, qty_sold, ad_spend, channel, competitor_price, stock, age, day_of_year, product_id, customer_id.
    Do not add null or None values.
    
    The manager's question is: "{query}"

    Example output:
    {{
      "brand": "Nike",
      "category": "Sneakers",
      "region": "Bangalore",
      "price": 2500,
      "ad_spend": 5000,
      "channel": "Facebook",
      "qty_sold": 25
    }}
    """
    
    payload_extract = {
        "contents": [{"parts": [{"text": extract_prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "brand": {"type": "STRING"},
                    "category": {"type": "STRING"},
                    "region": {"type": "STRING"},
                    "loyalty_tier": {"type": "STRING"},
                    "price": {"type": "NUMBER"},
                    "discount": {"type": "NUMBER"},
                    "qty_sold": {"type": "NUMBER"},
                    "ad_spend": {"type": "NUMBER"},
                    "channel": {"type": "STRING"},
                    "competitor_price": {"type": "NUMBER"},
                    "stock": {"type": "NUMBER"},
                    "age": {"type": "NUMBER"},
                    "day_of_year": {"type": "NUMBER"},
                    "product_id": {"type": "STRING"},
                    "customer_id": {"type": "STRING"}
                }
            }
        }
    }

    try:
        with st.spinner("Analyzing your query with GenAI..."):
            response_extract = requests.post(api_url, json=payload_extract)
            response_extract.raise_for_status()
            extracted_data = json.loads(response_extract.json()['candidates'][0]['content']['parts'][0]['text'])
    except Exception as e:
        st.error(f"Error during feature extraction with GenAI: {e}")
        return "I am having trouble understanding your request. Please try rephrasing."

    
    df_pred = pd.DataFrame([extracted_data])
    

    all_features = ['category', 'brand', 'region', 'loyalty_tier', 'channel', 'product_id', 'customer_id',
                    'age', 'price', 'discount', 'qty_sold', 'ad_spend', 'competitor_price', 'stock', 'day_of_year']
    for col in all_features:
        if col not in df_pred.columns:
            if col in ['age', 'price', 'discount', 'qty_sold', 'ad_spend', 'competitor_price', 'stock', 'day_of_year']:
                df_pred[col] = 0
            else:
                df_pred[col] = 'Unknown'

    df_pred = df_pred[all_features]

  
    try:
        with st.spinner("Predicting sales..."):
            prediction = model_pipeline.predict(df_pred)[0]
    except Exception as e:
        st.error(f"Error during model prediction: {e}")
        return "I was unable to make a prediction based on your request. The data provided might be outside the scope of my training."
    

    feature_importances = model_pipeline.named_steps['regressor'].feature_importances_
    

    one_hot_features = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
    

    numerical_features = ['age', 'price', 'discount', 'qty_sold', 'ad_spend', 'competitor_price', 'stock', 'day_of_year']
    
    feature_names = np.concatenate([one_hot_features, numerical_features])
    
    sorted_idx = np.argsort(feature_importances)[::-1]
    top_features = []
    for i in sorted_idx[:3]:
        name = feature_names[i].split('__')[-1]
        top_features.append(name)
    
    top_features_string = ', '.join(top_features)

    respond_prompt = f"""
    A sales manager asked a question. Based on their original query and the following sales prediction, formulate a friendly and professional response.

    Original query: "{query}"
    Predicted sales revenue: ${prediction:,.2f}
    Top contributing factors to the prediction: {top_features_string}

    Your response should be direct, polite. Use the predicted sales value to answer the question."
    Also, provide a brief summary of the top 3 factors that most influenced this prediction, based on the feature importance data provided.
    """
    
    payload_respond = {
        "contents": [{"parts": [{"text": respond_prompt}]}]
    }

    try:
        with st.spinner("Generating a conversational response..."):
            response_respond = requests.post(api_url, json=payload_respond)
            response_respond.raise_for_status()
            final_response = response_respond.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        st.error(f"Error during response generation with GenAI: {e}")
        final_response = f"Based on the data you provided, the predicted sales revenue is approximately ${prediction:,.2f}. The top factors were: {top_features_string}."

    return final_response


st.set_page_config(layout="wide")

st.title("🛍️ Sales Prediction Dashboard")
st.write("Ask a natural language question about future sales, and I will predict the revenue.")


model = load_model()

if model is not None:
    st.markdown("---")
    user_query = st.text_input(
        "**Your Question:**",
        "What are the predicted sales for Nike sneakers in Bangalore with a price of $2500 and an ad spend of $1500?"
    )

    if st.button("Predict Sales"):
        if user_query:
            result = predict_sales(user_query, model)
            st.markdown("---")
            st.subheader("Predicted Revenue")
            st.success(result)
        else:
            st.warning("Please enter a question to get a prediction.")
