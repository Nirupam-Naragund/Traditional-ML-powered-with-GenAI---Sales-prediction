import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

MODEL_PATH = 'sales_prediction_model.joblib'

def train_model():
    """
    Loads data, preprocesses it, and trains a RandomForestRegressor model
    to predict sales revenue. Also stores medians for numeric features.
    """
    print("Training the sales prediction model...")

    try:
        # Load the CSV file
        df = pd.read_csv('sales_data_5000.csv')
    except FileNotFoundError:
        print("Error: The 'sales_data_5000.csv' file was not found. Please run sales_data_generator.py first.")
        return

    # Drop holiday column and process dates
    df = df.drop(columns=['holiday'])
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df = df.drop(columns=['date'])

    # Define features (X) and target (y)
    X = df.drop(columns=['revenue'])
    y = df['revenue']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Categorical & numerical features
    categorical_features = ['category', 'brand', 'region', 'loyalty_tier', 'channel', 'product_id', 'customer_id']
    numerical_features = ['age', 'price', 'discount', 'qty_sold', 'ad_spend', 'competitor_price', 'stock', 'day_of_year']

    # Preprocessor for categorical features
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
        remainder='passthrough'
    )

    # Model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Train model
    model_pipeline.fit(X_train, y_train)

    # Evaluate model
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("\n--- Model Evaluation ---")
    print(f"R-squared (R^2) Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print("------------------------")

    # Save medians for numeric features
    numeric_medians = X_train[numerical_features].median().to_dict()

    # Save model & medians together
    model_artifacts = {
        "model_pipeline": model_pipeline,
        "numeric_medians": numeric_medians
    }
    joblib.dump(model_artifacts, MODEL_PATH)
    print("Model and medians saved to 'sales_prediction_model.joblib'.")

if __name__ == "__main__":
    train_model()
