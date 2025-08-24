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
    to predict sales revenue.
    """
    print("Training the sales prediction model...")

    try:
        # Load the CSV file.
        df = pd.read_csv('sales_data_5000.csv')
    except FileNotFoundError:
        print("Error: The 'sales_data_5000.csv' file was not found. Please run sales_data_generator.py first.")
        return

    # Drop the holiday column as it's a binary categorical feature which might not add much value.
    # The competitor price is also a good indicator so we will keep it.
    df = df.drop(columns=['holiday'])

    # Feature Engineering: Convert date to a numerical feature (day of the year)
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df = df.drop(columns=['date'])

    # Define features (X) and target (y)
    X = df.drop(columns=['revenue'])
    y = df['revenue']

    # --- NEW: Split the data into training and testing sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Identify categorical and numerical features
    categorical_features = ['category', 'brand', 'region', 'loyalty_tier', 'channel', 'product_id', 'customer_id']
    numerical_features = ['age', 'price', 'discount', 'qty_sold', 'ad_spend', 'competitor_price', 'stock', 'day_of_year']

    # Create a preprocessor pipeline for one-hot encoding categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full model pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Fit the pipeline to the training data only
    model_pipeline.fit(X_train, y_train)

    # --- NEW: Evaluate the model on the test data ---
    y_pred = model_pipeline.predict(X_test)

    # Calculate evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("\n--- Model Evaluation ---")
    print(f"R-squared (R^2) Score: {r2:.4f}")
    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print("------------------------")

    # Save the trained model and preprocessor
    joblib.dump(model_pipeline, MODEL_PATH)

    print("Model training complete. Model saved to 'sales_prediction_model.joblib'.")

if __name__ == "__main__":
    train_model()
