import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import time

# Set page config
st.set_page_config(page_title="SmartValue", page_icon="🏠")

# App title
st.title("🏠 SmartValue")

# Define paths
MODEL_PATH = "house_price_model.joblib"
PREPROCESSOR_PATH = "preprocessor.joblib"

# Function to load data
@st.cache_data
def load_data():
    """Load and return the training and testing datasets"""
    train_file = 'train.csv'
    test_file = 'test.csv'
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        return train_df, test_df
    else:
        raise FileNotFoundError("Dataset files not found. Make sure 'train.csv' and 'test.csv' are in the current directory.")

# Function to get important features
def get_important_features():
    """Returns a list of important features for house price prediction"""
    # Core features that generally have high impact on house prices
    numerical_features = [
        'GrLivArea', 'TotalBsmtSF', 'YearBuilt', 'YearRemodAdd', 
        '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',
        'GarageArea', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'WoodDeckSF',
        'OpenPorchSF', 'OverallQual', 'OverallCond'
    ]
    
    categorical_features = [
        'MSZoning', 'Neighborhood', 'BldgType', 'HouseStyle', 'CentralAir',
        'GarageType', 'SaleCondition', 'Foundation', 'Heating', 'Electrical',
        'KitchenQual', 'ExterQual', 'BsmtQual'
    ]
    
    return numerical_features, categorical_features

# Function to preprocess data
def preprocess_data(df, target_col=None, test_size=0.2, random_state=42):
    """Preprocess the data for model training"""
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Get important features
    numerical_features, categorical_features = get_important_features()
    
    # Filter out features that don't exist in the dataframe
    numerical_features = [f for f in numerical_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]
    
    # Create preprocessor
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop other columns
    )
    
    # For training data, split into train/test sets
    if target_col and target_col in df.columns:
        # Drop rows with missing target values
        df = df.dropna(subset=[target_col])
        
        # Get features and target
        X = df[numerical_features + categorical_features]
        y = df[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Fit and transform the training data
        X_train_preprocessed = preprocessor.fit_transform(X_train)
        X_test_preprocessed = preprocessor.transform(X_test)
        
        return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocessor
    
    # For test data, just preprocess
    else:
        X = df[numerical_features + categorical_features]
        X_preprocessed = preprocessor.fit_transform(X)
        return X_preprocessed, None, None, None, preprocessor

# Function to prepare a single prediction
def prepare_single_prediction(input_data, preprocessor):
    """Prepare a single input for prediction using the fitted preprocessor"""
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply the same preprocessing as training data
    processed_input = preprocessor.transform(input_df)
    
    return processed_input

# Function to train models
def train_all_models(X_train, y_train, X_test, y_test):
    """Train all models and select the best one"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    all_metrics = {}
    all_models = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        all_models[model_name] = model
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        all_metrics[model_name] = {
            'R2': r2
        }
    
    # Find the best model based on R2 score
    best_model_name = max(all_metrics, key=lambda model: all_metrics[model]['R2'])
    best_model = all_models[best_model_name]
    
    return best_model, best_model_name

# Function to predict price
def predict_price(model, input_data):
    """Predict house price using a trained model"""
    predicted_price = model.predict(input_data)[0]
    return predicted_price

# Function to save model
def save_model(model, filename):
    """Save a trained model to disk"""
    joblib.dump(model, filename)

# Function to load model
def load_model(filename):
    """Load a trained model from disk"""
    return joblib.load(filename)

# Function to get feature descriptions
def get_feature_descriptions():
    """Return descriptions for the features"""
    return {
        'BldgType': 'Type of building (e.g., single-family, townhouse)',
        'HouseStyle': 'Style of house (e.g., one-story, two-story)',
        'GarageType': 'Type of garage (e.g., attached, detached)'
    }

# Main function to run the app
def main():
    # Load data
    try:
        train_df, _ = load_data()
        
        # Check if model exists, otherwise train it
        if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
            # Load model and preprocessor
            model = load_model(MODEL_PATH)
            preprocessor = load_model(PREPROCESSOR_PATH)
            
            # Log message
            st.sidebar.success("Using pre-trained model")
        else:
            # Show spinner while training
            with st.spinner("Training model... Please wait"):
                # Preprocess the data
                X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
                    train_df, 
                    target_col='SalePrice',
                    test_size=0.2
                )
                
                # Train models and select best
                model, best_model_name = train_all_models(X_train, y_train, X_test, y_test)
                
                # Save model and preprocessor
                save_model(model, MODEL_PATH)
                save_model(preprocessor, PREPROCESSOR_PATH)
                
                # Log message
                st.sidebar.success(f"Trained a new model: {best_model_name}")
        
        # Get feature descriptions
        descriptions = get_feature_descriptions()
        
        # Create prediction interface
        st.markdown("## House Price Prediction")
        st.markdown("Enter details about the house to get a price estimate:")
        
        # Dictionary to store inputs
        input_data = {}
        
        # Create a form for the prediction inputs
        with st.form("prediction_form"):
            # Building Type
            if 'BldgType' in train_df.columns:
                unique_values = train_df['BldgType'].dropna().unique().tolist()
                input_data['BldgType'] = st.selectbox(
                    "Building Type", 
                    options=unique_values,
                    help=descriptions.get('BldgType', 'Type of building')
                )
            
            # House Style
            if 'HouseStyle' in train_df.columns:
                unique_values = train_df['HouseStyle'].dropna().unique().tolist()
                input_data['HouseStyle'] = st.selectbox(
                    "House Style", 
                    options=unique_values,
                    help=descriptions.get('HouseStyle', 'Style of house')
                )
            
            # Garage Type
            if 'GarageType' in train_df.columns:
                unique_values = train_df['GarageType'].dropna().unique().tolist()
                input_data['GarageType'] = st.selectbox(
                    "Garage Type", 
                    options=unique_values,
                    help=descriptions.get('GarageType', 'Type of garage')
                )
            
            # Add the prediction button
            predict_button = st.form_submit_button("Predict House Price")
        
        # Use the model to make a prediction
        if predict_button:
            # Get all features used in the model
            num_features = preprocessor.transformers_[0][2]
            cat_features = preprocessor.transformers_[1][2]
            used_features = list(num_features) + list(cat_features)
            
            # Add default values for missing features
            for feature in used_features:
                if feature not in input_data:
                    if feature in train_df.select_dtypes(include=[np.number]).columns:
                        # For numeric features, use median
                        input_data[feature] = train_df[feature].median()
                    elif feature in train_df.columns:
                        # For categorical features, use most common value
                        input_data[feature] = train_df[feature].mode()[0]
            
            # Now make the prediction
            with st.spinner("Calculating price..."):
                # Prepare the input data
                processed_input = prepare_single_prediction(input_data, preprocessor)
                
                # Make prediction
                predicted_price = predict_price(model, processed_input)
                
                # Display the result with larger font and styling
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-top: 20px;">
                    <h2 style="text-align: center; color: #1e88e5;">Predicted House Price</h2>
                    <h1 style="text-align: center; color: #2e7d32; font-size: 48px;">₹{84*predicted_price:,.2f}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Show a simple summary of inputs
                st.subheader("Your Selections")
                
                # Create a table with the input features
                input_summary = pd.DataFrame([
                    {"Feature": "Building Type", "Value": input_data['BldgType']},
                    {"Feature": "House Style", "Value": input_data['HouseStyle']},
                    {"Feature": "Garage Type", "Value": input_data['GarageType']}
                ])
                st.table(input_summary)
        
        # Add an information box
        st.sidebar.info("""
        ## How it works
        
        This app uses machine learning to estimate house prices based on key characteristics. 
        
        The model was trained on the Ames Housing dataset with various house features.
        """)
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure you have the required data files (train.csv and test.csv) in the application directory.")

if __name__ == "__main__":
    main()