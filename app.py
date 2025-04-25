import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
import os
from scipy.stats import randint, uniform
import io
import base64

# Import the data loader
try:
    from data_loader import ensure_data_file_exists
    # Try to ensure the data file exists before proceeding
    ensure_data_file_exists()
except Exception as e:
    st.warning(f"Data loader initialization error: {str(e)}")
    # Continue with the app - we'll handle missing data gracefully later

# Set page configuration
st.set_page_config(
    page_title="Earthquake Risk Analysis",
    page_icon="🌎",
    layout="wide"
)

# Function to encode the image
def get_base64_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Title and logo at the top 
st.markdown("""
<div style="display: flex; align-items: center; gap: 20px; padding: 10px; background-color: #1E1E2E; border-radius: 5px; margin-bottom: 20px;">
    <img src="data:image/jpeg;base64,{}" width="100">
    <div>
        <h1 style="margin: 0; color: white;">Earthquake Risk Analysis</h1>
        <p style="margin: 0; color: #CCC;">This application visualizes earthquake data across North America</p>
    </div>
</div>
""".format(get_base64_encoded_image("logo.png")), unsafe_allow_html=True)

# File upload option and dataset selection
st.subheader("Dataset Selection")
dataset_option = st.radio("Choose dataset source:", ["Use default North America dataset", "Upload your own dataset"])

# Required columns for the dataset
required_columns = ['LATITUDE', 'LONGITUDE', 'MAGNITUDE', 'DATE']

# Function to validate dataset
def validate_dataset(df):
    # Convert column names to uppercase for consistency
    df.columns = [col.upper() for col in df.columns]
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    # Basic data type validation
    try:
        # Ensure numeric columns are numeric
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'])
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'])
        df['MAGNITUDE'] = pd.to_numeric(df['MAGNITUDE'])
        
        # Try to convert DATE to datetime
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        return True, df
    except Exception as e:
        return False, f"Error processing data: {str(e)}"

# Load data based on user choice
@st.cache_data
def load_default_data():
    # CSV file is in the root directory, not in src
    dataset_path = "earthquake_cleandata_posteda.csv"
    
    try:
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            if not df.empty:
                st.success(f"Successfully loaded dataset with {len(df)} records")
                return df
            else:
                st.error(f"The dataset at {dataset_path} is empty")
        else:
            st.error(f"Dataset file not found at {dataset_path}")
            st.info("Please ensure the CSV file is in the project directory")
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
    
    # Return an empty DataFrame with the required columns if we can't load the file
    return pd.DataFrame(columns=required_columns)

if dataset_option == "Use default North America dataset":
    df = load_default_data()
else:
    uploaded_file = st.file_uploader("Upload CSV file containing earthquake data", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df_uploaded = pd.read_csv(uploaded_file)
            
            # Display raw data preview
            with st.expander("Preview uploaded data"):
                st.dataframe(df_uploaded.head())
            
            # Validate the dataset
            is_valid, result = validate_dataset(df_uploaded)
            
            if is_valid:
                df = result
                st.success(f"Successfully loaded dataset with {len(df)} records")
                
                # Show column mapping
                with st.expander("Column mapping"):
                    st.write("The following columns were found in your dataset:")
                    for col in df.columns:
                        st.write(f"- {col}")
            else:
                st.error(result)
                # Fall back to default dataset
                df = load_default_data()
                st.warning("Using default dataset instead")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            # Fall back to default dataset
            df = load_default_data()
            st.warning("Using default dataset instead")
    else:
        # If no file is uploaded, use default dataset
        df = load_default_data()
        
# Dataset statistics
with st.expander("Dataset Statistics"):
    if df.empty:
        st.warning("No data available to show statistics.")
    else:
        st.write(f"Total records: {len(df)}")
        
        # Only try to display date range if DATE column exists and has data
        if 'DATE' in df.columns and not df['DATE'].empty:
            try:
                min_date = pd.to_datetime(df['DATE']).min().date()
                max_date = pd.to_datetime(df['DATE']).max().date()
                st.write(f"Date range: {min_date} to {max_date}")
            except Exception as e:
                st.error(f"Error processing date range: {str(e)}")
        
        # Only try to display magnitude range if MAGNITUDE column exists and has data
        if 'MAGNITUDE' in df.columns and not df['MAGNITUDE'].empty:
            try:
                min_mag = df['MAGNITUDE'].min()
                max_mag = df['MAGNITUDE'].max()
                st.write(f"Magnitude range: {min_mag:.1f} to {max_mag:.1f}")
            except Exception as e:
                st.error(f"Error processing magnitude range: {str(e)}")
        
        # Display sample of the data
        st.dataframe(df.head())

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Zipcode Risk Checker", "Magnitude Predictor", "Insights"])

# Tab 1: Heatmap
with tab1:
    st.header("Earthquake Heatmap - North America")
    
    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        min_magnitude = st.slider("Minimum Magnitude", 
                                 float(df['MAGNITUDE'].min()), 
                                 float(df['MAGNITUDE'].max()), 
                                 2.5)
    with col2:
        # Convert DATE to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
            df['DATE'] = pd.to_datetime(df['DATE'])
            
        # Get min and max dates
        min_date = df['DATE'].min().date()
        max_date = df['DATE'].max().date()
        
        # Date range picker
        date_range = st.date_input("Date Range", 
                                  [min_date, max_date],
                                  min_value=min_date,
                                  max_value=max_date)
    
    # Filter data based on user selections
    filtered_df = df[df['MAGNITUDE'] >= min_magnitude]
    
    # Apply date filter if both dates are selected
    if len(date_range) == 2:
        start_date, end_date = date_range
        # Convert to datetime and ensure time component for proper filtering
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        
        # Apply date filter
        filtered_df = filtered_df[(filtered_df['DATE'] >= start_datetime) & 
                                  (filtered_df['DATE'] <= end_datetime)]
        
        # Show how many earthquakes match the filters
        st.write(f"Showing {len(filtered_df)} earthquakes from {start_date} to {end_date}")
    
    # Create heatmap using Plotly
    if len(filtered_df) > 0:
        fig = px.density_mapbox(filtered_df, 
                               lat='LATITUDE', 
                               lon='LONGITUDE', 
                               z='MAGNITUDE', 
                               radius=10,
                               center=dict(lat=39.8, lon=-98.5), 
                               zoom=3,
                               mapbox_style="carto-positron",
                               title="Earthquake Intensity Map",
                               color_continuous_scale="Viridis")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No earthquakes match your filter criteria. Please adjust the filters.")
    
    # Additional statistics
    st.subheader("Earthquake Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Earthquakes", len(filtered_df))
    with col2:
        st.metric("Average Magnitude", round(filtered_df['MAGNITUDE'].mean(), 2) if len(filtered_df) > 0 else 0)
    with col3:
        st.metric("Maximum Magnitude", round(filtered_df['MAGNITUDE'].max(), 2) if len(filtered_df) > 0 else 0)

# Tab 2: Zipcode Risk Checker
with tab2:
    st.header("Zipcode Risk Checker")
    
    # Cache the geocoding function to avoid repeated API calls
    @st.cache_data
    def get_coordinates_from_zipcode(zipcode):
        try:
            geolocator = Nominatim(user_agent="earthquake_app")
            location = geolocator.geocode(f"{zipcode}, USA")
            if location:
                return location.latitude, location.longitude
            else:
                return None
        except:
            return None
    
    # Function to calculate risk score based on nearby earthquakes
    def calculate_risk_score(lat, lon, earthquake_df):
        # Define parameters for risk calculation
        max_distance_km = 100  # Consider earthquakes within this distance
        time_weight = 0.2  # Weight for recency of earthquakes (reduced)
        magnitude_weight = 0.8  # Weight for earthquake magnitude (increased)
        
        # Make a copy to avoid SettingWithCopyWarning
        local_df = earthquake_df.copy()
        
        # Calculate distance from zipcode to each earthquake
        local_df['DISTANCE_KM'] = local_df.apply(
            lambda row: geodesic((lat, lon), (row['LATITUDE'], row['LONGITUDE'])).kilometers,
            axis=1
        )
        
        # Filter earthquakes within the specified distance
        nearby_earthquakes = local_df[local_df['DISTANCE_KM'] <= max_distance_km]
        
        if len(nearby_earthquakes) == 0:
            return 0, pd.DataFrame(), {}  # No risk if no nearby earthquakes
        
        # Convert dates to datetime for time-based weighting
        nearby_earthquakes['DATE'] = pd.to_datetime(nearby_earthquakes['DATE'])
        
        # Calculate days since the earthquake (older earthquakes have less impact)
        max_date = nearby_earthquakes['DATE'].max()
        nearby_earthquakes['DAYS_AGO'] = (max_date - nearby_earthquakes['DATE']).dt.days
        max_days = nearby_earthquakes['DAYS_AGO'].max() if nearby_earthquakes['DAYS_AGO'].max() > 0 else 1
        
        # Normalize days (0 = most recent, 1 = oldest)
        nearby_earthquakes['TIME_FACTOR'] = 1 - (nearby_earthquakes['DAYS_AGO'] / max_days)
        
        # Enhanced magnitude calculations
        avg_magnitude = nearby_earthquakes['MAGNITUDE'].mean()
        max_magnitude = nearby_earthquakes['MAGNITUDE'].max()
        
        # Apply exponential weighting to magnitudes (higher magnitudes have exponentially more impact)
        nearby_earthquakes['MAG_FACTOR'] = nearby_earthquakes['MAGNITUDE'].apply(lambda m: np.exp(m/2) / np.exp(8/2))
        
        # Calculate weighted risk factor for each earthquake with more emphasis on magnitude
        nearby_earthquakes['RISK_FACTOR'] = (
            time_weight * nearby_earthquakes['TIME_FACTOR'] +
            magnitude_weight * nearby_earthquakes['MAG_FACTOR']
        ) * (1 - nearby_earthquakes['DISTANCE_KM'] / max_distance_km)
        
        # Calculate risk score based on magnitude distribution and factors
        # We put more emphasis on max magnitude and less on earthquake count
        count_factor = min(1.0, np.log10(len(nearby_earthquakes)) / 3)  # Logarithmic scaling of count
        magnitude_factor = max_magnitude / 10  # Normalized magnitude (assume max possible is ~10)
        
        # Overall risk score (0-100)
        risk_score = min(100, (
            (magnitude_factor * 0.6) +  # Max magnitude contributes 60%
            (avg_magnitude / 10 * 0.3) +  # Average magnitude contributes 30%
            (count_factor * 0.1)  # Number of earthquakes contributes only 10%
        ) * 100)
        
        # Sort by most relevant (highest magnitude and closest)
        nearby_earthquakes = nearby_earthquakes.sort_values(
            by=['MAGNITUDE', 'DISTANCE_KM'], 
            ascending=[False, True]
        )
        
        # Generate additional statistics
        stats = {
            'max_magnitude': max_magnitude,
            'avg_magnitude': avg_magnitude,
            'earthquake_count': len(nearby_earthquakes),
            'significant_count': len(nearby_earthquakes[nearby_earthquakes['MAGNITUDE'] >= 5.0]),
            'moderate_count': len(nearby_earthquakes[(nearby_earthquakes['MAGNITUDE'] >= 3.0) & (nearby_earthquakes['MAGNITUDE'] < 5.0)]),
            'minor_count': len(nearby_earthquakes[nearby_earthquakes['MAGNITUDE'] < 3.0]),
            'recent_count': len(nearby_earthquakes[nearby_earthquakes['DAYS_AGO'] <= 365]),
            'mag_std': nearby_earthquakes['MAGNITUDE'].std(),
            'magnitude_distribution': nearby_earthquakes['MAGNITUDE'].value_counts(bins=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10]).to_dict()
        }
        
        return risk_score, nearby_earthquakes, stats
    
    # Input for zipcode
    zipcode = st.text_input("Enter Zipcode")
    
    if st.button("Check Risk"):
        if zipcode:
            # Try to get coordinates from zipcode
            with st.spinner("Looking up location..."):
                coords = get_coordinates_from_zipcode(zipcode)
            
            if coords:
                lat, lon = coords
                st.success(f"Location found: {lat:.4f}, {lon:.4f}")
                
                # Calculate risk score and get nearby earthquakes
                with st.spinner("Analyzing earthquake risk..."):
                    risk_score, nearby_earthquakes, stats = calculate_risk_score(lat, lon, df)
                
                # Display risk assessment
                st.subheader("Risk Assessment")
                
                risk_color = "green" if risk_score < 30 else "orange" if risk_score < 70 else "red"
                st.markdown(f"<h2 style='text-align: center; color: {risk_color};'>Risk Score: {risk_score:.1f}/100</h2>", unsafe_allow_html=True)
                
                if risk_score < 30:
                    st.success("Low Risk Area - This location has relatively low seismic activity.")
                elif risk_score < 70:
                    st.warning("Moderate Risk Area - This location has notable seismic activity.")
                else:
                    st.error("High Risk Area - This location has significant seismic activity.")
                
                # Display magnitude statistics
                st.subheader("Magnitude Statistics")
                
                # Metrics row
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Max Magnitude", f"{stats['max_magnitude']:.1f}")
                with col2:
                    st.metric("Average Magnitude", f"{stats['avg_magnitude']:.1f}")
                with col3:
                    st.metric("Earthquake Count", stats['earthquake_count'])
                with col4:
                    st.metric("Recent (1 year)", stats['recent_count'])
                
                # Magnitude category breakdown
                st.subheader("Earthquake Categories")
                categories = {
                    'Significant (≥5.0)': stats['significant_count'],
                    'Moderate (3.0-4.9)': stats['moderate_count'],
                    'Minor (<3.0)': stats['minor_count']
                }
                
                # Create a bar chart of magnitude categories
                if nearby_earthquakes.empty:
                    st.info("No earthquakes found within 100km of this location.")
                else:
                    # Earthquake category bar chart
                    category_df = pd.DataFrame({
                        'Category': list(categories.keys()),
                        'Count': list(categories.values())
                    })
                    
                    fig_cat = px.bar(
                        category_df,
                        x='Category',
                        y='Count',
                        color='Category',
                        color_discrete_map={
                            'Significant (≥5.0)': 'red',
                            'Moderate (3.0-4.9)': 'orange',
                            'Minor (<3.0)': 'green'
                        },
                        title="Earthquake Magnitude Categories"
                    )
                    st.plotly_chart(fig_cat, use_container_width=True)
                    
                    # Create histogram of magnitudes
                    fig_mag_hist = px.histogram(
                        nearby_earthquakes,
                        x='MAGNITUDE',
                        nbins=20,
                        color_discrete_sequence=['blue'],
                        title="Distribution of Earthquake Magnitudes"
                    )
                    fig_mag_hist.update_layout(xaxis_title="Magnitude", yaxis_title="Count")
                    st.plotly_chart(fig_mag_hist, use_container_width=True)
                
                # Display map with zipcode and nearby earthquakes
                if not nearby_earthquakes.empty:
                    st.subheader("Earthquake Map")
                    
                    # Create dataframe for the zipcode location
                    zipcode_df = pd.DataFrame({
                        'lat': [lat],
                        'lon': [lon],
                        'type': ['Your Location']
                    })
                    
                    # Prepare earthquake data for map
                    map_data = nearby_earthquakes.head(50).copy()
                    map_data['type'] = 'Earthquake'
                    map_data = map_data.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
                    
                    # Combine location and earthquake data
                    combined_data = pd.concat([
                        zipcode_df,
                        map_data[['lat', 'lon', 'type', 'MAGNITUDE', 'DISTANCE_KM', 'DATE']]
                    ])
                    
                    # Create map
                    fig = px.scatter_mapbox(
                        combined_data,
                        lat='lat',
                        lon='lon',
                        color='type',
                        size=combined_data['MAGNITUDE'].fillna(3) if 'MAGNITUDE' in combined_data else None,
                        hover_data=['MAGNITUDE', 'DISTANCE_KM', 'DATE'] if 'MAGNITUDE' in combined_data else None,
                        zoom=8,
                        height=500,
                        mapbox_style="carto-positron"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display nearby earthquake history
                    st.subheader("Nearby Earthquake History")
                    
                    if not nearby_earthquakes.empty:
                        display_df = nearby_earthquakes.head(10)[
                            ['DATE', 'PLACE', 'MAGNITUDE', 'DEPTH', 'DISTANCE_KM']
                        ].reset_index(drop=True)
                        display_df['DISTANCE_KM'] = display_df['DISTANCE_KM'].round(2)
                        st.dataframe(display_df)
                    else:
                        st.info("No earthquakes found within 100km of this location.")
            else:
                st.error("Could not find coordinates for this zipcode. Please check the zipcode and try again.")
        else:
            st.warning("Please enter a zipcode")

# Tab 3: Magnitude Predictor using Random Forest
with tab3:
    st.header("Earthquake Magnitude Predictor")
    st.write("Predict potential earthquake magnitude for a location using Advanced ML Models")
    
    # Performance optimization options
    with st.expander("Model Training Options"):
        # Select model type BEFORE the cached function
        model_type = st.radio(
            "Select Model Type",
            options=["Random Forest", "Gradient Boosting"],
            index=0
        )
        
        # Add options to control training performance
        col1, col2 = st.columns(2)
        with col1:
            use_sample = st.checkbox("Use data sampling for faster training", value=True)
            if use_sample:
                sample_size = st.slider("Sample size", 1000, 10000, 5000, step=1000)
            
            n_iter = st.slider("Number of hyperparameter combinations to try", 3, 20, 5)
            cv_folds = st.slider("Cross-validation folds", 2, 5, 3)
        
        with col2:
            max_trees = st.slider("Maximum number of trees/estimators", 50, 300, 100)
            cache_model = st.checkbox("Cache trained model", value=True)
            
            # Add option to load pre-trained model if it exists
            model_filename = f"assets/{model_type.lower().replace(' ', '_')}_model.joblib"
            if os.path.exists(model_filename) and cache_model:
                use_cached = st.checkbox("Use previously trained model if available", value=True)
            else:
                use_cached = False
    
    # Function to train and save Advanced ML model
    @st.cache_resource(ttl=3600, show_spinner=False)  # Cache for 1 hour
    def train_advanced_model(df, model_type, use_sample=True, sample_size=5000, 
                            n_iter=5, cv_folds=3, max_trees=100):
        # Try to load cached model if available and requested
        model_filename = f"assets/{model_type.lower().replace(' ', '_')}_model.joblib"
        if os.path.exists(model_filename) and use_cached:
            try:
                st.info(f"Loading pre-trained {model_type} model...")
                return joblib.load(model_filename)
            except:
                st.warning("Could not load cached model. Training a new one...")
        
        # Prepare the dataset
        # Converting DATE to datetime
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        # Feature engineering
        df['YEAR'] = df['DATE'].dt.year
        df['MONTH'] = df['DATE'].dt.month
        df['DAY'] = df['DATE'].dt.day
        df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek
        
        # Select essential features only
        numeric_features = ['LATITUDE', 'LONGITUDE', 'DEPTH', 'YEAR', 'MONTH']
        
        # Add only the most important additional features if they exist
        important_features = ['GAP', 'DMIN', 'NST', 'MAGNST']
        
        for feature in important_features:
            if feature in df.columns:
                numeric_features.append(feature)
        
        # Only include MAGTYPE as categorical feature (skip STATE to reduce dimensionality)
        categorical_features = []
        if 'MAGTYPE' in df.columns:
            categorical_features.append('MAGTYPE')
        
        # Display selected features
        st.write(f"Training with {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
        
        # Filter out rows with missing values in key columns
        key_features = ['LATITUDE', 'LONGITUDE', 'DEPTH', 'MAGNITUDE']
        model_df = df.dropna(subset=key_features)
        
        # Use sampling for faster training if requested
        if use_sample and len(model_df) > sample_size:
            # Keep all high magnitude samples (important for prediction)
            high_mag_threshold = model_df['MAGNITUDE'].quantile(0.9)
            high_mag_df = model_df[model_df['MAGNITUDE'] >= high_mag_threshold]
            
            # Sample from the rest
            low_mag_df = model_df[model_df['MAGNITUDE'] < high_mag_threshold]
            if len(low_mag_df) > sample_size - len(high_mag_df):
                low_mag_df = low_mag_df.sample(n=sample_size - len(high_mag_df), random_state=42)
            
            # Combine
            model_df = pd.concat([high_mag_df, low_mag_df])
            st.info(f"Using {len(model_df)} samples for training ({len(high_mag_df)} high magnitude events)")
        
        # Fill missing values in numeric columns
        for feature in numeric_features:
            if feature in model_df.columns:
                model_df[feature] = model_df[feature].fillna(model_df[feature].median())
        
        # For categorical features, fill with the most common value
        for feature in categorical_features:
            if feature in model_df.columns:
                model_df[feature] = model_df[feature].fillna(model_df[feature].mode()[0])
        
        # Split the data
        X = model_df[numeric_features + categorical_features]
        y = model_df['MAGNITUDE']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        preprocessor_steps = [
            ('num', numeric_transformer, numeric_features)
        ]
        
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            preprocessor_steps.append(('cat', categorical_transformer, categorical_features))
        
        preprocessor = ColumnTransformer(transformers=preprocessor_steps)
        
        # Set up model and parameters based on model_type (which is now passed in)
        if model_type == "Random Forest":
            # Simplified hyperparameter tuning for Random Forest
            param_dist = {
                'n_estimators': [max(50, max_trees // 2), max_trees],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
            
            base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        else:  # Gradient Boosting
            # Simplified hyperparameter tuning for Gradient Boosting
            param_dist = {
                'n_estimators': [max(50, max_trees // 2), max_trees],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
            
            base_model = GradientBoostingRegressor(random_state=42)
        
        # Create full pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', base_model)
        ])
        
        # Set up RandomizedSearchCV with reduced iterations and CV folds
        search = RandomizedSearchCV(
            full_pipeline,
            param_distributions={'model__' + key: value for key, value in param_dist.items()},
            n_iter=n_iter,  # Reduced number of iterations
            cv=cv_folds,    # Reduced cross-validation
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Track time for better user feedback
        start_time = time.time()
        status_text.text("Fitting model...")
        
        # Actual training
        search.fit(X_train, y_train)
        
        # Training completed
        elapsed_time = time.time() - start_time
        status_text.text(f"Model training complete! Time elapsed: {elapsed_time:.1f} seconds")
        progress_bar.progress(100)
        
        # Get the best model
        best_model = search.best_estimator_
        
        # Evaluate the model
        train_pred = best_model.predict(X_train)
        test_pred = best_model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Feature importances (only for the model, not the preprocessor)
        if model_type == "Random Forest":
            importances = best_model.named_steps['model'].feature_importances_
        else:  # Gradient Boosting
            importances = best_model.named_steps['model'].feature_importances_
        
        # Get feature names after preprocessing (one-hot encoding changes the names)
        if categorical_features:
            feature_names = (
                numeric_features +
                list(best_model.named_steps['preprocessor']
                    .transformers_[1][1]
                    .named_steps['onehot']
                    .get_feature_names_out(categorical_features))
            )
        else:
            feature_names = numeric_features
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': feature_names[:len(importances)],  # Ensure same length as importances
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Get best parameters
        best_params = search.best_params_
        
        # Save the model if caching is requested
        if cache_model:
            # Ensure assets directory exists
            if not os.path.exists('assets'):
                os.makedirs('assets')
            # Save model to disk
            joblib.dump({
                'model': best_model,
                'features': {
                    'numeric': numeric_features,
                    'categorical': categorical_features
                },
                'metrics': {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'test_mae': test_mae,
                    'test_r2': test_r2
                },
                'feature_importance': feature_importance,
                'best_params': best_params
            }, model_filename)
        
        return {
            'model': best_model,
            'features': {
                'numeric': numeric_features,
                'categorical': categorical_features
            },
            'metrics': {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2
            },
            'feature_importance': feature_importance,
            'best_params': best_params,
            'training_time': elapsed_time
        }
    
    # Train the model - pass the model_type from outside
    with st.spinner("Training model (this may take a few moments)..."):
        model_data = train_advanced_model(
            df, 
            model_type, 
            use_sample=use_sample, 
            sample_size=sample_size if 'sample_size' in locals() else 5000,
            n_iter=n_iter,
            cv_folds=cv_folds,
            max_trees=max_trees
        )
    
    # Display model metrics
    st.subheader("Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training RMSE", f"{model_data['metrics']['train_rmse']:.3f}")
    with col2:
        st.metric("Testing RMSE", f"{model_data['metrics']['test_rmse']:.3f}")
    with col3:
        st.metric("Testing MAE", f"{model_data['metrics']['test_mae']:.3f}")
    with col4:
        st.metric("R² Score", f"{model_data['metrics']['test_r2']:.3f}")
    
    # Show training time if available
    if 'training_time' in model_data:
        st.info(f"Model training completed in {model_data['training_time']:.1f} seconds")
    
    # Display best parameters
    st.subheader("Best Model Parameters")
    # Extract model parameters without the 'model__' prefix
    best_params = {k.replace('model__', ''): v for k, v in model_data['best_params'].items()}
    st.json(best_params)
    
    # Feature importance plot
    st.subheader("Feature Importance")
    fig_importance = px.bar(
        model_data['feature_importance'].head(15),  # Show top 15 features
        x='Importance',
        y='Feature',
        orientation='h',
        title="Feature Importance for Magnitude Prediction"
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Prediction form
    st.subheader("Predict Earthquake Magnitude")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_method = st.radio(
            "Input Method",
            options=["Zipcode", "Coordinates"],
            index=0
        )
        
    with col2:
        current_year = pd.Timestamp.now().year
        current_month = pd.Timestamp.now().month
        
        pred_year = st.number_input(
            "Year",
            min_value=current_year,
            max_value=current_year + 10,
            value=current_year
        )
        
        pred_month = st.slider(
            "Month",
            min_value=1,
            max_value=12,
            value=current_month
        )
    
    # Get location input
    if prediction_method == "Zipcode":
        pred_zipcode = st.text_input("Enter Zipcode for Prediction")
        pred_button = st.button("Predict Magnitude")
        
        if pred_button:
            if pred_zipcode:
                # Get coordinates from zipcode
                with st.spinner("Looking up location..."):
                    coords = get_coordinates_from_zipcode(pred_zipcode)
                
                if coords:
                    lat, lon = coords
                    st.success(f"Location found: {lat:.4f}, {lon:.4f}")
                    
                    # Get depth data from nearby earthquakes
                    nearby_df = df.copy()
                    nearby_df['DISTANCE'] = nearby_df.apply(
                        lambda row: geodesic((lat, lon), (row['LATITUDE'], row['LONGITUDE'])).kilometers,
                        axis=1
                    )
                    nearby_df = nearby_df.sort_values('DISTANCE').head(50)
                    avg_depth = nearby_df['DEPTH'].mean()
                    
                    # Prepare input for prediction
                    input_data = pd.DataFrame({
                        'LATITUDE': [lat],
                        'LONGITUDE': [lon],
                        'DEPTH': [avg_depth],
                        'YEAR': [pred_year],
                        'MONTH': [pred_month],
                        'DAY': [15],  # Middle of the month as default
                        'DAY_OF_WEEK': [pd.Timestamp(year=pred_year, month=pred_month, day=15).dayofweek]
                    })
                    
                    # Add additional features if they were used in training
                    for feature in model_data['features']['numeric']:
                        if feature not in input_data.columns:
                            # For missing features, use the median from nearby earthquakes if available
                            if feature in nearby_df.columns:
                                input_data[feature] = nearby_df[feature].median()
                            else:
                                input_data[feature] = 0  # Default value if not available
                    
                    # Add categorical features if they were used
                    for feature in model_data['features']['categorical']:
                        if feature == 'MAGTYPE':
                            # Use most common MAGTYPE from nearby earthquakes
                            if 'MAGTYPE' in nearby_df.columns:
                                input_data['MAGTYPE'] = nearby_df['MAGTYPE'].mode()[0] if not nearby_df.empty else 'ml'
                            else:
                                input_data['MAGTYPE'] = 'ml'  # Default
                        elif feature == 'STATE':
                            # Try to determine state from zipcode or use most common from nearby
                            if 'STATE' in nearby_df.columns:
                                input_data['STATE'] = nearby_df['STATE'].mode()[0] if not nearby_df.empty else 'California'
                            else:
                                input_data['STATE'] = 'California'  # Default
                    
                    # Make prediction
                    predicted_magnitude = model_data['model'].predict(input_data)[0]
                    
                    # Display prediction
                    st.subheader("Prediction Result")
                    magnitude_color = "green" if predicted_magnitude < 4.0 else "orange" if predicted_magnitude < 6.0 else "red"
                    st.markdown(f"<h1 style='text-align: center; color: {magnitude_color};'>{predicted_magnitude:.2f}</h1>", unsafe_allow_html=True)
                    
                    # Interpret prediction
                    if predicted_magnitude < 2.5:
                        st.success("Predicted magnitude is very low. Earthquakes of this size are usually not felt.")
                    elif predicted_magnitude < 4.0:
                        st.success("Predicted magnitude is low. Such earthquakes are often felt but rarely cause damage.")
                    elif predicted_magnitude < 6.0:
                        st.warning("Predicted magnitude is moderate. Such earthquakes can cause damage to poorly constructed buildings.")
                    else:
                        st.error("Predicted magnitude is high. Such earthquakes can cause serious damage over large areas.")
                    
                    # Show map with prediction location
                    pred_location_df = pd.DataFrame({
                        'lat': [lat],
                        'lon': [lon],
                        'Predicted_Magnitude': [predicted_magnitude]
                    })
                    
                    fig = px.scatter_mapbox(
                        pred_location_df,
                        lat='lat',
                        lon='lon',
                        size=[30],
                        color_discrete_sequence=["red"],
                        zoom=8,
                        height=400,
                        width=700,
                        mapbox_style="carto-positron"
                    )
                    
                    st.plotly_chart(fig)
                    
                    # Show nearby historical earthquakes
                    st.subheader("Historical Earthquakes in this Area")
                    st.dataframe(nearby_df[['DATE', 'PLACE', 'MAGNITUDE', 'DEPTH', 'DISTANCE']].head(10).reset_index(drop=True))
                else:
                    st.error("Could not find coordinates for this zipcode. Please check the zipcode and try again.")
            else:
                st.warning("Please enter a zipcode")
    else:
        # Coordinate input
        col1, col2 = st.columns(2)
        with col1:
            pred_lat = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=37.7749)
        with col2:
            pred_lon = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-122.4194)
        
        # Get depth data
        depth_method = st.radio(
            "Depth Estimation Method",
            options=["Use nearby earthquakes' average depth", "Specify depth manually"],
            index=0
        )
        
        if depth_method == "Specify depth manually":
            pred_depth = st.number_input("Depth (km)", min_value=0.0, max_value=100.0, value=10.0)
        
        pred_button = st.button("Predict Magnitude")
        
        if pred_button:
            # Get depth data if using nearby earthquakes method
            nearby_df = df.copy()
            nearby_df['DISTANCE'] = nearby_df.apply(
                lambda row: geodesic((pred_lat, pred_lon), (row['LATITUDE'], row['LONGITUDE'])).kilometers,
                axis=1
            )
            nearby_df = nearby_df.sort_values('DISTANCE').head(50)
            
            if depth_method == "Use nearby earthquakes' average depth":
                pred_depth = nearby_df['DEPTH'].mean()
            
            # Prepare input for prediction
            input_data = pd.DataFrame({
                'LATITUDE': [pred_lat],
                'LONGITUDE': [pred_lon],
                'DEPTH': [pred_depth],
                'YEAR': [pred_year],
                'MONTH': [pred_month],
                'DAY': [15],  # Middle of the month as default
                'DAY_OF_WEEK': [pd.Timestamp(year=pred_year, month=pred_month, day=15).dayofweek]
            })
            
            # Add additional features if they were used in training
            for feature in model_data['features']['numeric']:
                if feature not in input_data.columns:
                    # For missing features, use the median from nearby earthquakes if available
                    if feature in nearby_df.columns:
                        input_data[feature] = nearby_df[feature].median()
                    else:
                        input_data[feature] = 0  # Default value if not available
            
            # Add categorical features if they were used
            for feature in model_data['features']['categorical']:
                if feature == 'MAGTYPE':
                    # Use most common MAGTYPE from nearby earthquakes
                    if 'MAGTYPE' in nearby_df.columns:
                        input_data['MAGTYPE'] = nearby_df['MAGTYPE'].mode()[0] if not nearby_df.empty else 'ml'
                    else:
                        input_data['MAGTYPE'] = 'ml'  # Default
                elif feature == 'STATE':
                    # Try to determine state from location or use most common from nearby
                    if 'STATE' in nearby_df.columns:
                        input_data['STATE'] = nearby_df['STATE'].mode()[0] if not nearby_df.empty else 'California'
                    else:
                        input_data['STATE'] = 'California'  # Default
            
            # Make prediction
            predicted_magnitude = model_data['model'].predict(input_data)[0]
            
            # Display prediction
            st.subheader("Prediction Result")
            magnitude_color = "green" if predicted_magnitude < 4.0 else "orange" if predicted_magnitude < 6.0 else "red"
            st.markdown(f"<h1 style='text-align: center; color: {magnitude_color};'>{predicted_magnitude:.2f}</h1>", unsafe_allow_html=True)
            
            # Interpret prediction
            if predicted_magnitude < 2.5:
                st.success("Predicted magnitude is very low. Earthquakes of this size are usually not felt.")
            elif predicted_magnitude < 4.0:
                st.success("Predicted magnitude is low. Such earthquakes are often felt but rarely cause damage.")
            elif predicted_magnitude < 6.0:
                st.warning("Predicted magnitude is moderate. Such earthquakes can cause damage to poorly constructed buildings.")
            else:
                st.error("Predicted magnitude is high. Such earthquakes can cause serious damage over large areas.")
            
            # Show map with prediction location
            pred_location_df = pd.DataFrame({
                'lat': [pred_lat],
                'lon': [pred_lon],
                'Predicted_Magnitude': [predicted_magnitude]
            })
            
            fig = px.scatter_mapbox(
                pred_location_df,
                lat='lat',
                lon='lon',
                size=[30],
                color_discrete_sequence=["red"],
                zoom=8,
                height=400,
                width=700,
                mapbox_style="carto-positron"
            )
            
            st.plotly_chart(fig)
            
            # Show nearby historical earthquakes
            st.subheader("Historical Earthquakes in this Area")
            st.dataframe(nearby_df[['DATE', 'PLACE', 'MAGNITUDE', 'DEPTH', 'DISTANCE']].head(10).reset_index(drop=True))

# Tab 4: Insights
with tab4:
    st.header("Earthquake Insights")
    
    # Time series analysis
    st.subheader("Earthquake Frequency Over Time")
    # Convert DATE to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    # Group by month and count earthquakes
    monthly_quakes = df.resample('M', on='DATE').size().reset_index(name='count')
    
    # Plot time series
    fig_time = px.line(monthly_quakes, x='DATE', y='count', 
                      title='Monthly Earthquake Frequency')
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Magnitude distribution
    st.subheader("Magnitude Distribution")
    fig_hist = px.histogram(df, x='MAGNITUDE', nbins=50,
                           title='Distribution of Earthquake Magnitudes')
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # State comparison (if available)
    if 'STATE' in df.columns:
        st.subheader("Earthquake Frequency by State")
        state_counts = df['STATE'].value_counts().reset_index()
        state_counts.columns = ['State', 'Count']
        state_counts = state_counts.sort_values('Count', ascending=False).head(10)
        
        fig_bar = px.bar(state_counts, x='State', y='Count',
                         title='Top 10 States by Earthquake Frequency')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlation")
    numeric_df = df.select_dtypes(include=[np.number])
    # Remove columns with all NaN values
    numeric_df = numeric_df.dropna(axis=1, how='all')
    # Fill remaining NaN values with column means
    numeric_df = numeric_df.fillna(numeric_df.mean())
    
    corr = numeric_df.corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                        title="Correlation Between Features")
    st.plotly_chart(fig_corr, use_container_width=True) 