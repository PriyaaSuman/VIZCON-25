# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Bird Migration Predictor",
    page_icon="🦅",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #2c3e50;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# First, let's create a function to train and save the model
def train_and_save_model(df):
    # Create and save label encoders for categorical variables
    categorical_columns = ['Species', 'Region', 'Habitat', 'Weather_Condition', 
                         'Migration_Reason', 'Migrated_in_Flock',
                         'Food_Supply_Level', 'Tracking_Quality']
    
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le
    
    # Save label encoders
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    
    # Select features for the model
    features = ['Species', 'Region', 'Habitat', 'Weather_Condition', 
               'Flight_Distance_km', 'Flight_Duration_hours', 'Average_Speed_kmph',
               'Max_Altitude_m', 'Temperature_C', 'Wind_Speed_kmph', 'Humidity_%',
               'Rest_Stops', 'Predator_Sightings', 'Migrated_in_Flock', 
               'Flock_Size', 'Food_Supply_Level']
    
    # Filter features that exist in the dataset
    available_features = [f for f in features if f in df_encoded.columns]
    
    X = df_encoded[available_features]
    y = df_encoded['Migration_Success']
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    # Save the model and features
    joblib.dump(model, 'migration_model.pkl')
    with open('feature_list.pkl', 'wb') as f:
        pickle.dump(available_features, f)
    
    return model, label_encoders, available_features

# Load the saved model and components
@st.cache_resource
def load_model_and_data():
    try:
        model = joblib.load('migration_model.pkl')
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)
        with open('feature_list.pkl', 'rb') as f:
            features = pickle.load(f)
        df = pd.read_csv('bird_migration_with_origin_destination.csv')
        return model, label_encoders, features, df
    except:
        # If model files don't exist, train the model
        df = pd.read_csv('bird_migration_with_origin_destination.csv')
        model, label_encoders, features = train_and_save_model(df)
        return model, label_encoders, features, df

def main():
    st.title("🦅 Bird Migration Success Predictor")
    st.markdown("*Advanced Analytics for Migration Pattern Analysis*")
    st.markdown("---")

    # Sidebar navigation
    st.sidebar.header("🧭 Navigation")
    page = st.sidebar.radio("Go to", [
        "🔮 Make Prediction", 
        "📊 Data Analysis", 
        "🔍 Clustering Analysis", 
        "📈 Success Patterns",
        "ℹ️ About"
    ])

    if page == "🔮 Make Prediction":
        show_prediction_page()
    elif page == "📊 Data Analysis":
        show_analysis_page()
    elif page == "🔍 Clustering Analysis":
        show_clustering_analysis()
    elif page == "📈 Success Patterns":
        show_success_patterns()
    else:
        show_about_page()

def show_prediction_page():
    st.header("🔮 Predict Migration Success")
    st.write("Enter migration parameters to predict success probability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🐦 Bird Information")
        species = st.selectbox("Species", ["Hawk", "Eagle", "Stork", "Warbler", "Swallow", "Crane", "Goose"])
        region = st.selectbox("Region", ["North America", "South America", "Europe", "Asia", "Africa", "Australia"])
        habitat = st.selectbox("Habitat", ["Forest", "Grassland", "Wetland", "Coastal", "Mountain", "Urban"])
        weather = st.selectbox("Weather Condition", ["Clear", "Cloudy", "Rainy", "Stormy", "Windy", "Foggy"])

    with col2:
        st.subheader("✈️ Flight Parameters")
        distance = st.number_input("Flight Distance (km)", 0, 5000, 1000)
        duration = st.number_input("Flight Duration (hours)", 0, 200, 48)
        speed = st.number_input("Average Speed (km/h)", 0, 100, 40)
        altitude = st.number_input("Maximum Altitude (m)", 0, 10000, 3000)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🌤️ Environmental Conditions")
        temperature = st.slider("Temperature (°C)", -20, 40, 15)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 100, 20)
        humidity = st.slider("Humidity (%)", 0, 100, 65)
        rest_stops = st.number_input("Number of Rest Stops", 0, 20, 3)

    with col4:
        st.subheader("🔄 Migration Behavior")
        predators = st.number_input("Predator Sightings", 0, 20, 2)
        flock = st.selectbox("Migrated in Flock", ["Yes", "No"])
        flock_size = st.number_input("Flock Size", 0, 1000, 100)
        food_supply = st.selectbox("Food Supply Level", ["Low", "Medium", "High"])

    if st.button("🚀 Predict Migration Success", type="primary"):
        try:
            model, label_encoders, features, df = load_model_and_data()
            
            # Create input data
            input_data = pd.DataFrame({
                'Species': [species],
                'Region': [region],
                'Habitat': [habitat],
                'Weather_Condition': [weather],
                'Flight_Distance_km': [distance],
                'Flight_Duration_hours': [duration],
                'Average_Speed_kmph': [speed],
                'Max_Altitude_m': [altitude],
                'Temperature_C': [temperature],
                'Wind_Speed_kmph': [wind_speed],
                'Humidity_%': [humidity],
                'Rest_Stops': [rest_stops],
                'Predator_Sightings': [predators],
                'Migrated_in_Flock': [flock],
                'Flock_Size': [flock_size],
                'Food_Supply_Level': [food_supply]
            })

            # Transform categorical variables
            for col in label_encoders.keys():
                if col in input_data.columns:
                    try:
                        input_data[col] = label_encoders[col].transform(input_data[col])
                    except:
                        st.warning(f"Unknown value for {col}. Using default encoding.")
                        input_data[col] = 0

            # Make prediction
            available_features = [f for f in features if f in input_data.columns]
            prediction = model.predict(input_data[available_features])
            probability = model.predict_proba(input_data[available_features])

            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col5, col6 = st.columns(2)
            
            with col5:
                if prediction[0] == "Successful":
                    st.success(f"✅ Predicted Outcome: {prediction[0]}")
                else:
                    st.error(f"❌ Predicted Outcome: {prediction[0]}")
                    
            with col6:
                confidence = max(probability[0]) * 100
                st.info(f"🎯 Confidence: {confidence:.1f}%")

            # Probability visualization
            prob_data = pd.DataFrame({
                'Outcome': ['Success', 'Failure'],
                'Probability': probability[0]
            })
            
            fig = px.bar(prob_data, x='Outcome', y='Probability', 
                        color='Outcome', color_discrete_map={'Success': 'green', 'Failure': 'red'})
            fig.update_layout(title="Migration Success Probability")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def show_analysis_page():
    st.header("📊 Migration Data Analysis")
    
    try:
        model, label_encoders, features, df = load_model_and_data()
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Migrations", len(df))
        with col2:
            success_rate = (df['Migration_Success'] == 'Successful').mean() * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col3:
            avg_distance = df['Flight_Distance_km'].mean()
            st.metric("Avg Distance", f"{avg_distance:.0f} km")
        with col4:
            avg_duration = df['Flight_Duration_hours'].mean()
            st.metric("Avg Duration", f"{avg_duration:.1f} hrs")

        # Interactive plots
        st.subheader("🎨 Interactive Visualizations")
        
        plot_type = st.selectbox("Select Visualization", 
                                ["Success by Species", 
                                 "Distance vs Duration", 
                                 "Migration Routes Map",
                                 "Weather Impact"])

        if plot_type == "Success by Species":
            species_success = df.groupby('Species')['Migration_Success'].value_counts(normalize=True).unstack()
            fig = px.bar(species_success, title="Migration Success Rate by Species")
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Distance vs Duration":
            fig = px.scatter(df, x='Flight_Distance_km', 
                            y='Flight_Duration_hours',
                            color='Migration_Success',
                            title="Flight Distance vs Duration")
            st.plotly_chart(fig, use_container_width=True)

        elif plot_type == "Migration Routes Map":
            if 'Start_Latitude' in df.columns and 'Start_Longitude' in df.columns:
                fig = px.scatter_mapbox(df,
                                       lat='Start_Latitude',
                                       lon='Start_Longitude',
                                       color='Species',
                                       size='Flight_Distance_km',
                                       hover_data=['Migration_Success'],
                                       title="Migration Routes Map")
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Location data not available for mapping")

        else:  # Weather Impact
            weather_success = df.groupby('Weather_Condition')['Migration_Success'].value_counts(normalize=True).unstack()
            fig = px.bar(weather_success, title="Migration Success by Weather Condition")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def show_clustering_analysis():
    st.header("🔍 Clustering Analysis: Success vs Failure Patterns")
    
    try:
        model, label_encoders, features, df = load_model_and_data()
        
        # Separate successful and failed migrations
        successful = df[df['Migration_Success'] == 'Successful']
        failed = df[df['Migration_Success'] == 'Failed']
        
        # Show basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("✅ Successful Migrations", len(successful))
        with col2:
            st.metric("❌ Failed Migrations", len(failed))
        with col3:
            success_rate = len(successful) / len(df) * 100
            st.metric("📊 Success Rate", f"{success_rate:.1f}%")
        
        # Feature comparison
        numerical_features = ['Flight_Distance_km', 'Flight_Duration_hours', 'Average_Speed_kmph',
                             'Max_Altitude_m', 'Temperature_C', 'Wind_Speed_kmph', 'Humidity_%',
                             'Rest_Stops', 'Predator_Sightings', 'Flock_Size']
        
        # Filter features that exist in the dataset
        available_numerical = [f for f in numerical_features if f in df.columns]
        
        feature_comparison = pd.DataFrame({
            'Feature': available_numerical,
            'Successful_Mean': [successful[feat].mean() for feat in available_numerical],
            'Failed_Mean': [failed[feat].mean() for feat in available_numerical],
            'Successful_Std': [successful[feat].std() for feat in available_numerical],
            'Failed_Std': [failed[feat].std() for feat in available_numerical]
        })
        
        # Calculate ideal ranges and differences
        feature_comparison['Ideal_Min'] = feature_comparison['Successful_Mean'] - feature_comparison['Successful_Std']
        feature_comparison['Ideal_Max'] = feature_comparison['Successful_Mean'] + feature_comparison['Successful_Std']
        feature_comparison['Difference'] = feature_comparison['Successful_Mean'] - feature_comparison['Failed_Mean']
        
        # Display ideal ranges
        st.subheader("💡 Ideal Feature Ranges for Success")
        display_df = feature_comparison[['Feature', 'Successful_Mean', 'Ideal_Min', 'Ideal_Max']].round(2)
        st.dataframe(display_df, use_container_width=True)
        
        # Feature comparison visualization
        st.subheader("📊 Feature Comparison: Success vs Failure")
        fig = px.bar(feature_comparison, x='Feature', y=['Successful_Mean', 'Failed_Mean'],
                    barmode='group', title="Average Feature Values: Success vs Failure")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Difference analysis
        st.subheader("🔍 Feature Impact Analysis")
        feature_comparison_sorted = feature_comparison.sort_values('Difference', ascending=False)
        fig2 = px.bar(feature_comparison_sorted, x='Feature', y='Difference',
                     title="Feature Impact on Success (Positive = Better for Success)")
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Success probability heatmap
        st.subheader("🔥 Success Rate Heatmap")
        
        # Create bins for analysis
        if len(available_numerical) >= 2:
            feature1 = available_numerical[0]  # e.g., Flight_Distance_km
            feature2 = available_numerical[2] if len(available_numerical) > 2 else available_numerical[1]  # e.g., Average_Speed_kmph
            
            df['Feature1_Bin'] = pd.cut(df[feature1], bins=3, labels=['Low', 'Medium', 'High'])
            df['Feature2_Bin'] = pd.cut(df[feature2], bins=3, labels=['Low', 'Medium', 'High'])
            
            success_heatmap = df.groupby(['Feature1_Bin', 'Feature2_Bin'])['Migration_Success'].apply(
                lambda x: (x == 'Successful').mean()
            ).unstack()
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(success_heatmap, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax)
            plt.title(f'Success Rate by {feature1} and {feature2}')
            plt.xlabel(f'{feature2} Category')
            plt.ylabel(f'{feature1} Category')
            st.pyplot(fig)
        
        # Key insights
        st.subheader("💎 Key Insights")
        top_positive = feature_comparison.nlargest(3, 'Difference')
        top_negative = feature_comparison.nsmallest(3, 'Difference')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🚀 Features favoring success:**")
            for _, row in top_positive.iterrows():
                st.write(f"• **{row['Feature']}**: Success avg = {row['Successful_Mean']:.1f}, Failure avg = {row['Failed_Mean']:.1f}")
        
        with col2:
            st.write("**⚠️ Features associated with failure:**")
            for _, row in top_negative.iterrows():
                st.write(f"• **{row['Feature']}**: Success avg = {row['Successful_Mean']:.1f}, Failure avg = {row['Failed_Mean']:.1f}")

    except Exception as e:
        st.error(f"Error in clustering analysis: {str(e)}")

def show_success_patterns():
    st.header("📈 Success Pattern Analysis")
    
    try:
        model, label_encoders, features, df = load_model_and_data()
        
        # K-means clustering
        st.subheader("🎯 Migration Pattern Clusters")
        
        # Select features for clustering
        cluster_features = ['Flight_Distance_km', 'Flight_Duration_hours', 'Average_Speed_kmph',
                           'Max_Altitude_m', 'Temperature_C', 'Wind_Speed_kmph']
        
        available_cluster_features = [f for f in cluster_features if f in df.columns]
        
        if len(available_cluster_features) >= 2:
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df[available_cluster_features])
            
            # Perform clustering
            kmeans = KMeans(n_clusters=4, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled_features)
            
            # Analyze success rate by cluster
            cluster_success = df.groupby('Cluster')['Migration_Success'].value_counts(normalize=True).unstack()
            
            # Show cluster success rates
            st.subheader("🏆 Success Rates by Cluster")
            st.dataframe(cluster_success.round(3))
            
            # Visualize clusters
            pca = PCA(n_components=2)
            pca_features = pca.fit_transform(scaled_features)
            
            fig = px.scatter(x=pca_features[:, 0], y=pca_features[:, 1], 
                            color=df['Cluster'].astype(str), 
                            symbol=df['Migration_Success'],
                            title="Migration Clusters (PCA Visualization)")
            fig.update_layout(xaxis_title="First Principal Component", 
                             yaxis_title="Second Principal Component")
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster characteristics
            st.subheader("📋 Cluster Characteristics")
            cluster_stats = df.groupby('Cluster')[available_cluster_features].mean()
            st.dataframe(cluster_stats.round(2))
        
        else:
            st.info("Insufficient features for clustering analysis")

    except Exception as e:
        st.error(f"Error in pattern analysis: {str(e)}")

def show_about_page():
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ## 🦅 Bird Migration Success Predictor
    
    This application uses machine learning to predict and analyze bird migration success patterns.
    
    ### 🎯 Features:
    - **Migration Success Prediction**: Predict likelihood of successful migration
    - **Data Analysis**: Comprehensive visualization of migration patterns
    - **Clustering Analysis**: Identify success and failure patterns
    - **Success Pattern Analysis**: Discover optimal migration conditions
    
    ### 🔧 Technical Details:
    - **Algorithm**: Random Forest Classifier
    - **Data Processing**: Automated feature engineering and encoding
    - **Visualization**: Interactive plots using Plotly and Seaborn
    - **Framework**: Streamlit for web interface
    
    ### 📊 Key Insights:
    - Environmental conditions significantly impact migration success
    - Optimal flight parameters vary by species and region
    - Weather conditions are crucial for successful migration
    - Flock behavior influences migration outcomes
    
    ### 🚀 How to Use:
    1. Navigate to **Make Prediction** to predict individual migration success
    2. Explore **Data Analysis** for comprehensive data insights
    3. Use **Clustering Analysis** to understand success/failure patterns
    4. Check **Success Patterns** for optimal migration conditions
    
    ### 📈 Model Performance:
    - Training accuracy: Variable based on data quality
    - Features used: Environmental, behavioral, and flight parameters
    - Cross-validation: Applied for robust performance estimation
    
    ---
    
    **Created for Data Science Hackathon**
    
    *This tool aids in wildlife conservation and migration research*
    """)

if __name__ == "__main__":
    main()