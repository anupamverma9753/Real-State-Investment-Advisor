import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


# Page Configuration

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom Styling

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        text-align: center;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)


# Load Data & Models (Cached for Performance)

@st.cache_data
def load_data():
    data_path = os.path.join("data", "processed", "cleaned_housing_prices.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("..", "data", "processed", "cleaned_housing_prices.csv")
    return pd.read_csv(data_path)

@st.cache_resource
def load_models():
    models_dir = "models"
    cls_path = os.path.join(models_dir, "best_classification_pipeline.pkl")
    reg_path = os.path.join(models_dir, "best_regression_pipeline.pkl")
    
    if not os.path.exists(cls_path):
        cls_path = os.path.join("..", models_dir, "best_classification_pipeline.pkl")
        reg_path = os.path.join("..", models_dir, "best_regression_pipeline.pkl")
        
    cls_pipeline = joblib.load(cls_path)
    reg_pipeline = joblib.load(reg_path)
    return cls_pipeline, reg_pipeline

try:
    df = load_data()
    cls_pipeline, reg_pipeline = load_models()
    data_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading data or model files: {e}")
    st.info("Please make sure you have run Phase 2 (Preprocessing) and Phase 4 (Model Training) first!")
    data_loaded = False


# Header Section

st.markdown("<div class='main-header'>Real Estate Investment Advisor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Predict Property Profitability, Investment Suitability % & 5-Year Future Valuation</div>", unsafe_allow_html=True)

if data_loaded:
    
    # Navigation Tabs
    
    tab1, tab2, tab3 = st.tabs([
        "Live Investment Predictor", 
        "Market Analytics Dashboard", 
        "Model Performance & Architecture"
    ])

   
    # TAB 1: LIVE INVESTMENT PREDICTOR
   
    with tab1:
        st.subheader(" Enter Property Parameters")
        
        # Form layout: 3 Columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#####  Location & Type")
            city = st.selectbox("City", sorted(df['City'].unique()))
            
            # Filter localities based on selected city
            available_localities = sorted(df[df['City'] == city]['Locality'].unique())
            locality = st.selectbox("Locality", available_localities)
            
            # Get state corresponding to city
            state_val = df[df['City'] == city]['State'].iloc[0] if 'State' in df.columns else "Unknown"
            st.text_input("State", value=state_val, disabled=True)
            
            property_type = st.selectbox("Property Type", sorted(df['Property_Type'].unique()))
            furnished_status = st.selectbox("Furnished Status", sorted(df['Furnished_Status'].unique())) if 'Furnished_Status' in df.columns else "Semi"

        with col2:
            st.markdown("#####  Size & Pricing")
            bhk = st.number_input("BHK (Bedrooms)", min_value=1, max_value=10, value=2, step=1)
            size_sqft = st.number_input("Size (in SqFt)", min_value=200, max_value=20000, value=1100, step=50)
            price_lakhs = st.number_input("Current Price (in Lakhs ₹)", min_value=5.0, max_value=5000.0, value=65.0, step=2.5)
            
            # Auto-calculate Price per SqFt for information
            calc_price_sqft = (price_lakhs * 100000) / size_sqft if size_sqft > 0 else 0
            st.info(f" Calculated Rate: **₹ {calc_price_sqft:.2f} / SqFt**")
            
            age_property = st.slider("Property Age (Years)", min_value=0, max_value=50, value=5)

        with col3:
            st.markdown("#####  Infrastructure & Amenities")
            schools_rating = st.slider("Nearby Schools Rating", min_value=1, max_value=5, value=3)
            hospitals_rating = st.slider("Nearby Hospitals Rating", min_value=1, max_value=5, value=3)
            transport_rating = st.slider("Public Transport Access", min_value=1, max_value=5, value=4)
            
            parking_spaces = st.selectbox("Parking Spaces", options=[0, 1, 2, 3, 4], index=1) if 'Parking_Space' in df.columns else 1
            facing = st.selectbox("Facing Direction", sorted(df['Facing'].unique())) if 'Facing' in df.columns else "East"
            owner_type = st.selectbox("Owner Type", sorted(df['Owner_Type'].unique())) if 'Owner_Type' in df.columns else "Owner"
            avail_status = st.selectbox("Availability", sorted(df['Availability_Status'].unique())) if 'Availability_Status' in df.columns else "Ready to Move"

        st.markdown("---")
        predict_btn = st.button(" Analyze Property Investment", use_container_width=True, type="primary")

        if predict_btn:
            # 1. Compute Derived Features matching training phase
            price_per_sqft = (price_lakhs * 100000) / size_sqft
            infra_score = round((schools_rating + hospitals_rating + transport_rating) / 3.0, 2)
            year_built = 2025 - age_property

            # 2. Build Raw Input DataFrame
            input_dict = {
                'ID': [9999],
    'State': [state_val],
    'City': [city],
    'Locality': [locality],
    'Property_Type': [property_type],
    'BHK': [int(bhk)],
    'Size_in_SqFt': [int(size_sqft)],
    'Price_in_Lakhs': [float(price_lakhs)],
    'Price_per_SqFt': [float(price_per_sqft)],
    'Year_Built': [int(year_built)],
    'Furnished_Status': [str(furnished_status)],
    'Floor_No': [int(1)],           # Add or capture via slider if needed
    'Total_Floors': [int(5)],       # Add or capture via slider if needed
    'Age_of_Property': [int(age_property)],
    'Nearby_Schools': [int(schools_rating)],
    'Nearby_Hospitals': [int(hospitals_rating)],
    'Public_Transport_Accessibility': [str(transport_rating)],
    'Parking_Space': [str(parking_spaces)],
    'Security': [str("Yes")],       # Add if missing from your form
    'Amenities': [str("Gym, Pool")],# Add if missing from your form
    'Facing': [str(facing)],
    'Owner_Type': [str(owner_type)],
    'Availability_Status': [str(avail_status)],
    'Infrastructure_Score': [float(infra_score)]

            }
            
            input_df = pd.DataFrame(input_dict)
            
            # Align features with training columns
            expected_cols = cls_pipeline.feature_names_in_ if hasattr(cls_pipeline, "feature_names_in_") else input_df.columns
            for col in expected_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[expected_cols]

            # 3. Run Predictions
            is_good_inv = cls_pipeline.predict(input_df)[0]
            
            # Extract Probability Score %
            if hasattr(cls_pipeline, "predict_proba"):
                good_inv_pct = cls_pipeline.predict_proba(input_df)[0][1] * 100
            else:
                good_inv_pct = 100.0 if is_good_inv == 1 else 0.0

            future_price_5y = reg_pipeline.predict(input_df)[0]
            appreciation = future_price_5y - price_lakhs
            appreciation_pct = (appreciation / price_lakhs) * 100

            # 4. Display Results
            st.markdown("### Investment Analysis Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)

            with res_col1:
                st.markdown("#### 🟢 Recommendation")
                if is_good_inv == 1:
                    st.success("### ✅ GOOD INVESTMENT")
                    st.write("This property meets high profitability, infrastructure, and fair market value criteria.")
                else:
                    st.error("### 🔴 HIGH RISK / LOW GROWTH")
                    st.write("Property price or location parameters indicate below-average capital return potential.")

            with res_col2:
                st.markdown("#### Match Score")
                st.metric(label="Investment Suitability Potential", value=f"{good_inv_pct:.1f}%")
                st.progress(int(good_inv_pct))

            with res_col3:
                st.markdown("#### 📈 5-Year Valuation")
                st.metric(
                    label="Estimated Price in 5 Years", 
                    value=f"₹ {future_price_5y:.2f} Lakhs", 
                    delta=f"+₹ {appreciation:.2f} Lakhs ({appreciation_pct:.1f}%)"
                )

            # 5. Model Feature Importance / Insight Summary
            st.markdown("---")
            st.markdown("#### 🔍 Key Valuation Summary")
            info_col1, info_col2, info_col3 = st.columns(3)
            info_col1.write(f"• **Infrastructure Composite Score:** {infra_score} / 5.0")
            info_col2.write(f"• **Current Price / SqFt:** ₹ {price_per_sqft:.2f}")
            info_col3.write(f"• **Locality Median Price / SqFt:** ₹ {df[df['Locality']==locality]['Price_per_SqFt'].median():.2f}")

   
    # TAB 2: MARKET ANALYTICS DASHBOARD
    
    with tab2:
        st.subheader(" Real Estate Market Exploratory Visualizations")
        
        ana_col1, ana_col2 = st.columns(2)
        
        with ana_col1:
            st.markdown("##### Average Price in Lakhs by City")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            city_avg = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False)
            sns.barplot(x=city_avg.values, y=city_avg.index, palette="mako", ax=ax1)
            ax1.set_xlabel("Avg Price (in Lakhs ₹)")
            st.pyplot(fig1)

        with ana_col2:
            st.markdown("##### Price per SqFt Distribution by Property Type")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=df, x='Property_Type', y='Price_per_SqFt', palette="Set2", ax=ax2)
            ax2.set_ylabel("Price per SqFt (₹)")
            st.pyplot(fig2)

        st.markdown("---")
        
        ana_col3, ana_col4 = st.columns(2)
        
        with ana_col3:
            st.markdown("##### Transport Accessibility vs Investment Potential")
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.countplot(data=df, x='Public_Transport_Accessibility', hue='Good_Investment', palette="pastel", ax=ax3)
            ax3.set_xlabel("Public Transport Accessibility (1-5)")
            ax3.set_ylabel("Count")
            st.pyplot(fig3)

        with ana_col4:
            st.markdown("##### Infrastructure Score vs Price per SqFt")
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=df, x='Infrastructure_Score', y='Price_per_SqFt', hue='Good_Investment', palette="coolwarm", ax=ax4)
            st.pyplot(fig4)

    
    # TAB 3: MODEL PERFORMANCE & ARCHITECTURE

    with tab3:
        st.subheader("⚙️ Machine Learning Pipeline & MLflow Architecture")
        
        st.markdown("""
        ### 📌 Project Technical Stack:
        * **Preprocessing Pipeline:** `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder` for categorical variables.
        * **Classification Task (`Good_Investment`):** Trained on `LGBMClassifier` / `RandomForestClassifier`. Predicts binary label + probability confidence score.
        * **Regression Task (`Future_Price_5Y`):** Trained on `LGBMRegressor` / `RandomForestRegressor`. Forecasts 5-year capital valuation.
        * **Experiment Tracking:** **MLflow** tracked metric logs (Accuracy, F1-Score, ROC-AUC, RMSE, MAE, $R^2$).
        """)
        
        st.markdown("---")
        st.markdown("###  Loaded Pipeline Types:")
        st.code(f"Classification Model: {cls_pipeline.named_steps['classifier']}")
        st.code(f"Regression Model:     {reg_pipeline.named_steps['regressor']}")