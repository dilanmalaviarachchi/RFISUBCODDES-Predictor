import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the saved model and encoders
@st.cache_resource
def load_model_and_encoders():
    with open('models/oxgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoder_pos.pkl', 'rb') as f:
        le_pos = pickle.load(f)
    with open('models/label_encoder_rfisubcoddes.pkl', 'rb') as f:
        le_rfisubcoddes = pickle.load(f)
    return model, le_pos, le_rfisubcoddes

model, le_pos, le_rfisubcoddes = load_model_and_encoders()

# Load your original dataframe for visualization
@st.cache_data
def load_data():
    # Replace this with your actual data loading code
    df = pd.read_excel('C:\\Users\\Malavi\\Desktop\\RFISUBCODDES Predictor\\Ancillary Rev - 2015 - 2023.xlsx', header=2)
    df['Flight Date'] = pd.to_datetime(df['Flight Date'])
    df['Year'] = df['Flight Date'].dt.year
    df['Month'] = df['Flight Date'].dt.month
    return df

df = load_data()

# Get unique point of sale locations from the encoder
point_of_sale_options = le_pos.classes_

# Streamlit app
st.title("RFISUBCODDES Predictor & Analytics")
st.write("""
This app predicts the most likely RFISUBCODDES (ancillary revenue category) and provides historical insights.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Historical Trends", "Top Categories"])

with tab1:
    # Prediction tab
    st.header("Make a Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=2015, max_value=2030, value=2024, step=1, key="pred_year")
        month = st.number_input("Month", min_value=1, max_value=12, value=1, step=1, key="pred_month")
    
    with col2:
        point_of_sale = st.selectbox("Point of Sale", sorted(point_of_sale_options), key="pred_pos")
        count = st.number_input("Expected Transaction Count", min_value=1, value=50, key="pred_count")

    # Prediction function
    def predict_rfisubcoddes(year, month, point_of_sale, count):
        pos_encoded = le_pos.transform([point_of_sale])[0]
        prediction = model.predict([[year, month, pos_encoded, count]])
        return le_rfisubcoddes.inverse_transform(prediction)[0]

    if st.button("Predict RFISUBCODDES", key="predict_btn"):
        prediction = predict_rfisubcoddes(year, month, point_of_sale, count)
        st.success(f"Predicted RFISUBCODDES: **{prediction}**")
        
        if "BAGGAGE" in prediction or "WEIGHT" in prediction:
            st.info("This prediction suggests baggage-related ancillary services might be most profitable.")
        elif "SEAT" in prediction:
            st.info("This prediction suggests seat-related upgrades might be most profitable.")
        elif "UPGRADE" in prediction:
            st.info("This prediction suggests cabin upgrades might be most profitable.")

with tab2:
    # Historical trends tab
    st.header("Historical Trends Analysis")
    
    selected_pos = st.selectbox("Select Point of Sale for Analysis", sorted(point_of_sale_options), key="hist_pos")
    
    # Filter data for selected point of sale
    df_filtered = df[df['Point of Sale'] == selected_pos]
    
    if not df_filtered.empty:
        # Time series of RFISUBCODDES distribution
        st.subheader(f"RFISUBCODDES Distribution Over Time for {selected_pos}")
        
        # Create a time series count plot - FIXED DATE CREATION
        df_time = df_filtered.groupby(['Year', 'Month', 'RFISUBCODDES']).size().reset_index(name='Count')
        df_time['Date'] = pd.to_datetime(df_time['Year'].astype(str) + '-' + df_time['Month'].astype(str) + '-01')
        
        top_5 = df_filtered['RFISUBCODDES'].value_counts().nlargest(5).index
        df_top5 = df_time[df_time['RFISUBCODDES'].isin(top_5)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df_top5, x='Date', y='Count', hue='RFISUBCODDES', ax=ax)
        plt.xticks(rotation=45)
        plt.title(f"Top 5 RFISUBCODDES Trends for {selected_pos}")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Monthly patterns
        st.subheader("Monthly Patterns")
        df_monthly = df_filtered.groupby(['Month', 'RFISUBCODDES']).size().reset_index(name='Count')
        df_monthly_top5 = df_monthly[df_monthly['RFISUBCODDES'].isin(top_5)]
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        sns.barplot(data=df_monthly_top5, x='Month', y='Count', hue='RFISUBCODDES', ax=ax2)
        plt.title(f"Monthly Distribution of Top 5 RFISUBCODDES for {selected_pos}")
        plt.tight_layout()
        st.pyplot(fig2)
    else:
        st.warning(f"No data available for {selected_pos}")

with tab3:
    # Top categories tab
    st.header("Top RFISUBCODDES Categories")
    
    # Overall top categories
    st.subheader("Overall Top 5 RFISUBCODDES")
    top5_overall = df['RFISUBCODDES'].value_counts().nlargest(5)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    top5_overall.plot(kind='bar', ax=ax3)
    plt.title("Top 5 RFISUBCODDES Categories (All Time)")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig3)
    
    # Top by revenue
    st.subheader("Top 5 RFISUBCODDES by Revenue")
    top5_revenue = df.groupby('RFISUBCODDES')['Sum of Revenue USD'].sum().nlargest(5)
    
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    top5_revenue.plot(kind='bar', ax=ax4)
    plt.title("Top 5 RFISUBCODDES by Revenue (All Time)")
    plt.ylabel("Total Revenue (USD)")
    plt.xticks(rotation=45)
    st.pyplot(fig4)
    
    # Show the actual data
    st.subheader("Detailed Data")
    st.dataframe(top5_overall.reset_index().rename(columns={'index': 'RFISUBCODDES', 'RFISUBCODDES': 'Count'}))

# Add some additional information
st.sidebar.markdown("---")
st.sidebar.subheader("About the Model")
st.sidebar.write("""
- Trained on historical ancillary revenue data (2015-2020)
- Uses XGBoost algorithm
- Current accuracy: 75%
""")

st.sidebar.markdown("---")
st.sidebar.write("Data last updated: 2023")