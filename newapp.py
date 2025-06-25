import sqlite3
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load model and encoders
with open("C:\\Users\\Malavi\\Desktop\\RFISUBCODDES Predictor\\new models\\xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("C:\\Users\\Malavi\\Desktop\\RFISUBCODDES Predictor\\new models\\le_pos.pkl", "rb") as f:
    le_pos = pickle.load(f)

with open("C:\\Users\\Malavi\\Desktop\\RFISUBCODDES Predictor\\new models\\le_target.pkl", "rb") as f:
    le_target = pickle.load(f)

# Sample average values
avg_count = 5.0
avg_rpc = 10.0

# Connect to SQLite
def connect_to_sqlite():
    return sqlite3.connect("ancillary_predictions.db")

# Setup database
def setup_database():
    conn = connect_to_sqlite()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pos TEXT,
            year INTEGER,
            month INTEGER,
            top1 TEXT,
            top1_prob REAL,
            top2 TEXT,
            top2_prob REAL,
            top3 TEXT,
            top3_prob REAL,
            pred_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("SQLite DB Ready!")

setup_database()

# Streamlit UI
st.set_page_config(page_title="Ancillary Revenue Predictor", layout="wide", page_icon="📊")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #000;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
    }
    .prediction-card {
        background-color: black;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .header { color: #2c3e50; }
</style>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.title("📊 Ancillary Revenue Predictor")
    st.markdown("Enter details to get the top predicted ancillary types with visualizations.")

    st.subheader("Input Parameters")
    point_of_sale = st.selectbox("Point of Sale", le_pos.classes_)
    year = st.number_input("Year", min_value=2015, max_value=2030, step=1)
    month = st.selectbox("Month", list(range(1, 13)))

    if st.button("Predict"):
        encoded_pos = le_pos.transform([point_of_sale])[0]

        input_df = pd.DataFrame([{
            'Point_of_Sale_encoded': encoded_pos,
            'Year': int(year),
            'Month': int(month),
            'Count': avg_count,
            'Revenue_per_Count': avg_rpc
        }])

        prediction_proba = model.predict_proba(input_df)
        top_3_idx = prediction_proba[0].argsort()[-3:][::-1]
        top_3_classes = model.classes_[top_3_idx]
        top_3_probs = prediction_proba[0][top_3_idx]
        top_3_labels = le_target.inverse_transform(top_3_classes)

        # Store predictions in session state
        st.session_state['predictions'] = {
            'labels': top_3_labels,
            'probs': top_3_probs,
            'all_probs': prediction_proba[0],
            'all_labels': le_target.inverse_transform(model.classes_)
        }

        # Save to SQLite
        conn = connect_to_sqlite()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (pos, year, month, top1, top1_prob, top2, top2_prob, top3, top3_prob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            point_of_sale, year, month,
            top_3_labels[0], float(top_3_probs[0]),
            top_3_labels[1], float(top_3_probs[1]),
            top_3_labels[2], float(top_3_probs[2])
        ))
        conn.commit()
        conn.close()
        st.success("Prediction saved to SQLite!")

# Display results and visualizations
if 'predictions' in st.session_state:
    predictions = st.session_state['predictions']
    with col2:
        st.subheader("Prediction Results")

        with st.expander("Top Predictions", expanded=True):
            for i, (label, prob) in enumerate(zip(predictions['labels'], predictions['probs'])):
                progress_value = float(prob) * 100
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>#{i+1}: {label}</h3>
                    <p>Confidence: {round(progress_value, 2)}%</p>
                    <progress value="{progress_value}" max="100"></progress>
                </div>
                """, unsafe_allow_html=True)

        # TABS: 1. Distribution, 2. Pie, 3. Monthly, 4. Recent
        tab1, tab2, tab3, tab4 = st.tabs([
            "Probability Distribution", 
            "Top Categories", 
            "Monthly Trend", 
            "Recent Predictions"
        ])

        with tab1:
            st.subheader("Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=predictions['all_probs'] * 100,
                y=predictions['all_labels'],
                ax=ax,
                palette="viridis"
            )
            ax.set_xlabel("Probability (%)")
            ax.set_ylabel("Ancillary Type")
            ax.set_title("Prediction Probabilities for All Categories")
            st.pyplot(fig)

        with tab2:
            st.subheader("Top Categories Breakdown")
            fig, ax = plt.subplots(figsize=(8, 8))
            explode = [0.1 if x == predictions['labels'][0] else 0 for x in predictions['labels']]
            ax.pie(
                predictions['probs'] * 100,
                labels=predictions['labels'],
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                shadow=True,
                colors=sns.color_palette("pastel")
            )
            ax.axis('equal')
            ax.set_title("Top 3 Predictions Distribution")
            st.pyplot(fig)

        with tab3:
            st.subheader("Monthly Trend Analysis")
            months = list(range(1, 13))
            simulated_data = {
                'Month': months,
                predictions['labels'][0]: np.random.normal(loc=predictions['probs'][0]*100, scale=5, size=12),
                predictions['labels'][1]: np.random.normal(loc=predictions['probs'][1]*100, scale=5, size=12),
                predictions['labels'][2]: np.random.normal(loc=predictions['probs'][2]*100, scale=5, size=12),
            }
            df_simulated = pd.DataFrame(simulated_data)
            fig, ax = plt.subplots(figsize=(10, 6))
            for label in predictions['labels']:
                ax.plot(df_simulated['Month'], df_simulated[label], marker='o', label=label)
            ax.set_xlabel("Month")
            ax.set_ylabel("Probability (%)")
            ax.set_title("Simulated Monthly Trends for Top Categories")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        with tab4:
            st.subheader("📂 All Predictions")
            conn = connect_to_sqlite()
            df_recent = pd.read_sql_query(
            "SELECT pos, year, month, top1, top1_prob, top2, top2_prob, top3, top3_prob, pred_time FROM predictions ORDER BY pred_time DESC",
            conn
              )
            conn.close()
              
            st.dataframe(df_recent, use_container_width=True)

# Sidebar with info
with st.sidebar:
    st.markdown("## 📘 About")
    st.markdown("""
This tool predicts likely ancillary revenue types using:
- Point of Sale
- Year and Month
- Historical averages
""")

    st.markdown("## 📊 Model Info")
    st.write("**Model:** XGBoost Classifier")
    st.write(f"**Target Classes:** {len(le_target.classes_)}")
    st.write(f"**POS Options:** {len(le_pos.classes_)}")

    st.markdown("## 📝 Instructions")
    st.markdown("1. Select POS\n2. Choose Year and Month\n3. Click Predict")
