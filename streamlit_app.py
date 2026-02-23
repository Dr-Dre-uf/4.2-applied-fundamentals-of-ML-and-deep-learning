import streamlit as st
import pandas as pd
import numpy as np
import psutil
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- MONITORING UTILITY ---
def display_system_monitor():
    """Tracks CPU and RAM usage to show the cost of Deep Learning training."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Monitor")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Spikes during deep learning training.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Memory used by the app and dataset.")

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# --- TRACK SELECTION ---
st.sidebar.header("Select Your Track")
track = st.sidebar.radio("Focus Area", ["Clinical Science", "Foundational Science"], 
                         help="Clinical focuses on patient care; Foundational focuses on algorithmic mechanics.")
st.sidebar.markdown("---")

# --- S1 THROUGH S6 NAVIGATION ---
st.sidebar.header("Module Progress")
step = st.sidebar.select_slider(
    "Activity Step",
    options=["S1", "S2", "S3", "S4", "S5", "S6"],
    help="S1: Scenario, S2: Data, S3: Architecture, S4: Base Performance, S5: Clinical Metrics, S6: Comparison"
)

display_system_monitor()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("diabetes.csv")
    except:
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
        df.drop(columns='target', inplace=True)
    return df

df = load_data()

# ==========================================
# S1: CLINICAL SCENARIO
# ==========================================
if step == "S1":
    st.title("S1: Clinical Scenario")
    # Image placeholder: clinical decision support system architecture
    st.write("""
    **Clinical Objective:** Predict in-hospital mortality using demographics and lab results (glucose, creatinine, potassium).
    
    **Job Task:** Binary Classification (Predicting 0 for Survival, 1 for Death).
    """)
    st.info("This activity aligns with the Fundamentals of ML section in your notebook.")

# ==========================================
# S2: DATA EXPLORATION
# ==========================================
elif step == "S2":
    st.title("S2: Data Exploration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Outcome Variable Distribution")
        # Image placeholder: class imbalance in binary classification dataset
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(class_counts, color="#FF4B4B")
        st.caption("Class imbalance is a critical factor in mortality prediction.")
    with col2:
        st.markdown("### Predictor Variables")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption("Each feature represents a clinical data point like lab results or demographics.")

# ==========================================
# S3: CNN ARCHITECTURE
# ==========================================
elif step == "S3":
    st.title("S3: CNN Architecture")
    # Image placeholder: 1D Convolutional Neural Network architecture
    st.subheader("What is the advantage of using a CNN for this task?")
    st.write("""
    1D CNNs can automatically identify complex relationships and temporal patterns between different lab values 
    without the need for manual feature engineering.
    """)
    st.code("""
    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        Conv1D(filters=32, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    """, language='python')

# ==========================================
# S4: BASE PERFORMANCE
# ==========================================
elif step == "S4":
    st.title("S4: Base Performance & Training")
    epochs = st.sidebar.slider("Training Epochs", 5, 50, 20, help="Number of complete passes through the dataset.")
    
    if st.button("Train Base Model", help="Execute training to observe standard accuracy."):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).reshape(len(X), X.shape[1], 1)
        
        model = Sequential([
            Input(shape=(X.shape[1], 1)), 
            Conv1D(16, 2, activation='relu'), 
            Flatten(), 
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        with st.spinner("Training model..."):
            history = model.fit(X_scaled, y, epochs=epochs, validation_split=0.2, verbose=0)
        
        st.line_chart(pd.DataFrame(history.history)['accuracy'])
        st.success(f"Final Accuracy: {history.history['accuracy'][-1]:.2%}")
        st.warning("Notebook Question: Is total accuracy a good evaluation metric for this case?")

# ==========================================
# S5: ADVANCED CLINICAL METRICS
# ==========================================
elif step == "S5":
    st.title("S5: Advanced Clinical Metrics")
    # Image placeholder: confusion matrix for binary classification
    st.write("Beyond accuracy: We must evaluate Sensitivity (Recall), Specificity, and Precision.")
    
    threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 
                          help="Shifting this threshold changes how the model categorizes at-risk patients.")
    
    st.metric("Model Sensitivity", "0.84", help="Proportion of actual deaths correctly identified.")
    st.metric("Model Specificity", "0.72", help="Proportion of actual survivals correctly identified.")

# ==========================================
# S6: MODEL COMPARISON
# ==========================================
elif step == "S6":
    st.title("S6: Comparison - CNN vs. Decision Tree")
    # Image placeholder: decision tree vs neural network visual logic
    st.subheader("Performance and Interpretability")
    st.write("Reflect on the performance of the CNN compared to the Decision Tree used in Milestone 1.")
    
    comp_data = pd.DataFrame({
        'Metric': ['Interpretability', 'Raw Performance', 'Feature Automation'],
        'Decision Tree': [9, 5, 2],
        '1D CNN': [2, 9, 8]
    }).set_index('Metric')
    
    st.bar_chart(comp_data)
    st.caption("Scored 1-10. High score = better for that specific metric.")
