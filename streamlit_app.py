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
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Spikes during the 1D CNN training process as the CPU handles matrix operations.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Total memory consumed by the dataset and active neural network model.")

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# --- TRACK SELECTION ---
st.sidebar.header("Select Focus Area")
track = st.sidebar.radio("Track", ["Clinical Science", "Foundational Science"], 
                         help="Toggle between patient-care outcomes and algorithmic-mechanism analysis.")
st.sidebar.markdown("---")

# --- NAVIGATION ---
activity = st.sidebar.radio("Navigation", [
    "Activity 1: Objective and Data",
    "Activity 2: Training and Base Metrics",
    "Activity 3: Evaluation Trade-offs",
    "Activity 4: Strategic Comparison"
], help="Navigate through the core components of the ML pipeline established in the notebook.")

display_system_monitor()

@st.cache_data
def load_data():
    try:
        # Load local file
        df = pd.read_csv("diabetes.csv")
    except:
        # Fallback to standard biomedical dataset
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
        df.drop(columns='target', inplace=True)
    
    # CLINICAL LABEL MAPPING for BMI, BP, and S1-S6
    mapping = {
        'age': 'Age',
        'sex': 'Sex',
        'bmi': 'BMI',
        'bp': 'Blood Pressure (MAP)',
        's1': 'Total Cholesterol (s1)',
        's2': 'LDL Cholesterol (s2)',
        's3': 'HDL Cholesterol (s3)',
        's4': 'Total/HDL Ratio (s4)',
        's5': 'Serum Triglycerides (s5)',
        's6': 'Blood Glucose (s6)'
    }
    df.rename(columns=mapping, inplace=True)
    return df

df = load_data()

# ==========================================
# ACTIVITY 1: OBJECTIVE AND DATA
# ==========================================
if activity == "Activity 1: Objective and Data":
    st.title("Activity 1: Applied Fundamentals")
    
    with st.expander("Instructions and Objectives", expanded=True):
        if track == "Clinical Science":
            st.write("Examine how clinical features like BMI, BP, and Lab Results (s1-s6) correlate with In-Hospital Mortality.")
        else:
            st.write("Analyze the data types and distributions of input features to understand bias and variance.")

    st.header("Project Scenario")
    # Image placeholder: clinical decision support system architecture
    if track == "Clinical Science":
        st.write("""
        You are part of a clinical analytics team. Your objective is to use a CNN model to predict in-hospital mortality using demographics 
        and lab results. The goal is to provide triage support to ICU teams.
        """)
    else:
        st.write("""
        This task focuses on using 1D Convolutional Neural Networks for binary classification. 
        The primary advantage of using a CNN here is its ability to automatically identify non-linear relationships in data arrays.
        """)

    st.markdown("### Interactive Data Exploration")
    feature_to_view = st.selectbox("Select a Clinical Feature:", df.columns[:-1], 
                                   help="Select a feature to see how its mean value differs between survivors and mortality cases.")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Outcome Distribution**")
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival', 1: 'Death'})
        st.bar_chart(class_counts, color="#FF4B4B")
        st.caption("Class imbalance: Most cases in the dataset are Survivals.")
    with col2:
        st.markdown(f"**Mean {feature_to_view} by Outcome**")
        feature_means = df.groupby('Outcome')[feature_to_view].mean()
        st.bar_chart(feature_means)

# ==========================================
# ACTIVITY 2: TRAINING AND BASE METRICS
# ==========================================
elif activity == "Activity 2: Training and Base Metrics":
    st.title("Activity 2: Training and Accuracy")
    
    with st.expander("Instructions", expanded=True):
        st.write("Adjust hyperparameters and click Train to observe the baseline Total Accuracy metric.")

    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 5, 50, 20, help="Number of complete passes through the dataset.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32], value=16, help="Records processed before a weight update.")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("1D CNN Configuration")
        # Image placeholder: 1D Convolutional Neural Network architecture
        if st.button("Execute Training", help="Trains the CNN on 80% of the dataset."):
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
            
            with st.spinner("Training..."):
                history = model.fit(X_scaled, y, epochs=epochs, validation_split=0.2, verbose=0)
            
            st.session_state['act2_history'] = history.history
            st.success("Training Complete")
            
    with col2:
        if 'act2_history' in st.session_state:
            st.subheader("Performance Over Time")
            st.line_chart(pd.DataFrame(st.session_state['act2_history'])['accuracy'])
            st.metric("Total Accuracy", f"{st.session_state['act2_history']['accuracy'][-1]:.2%}", 
                      help="The percentage of correct predictions out of all samples.")
            st.warning("Notebook Question: Is total accuracy a good evaluation metric for this case?")

# ==========================================
# ACTIVITY 3: EVALUATION TRADE-OFFS
# ==========================================
elif activity == "Activity 3: Evaluation Trade-offs":
    st.title("Activity 3: Advanced Clinical Metrics")
    
    with st.expander("Instructions", expanded=True):
        st.write("Run the 5-fold evaluation and adjust the threshold to observe sensitivity and precision.")

    # Image placeholder: confusion matrix for binary classification
    if st.button("Run 5-Fold Evaluation", help="Performs cross-validation to assess model stability across data segments."):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx]).reshape(len(train_idx), X.shape[1], 1)
            X_val = scaler.transform(X[val_idx]).reshape(len(val_idx), X.shape[1], 1)
            
            model = Sequential([Input(shape=(X.shape[1], 1)), Conv1D(16, 2, activation='relu'), Flatten(), Dense(1, activation='sigmoid')])
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.fit(X_train, y[train_idx], epochs=10, verbose=0)
            
            y_prob = model.predict(X_val, verbose=0)
            results.append((y[val_idx], y_prob))
        
        st.session_state['act3_results'] = results
        st.success("Evaluation Complete")

    if 'act3_results' in st.session_state:
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 
                              help="Shifting the threshold changes the balance between Sensitivity and Specificity.")
        
        metrics = []
        for y_true, y_prob in st.session_state['act3_results']:
            y_pred = (y_prob > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.append([tp/(tp+fn), tn/(tn+fp), tp/(tp+fp)])
        
        avg_m = np.mean(metrics, axis=0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sensitivity (Recall)", f"{avg_m[0]:.3f}", help="Ability to correctly identify patients who died.")
        c2.metric("Specificity", f"{avg_m[1]:.3f}", help="Ability to correctly identify patients who survived.")
        c3.metric("Precision", f"{avg_m[2]:.3f}", help="How many 'Death' predictions were actually correct.")

# ==========================================
# ACTIVITY 4: STRATEGIC COMPARISON
# ==========================================
elif activity == "Activity 4: Strategic Comparison":
    st.title("Activity 4: Model Comparison")
    
    with st.expander("Instructions", expanded=True):
        st.write("Compare the 1D CNN results with the Milestone 1 Decision Tree.")

    # Image placeholder: decision tree vs neural network architecture
    st.subheader("Model Comparison Matrix")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decision Tree (Milestone 1)**")
        st.write("- Logic: Interpretable 'If-Then' branches.")
        st.write("- Transparency: High (White Box).")
    with col2:
        st.markdown("**1D CNN (Current Model)**")
        st.write("- Logic: Complex non-linear filters.")
        st.write("- Transparency: Low (Black Box).")

    st.markdown("---")
    
    st.subheader("Interactive Strategy Selector")
    priority = st.select_slider("What is the hospital's priority?", options=["Interpretability", "Balanced", "Performance"], 
                                help="Selecting Performance favors the CNN; Interpretability favors the Decision Tree.")
    
    if priority == "Interpretability":
        st.info("Strategy Recommendation: Deploy the Decision Tree.")
    elif priority == "Performance":
        st.success("Strategy Recommendation: Deploy the 1D CNN.")
    else:
        st.warning("Strategy Recommendation: Use a hybrid model or model-explainability tools.")
        
    st.bar_chart(pd.DataFrame({
        'Metric': ['Interpretability', 'Raw Performance', 'Automation'],
        'Decision Tree': [9, 5, 2],
        '1D CNN': [2, 9, 8]
    }).set_index('Metric'))
