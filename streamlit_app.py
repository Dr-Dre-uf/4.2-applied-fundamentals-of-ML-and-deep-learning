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
        # Load local file if available
        df = pd.read_csv("diabetes.csv")
    except:
        # Fallback to standard biomedical dataset
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
        df.drop(columns='target', inplace=True)
    
    # CLINICAL LABEL MAPPING for S1-S6
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
            st.write("Understand how patient demographics and lab results (S1-S6) serve as predictors for ICU mortality triage.")
        else:
            st.write("Analyze the statistical distribution of feature vectors and class imbalance before training a 1D CNN.")

    st.header("Project Scenario")
    # 

[Image of clinical decision support system architecture]

    if track == "Clinical Science":
        st.write("""
        You are part of a clinical analytics team. Your objective is to use a CNN model to predict in-hospital mortality using demographics 
        and lab results like Glucose and Cholesterol levels. The goal is to identify at-risk patients early.
        """)
    else:
        st.write("""
        This task focuses on using 1D Convolutional Neural Networks for binary classification on clinical arrays. 
        The primary advantage of CNNs here is their ability to identify non-linear relationships without manual feature engineering.
        """)

    st.markdown("### Interactive Data Exploration")
    feature_to_view = st.selectbox("Select a Clinical Feature:", df.columns[:-1], 
                                   help="Select a feature to see how its mean value differs between survival and mortality cases.")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Outcome Distribution**")
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival', 1: 'Death'})
        st.bar_chart(class_counts, color="#FF4B4B")
        st.caption("Class imbalance: Note that the majority of cases are Survivals.")
    with col2:
        st.markdown(f"**Mean {feature_to_view} by Outcome**")
        feature_means = df.groupby('Outcome')[feature_to_view].mean()
        st.bar_chart(feature_means)
        st.caption(f"Visualizing the correlation between {feature_to_view} and the binary outcome.")

# ==========================================
# ACTIVITY 2: TRAINING AND BASE METRICS
# ==========================================
elif activity == "Activity 2: Training and Base Metrics":
    st.title("Activity 2: Training and Accuracy")
    
    with st.expander("Instructions", expanded=True):
        st.write("Adjust hyperparameters and execute training to observe 'Total Accuracy'. Reflect on if this metric is sufficient for clinical use.")

    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 5, 50, 20, help="The number of complete passes the algorithm makes through the training data.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32], value=16, help="The number of records processed before updating the model weights.")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("1D CNN Configuration")
        # 
        if st.button("Execute Training", help="Initializes model weights and begins the optimization process."):
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
            st.subheader("Model Convergence")
            st.line_chart(pd.DataFrame(st.session_state['act2_history'])['accuracy'])
            st.metric("Total Accuracy", f"{st.session_state['act2_history']['accuracy'][-1]:.2%}", 
                      help="The percentage of correct predictions out of all samples.")
            
            if track == "Clinical Science":
                st.warning("Clinical Note: Is total accuracy enough? A model predicting survival for everyone would still be 'accurate' but clinically useless.")
            else:
                st.warning("Foundational Note: Global accuracy is biased by the majority class in imbalanced datasets.")

# ==========================================
# ACTIVITY 3: EVALUATION TRADE-OFFS
# ==========================================
elif activity == "Activity 3: Evaluation Trade-offs":
    st.title("Activity 3: Advanced Clinical Metrics")
    
    with st.expander("Instructions", expanded=True):
        st.write("Run the 5-fold evaluation to observe Sensitivity, Specificity, and Precision.")

    # 
    if st.button("Run 5-Fold Evaluation", help="Performs cross-validation to assess model stability across different data segments."):
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
                              help="Shifting the threshold toward 0.1 increases Sensitivity (Recall); shifting toward 0.9 increases Specificity.")
        
        metrics = []
        for y_true, y_prob in st.session_state['act3_results']:
            y_pred = (y_prob > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.append([tp/(tp+fn), tn/(tn+fp), tp/(tp+fp)])
        
        avg_m = np.mean(metrics, axis=0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sensitivity (Recall)", f"{avg_m[0]:.3f}", help="Percentage of actual deaths correctly identified.")
        c2.metric("Specificity", f"{avg_m[1]:.3f}", help="Percentage of actual survivals correctly identified.")
        c3.metric("Precision", f"{avg_m[2]:.3f}", help="Percentage of 'Death' predictions that were correct.")

# ==========================================
# ACTIVITY 4: STRATEGIC COMPARISON
# ==========================================
elif activity == "Activity 4: Strategic Comparison":
    st.title("Activity 4: Architecture Comparison")
    
    with st.expander("Instructions", expanded=True):
        st.write("Reflect on the trade-offs between the CNN and the Milestone 1 Decision Tree.")

    # [Image comparing decision tree architecture to neural network architecture]
    st.subheader("Model Comparison Matrix")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decision Tree (Milestone 1)**")
        st.write("- Advantage: High Interpretability (Doctors can see the 'If-Then' logic).")
        st.write("- Disadvantage: Requires manual feature engineering.")
    with col2:
        st.markdown("**1D CNN (Current Model)**")
        st.write("- Advantage: Automated Feature Extraction (Finds hidden patterns in lab results).")
        st.write("- Disadvantage: Black Box (Hard to explain individual predictions).")

    st.markdown("---")
    
    st.subheader("Interactive Strategy Selector")
    priority = st.select_slider("What is the hospital's current priority?", options=["Interpretability", "Balanced", "Performance"], 
                                help="Selecting 'Performance' favors the CNN; selecting 'Interpretability' favors the Decision Tree.")
    
    if priority == "Interpretability":
        st.info("Strategy: Deploy the Decision Tree to ensure clinician trust and regulatory compliance.")
    elif priority == "Performance":
        st.success("Strategy: Deploy the 1D CNN to maximize the identification of at-risk patients.")
    else:
        st.warning("Strategy: Use a hybrid model or model-agnostic explainability tools (like SHAP).")
        
    st.bar_chart(pd.DataFrame({
        'Metric': ['Interpretability', 'Raw Performance', 'Automation'],
        'Decision Tree': [9, 5, 2],
        '1D CNN': [2, 9, 8]
    }).set_index('Metric'))
