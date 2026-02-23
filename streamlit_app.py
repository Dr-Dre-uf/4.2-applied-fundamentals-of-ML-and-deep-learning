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
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Spikes during the 1D CNN training process.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Total memory consumed by the dataset and model.")

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# --- TRACK SELECTION ---
st.sidebar.header("Select Focus Area")
track = st.sidebar.radio("Track", ["Clinical Science", "Foundational Science"], 
                         help="Toggle context between patient outcomes and algorithmic mechanics.")
st.sidebar.markdown("---")

# --- NAVIGATION ---
activity = st.sidebar.radio("Navigation", [
    "Activity 1: Objective and Data",
    "Activity 2: Training and Base Metrics",
    "Activity 3: Evaluation Trade-offs",
    "Activity 4: Strategic Comparison"
], help="Switch between the core components of the ML pipeline.")

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
# ACTIVITY 1: OBJECTIVE AND DATA
# ==========================================
if activity == "Activity 1: Objective and Data":
    st.title("Activity 1: Applied Fundamentals")
    
    with st.expander("Instructions and Objectives", expanded=True):
        st.write("""
        1. Review the clinical scenario: Predicting in-hospital mortality using eICU data.
        2. Identify the Job Task: Binary Classification (0: Survival, 1: Death).
        3. Explore the clinical features: Glucose, Creatinine, and Potassium.
        """)

    st.header("Project Scenario")
    # Image placeholder: clinical decision support system architecture
    st.write("""
    This demo uses a **1D CNN model** to predict mortality using patient demographics and lab results. 
    Unlike traditional models, the **advantage of using a CNN** here is its ability to identify complex 
    relationships between lab values and demographics without the need for manual feature engineering.
    """)

    st.markdown("### Interactive Data Exploration")
    feature_to_view = st.selectbox("Select a Clinical Feature:", df.columns[:-1], help="Visualize the distribution of specific patient features.")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Outcome Distribution**")
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival', 1: 'Death'})
        st.bar_chart(class_counts, color="#FF4B4B")
    with col2:
        st.markdown(f"**Average {feature_to_view} by Outcome**")
        feature_means = df.groupby('Outcome')[feature_to_view].mean()
        st.bar_chart(feature_means)

# ==========================================
# ACTIVITY 2: TRAINING AND BASE METRICS
# ==========================================
elif activity == "Activity 2: Training and Base Metrics":
    st.title("Activity 2: Training the Neural Network")
    
    with st.expander("Instructions", expanded=True):
        st.write("Adjust hyperparameters in the sidebar and click Execute Training to observe initial accuracy.")

    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 5, 50, 10, help="Number of training passes through the dataset.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32], value=16, help="Samples processed per gradient update.")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("1D CNN Configuration")
        # Image placeholder: 1D Convolutional Neural Network architecture
        if st.button("Execute Training", help="Trains the 1D CNN on the clinical dataset."):
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
            
            with st.spinner("Optimizing model weights..."):
                history = model.fit(X_scaled, y, epochs=epochs, validation_split=0.2, verbose=0)
            
            st.session_state['act2_history'] = history.history
            st.success("Training Complete")
            
    with col2:
        if 'act2_history' in st.session_state:
            st.subheader("Convergence and Performance")
            st.line_chart(pd.DataFrame(st.session_state['act2_history'])['accuracy'])
            st.metric("Total Accuracy", f"{st.session_state['act2_history']['accuracy'][-1]:.2%}", 
                      help="The percentage of correct predictions out of all patients.")
            st.warning("Critical Analysis: Is total accuracy a reliable metric for an imbalanced mortality dataset?")

# ==========================================
# ACTIVITY 3: EVALUATION TRADE-OFFS
# ==========================================
elif activity == "Activity 3: Evaluation Trade-offs":
    st.title("Activity 3: Clinical Performance Evaluation")
    
    with st.expander("Instructions", expanded=True):
        st.write("Run the evaluation and adjust the threshold to see how Sensitivity and Specificity react.")

    # Image placeholder: confusion matrix for binary classification
    if st.button("Run 5-Fold Evaluation", help="Calculates cross-validated performance metrics."):
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
        st.success("Evaluation results generated.")

    if 'act3_results' in st.session_state:
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 
                              help="Shifting the decision boundary to prioritize catching deaths (Sensitivity) or avoiding false alarms (Specificity).")
        
        metrics = []
        for y_true, y_prob in st.session_state['act3_results']:
            y_pred = (y_prob > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.append([tp/(tp+fn), tn/(tn+fp), tp/(tp+fp)])
        
        avg_m = np.mean(metrics, axis=0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sensitivity (Recall)", f"{avg_m[0]:.3f}", help="Ability to correctly identify patients at high risk of death.")
        c2.metric("Specificity", f"{avg_m[1]:.3f}", help="Ability to correctly identify patients who will survive.")
        c3.metric("Precision", f"{avg_m[2]:.3f}", help="The likelihood that a patient predicted to die actually dies.")

# ==========================================
# ACTIVITY 4: STRATEGIC COMPARISON
# ==========================================
elif activity == "Activity 4: Strategic Comparison":
    st.title("Activity 4: Architecture Comparison")
    
    with st.expander("Strategic Goal", expanded=True):
        st.write("Analyze whether the CNN performance is superior to the Milestone 1 Decision Tree and why.")

    # Image placeholder: decision tree vs neural network visual logic
    st.subheader("Model Comparison Matrix")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decision Tree (Milestone 1)**")
        st.write("- Logic: Interpretable If-Then splits.")
        st.write("- Feature Extraction: Manual.")
        st.write("- Transparency: High (White Box).")
    with col2:
        st.markdown("**1D CNN (Deep Learning)**")
        st.write("- Logic: Non-linear convolutional filters.")
        st.write("- Feature Extraction: Automated.")
        st.write("- Transparency: Low (Black Box).")

    st.markdown("---")
    
    st.subheader("Interactive Decision Matrix")
    st.write("Based on your priority, which model should the hospital adopt?")
    priority = st.select_slider("Select Core Priority:", options=["Interpretability", "Balanced", "Performance"], help="Determines the recommended architecture.")
    
    if priority == "Interpretability":
        st.info("Recommendation: Decision Tree. Use this when clinicians must understand every branch of the decision.")
    elif priority == "Performance":
        st.success("Recommendation: 1D CNN. Use this when maximizing the detection of at-risk patients is the highest priority.")
    else:
        st.warning("Recommendation: Hybrid / Ensemble approach required.")
        
    st.bar_chart(pd.DataFrame({
        'Metric': ['Interpretability', 'Raw Performance', 'Automation'],
        'Decision Tree': [9, 5, 2],
        '1D CNN': [2, 9, 8]
    }).set_index('Metric'))
