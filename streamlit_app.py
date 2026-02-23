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
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Spikes during the 1D CNN training process as the CPU handles matrix multiplications.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Total memory currently consumed by the dataset and neural network model.")

st.set_page_config(page_title="Clinical ML Demo", layout="wide")

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
        # Fallback to standard dataset
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
        df.drop(columns='target', inplace=True)
    
    # CLINICAL LABEL MAPPING
    # Mapping S1-S6 and others to eICU Clinical Features
    mapping = {
        'age': 'Age',
        'sex': 'Sex',
        'bmi': 'BMI',
        'bp': 'Blood Pressure (MAP)',
        's1': 'Total Cholesterol (s1)',
        's2': 'LDL (s2)',
        's3': 'HDL (s3)',
        's4': 'Total/HDL Ratio (s4)',
        's5': 'Serum Triglycerides (s5)',
        's6': 'Glucose (s6)'
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
        st.write("""
        1. **Objective:** Predict in-hospital mortality using demographics and lab results.
        2. **Job Task:** Binary Classification (0: Survival, 1: Death).
        3. **CNN Advantage:** 1D CNNs automatically identify complex relationships between lab values without manual feature engineering.
        """)

    st.header("Clinical Scenario")
    # 

[Image of clinical decision support system architecture]

    st.write("""
    You are part of a hospital’s clinical analytics team using a **1D CNN model** to predict **in-hospital mortality** using data from the eICU Collaborative Research Database. The dataset includes demographics and 
    selected lab results such as glucose, creatinine, and potassium.
    """)

    st.markdown("### Interactive Data Exploration")
    feature_to_view = st.selectbox("Select a Clinical Feature:", df.columns[:-1], 
                                   help="Select a feature to see how its average value differs between survivors and mortality cases.")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Outcome Distribution**")
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival', 1: 'Death'})
        st.bar_chart(class_counts, color="#FF4B4B")
        st.caption("Notice the class imbalance; most patients survive, making 'Accuracy' a deceptive metric.")
    with col2:
        st.markdown(f"**Mean {feature_to_view} by Outcome**")
        feature_means = df.groupby('Outcome')[feature_to_view].mean()
        st.bar_chart(feature_means)
        st.caption(f"A visual check of how {feature_to_view} correlates with mortality.")

# ==========================================
# ACTIVITY 2: TRAINING AND BASE METRICS
# ==========================================
elif activity == "Activity 2: Training and Base Metrics":
    st.title("Activity 2: Training and Accuracy")
    
    with st.expander("Instructions", expanded=True):
        st.write("Adjust hyperparameters and click 'Execute Training' to observe how well the CNN learns the data patterns.")

    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 5, 50, 20, help="The number of complete passes through the training dataset.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32], value=16, help="The number of patient samples processed before updating the model's weights.")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("1D CNN Configuration")
        # 
        if st.button("Execute Training", help="Initializes and trains the CNN on 80% of the dataset."):
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
                      help="The percentage of correct predictions out of total samples.")
            st.warning("Notebook Question: Is total accuracy a good evaluation metric for this case?")

# ==========================================
# ACTIVITY 3: EVALUATION TRADE-OFFS
# ==========================================
elif activity == "Activity 3: Evaluation Trade-offs":
    st.title("Activity 3: Clinical Performance Evaluation")
    
    with st.expander("Instructions", expanded=True):
        st.write("Run the 5-fold cross-validation and adjust the threshold to see the trade-off between Sensitivity and Specificity.")

    if st.button("Run 5-Fold Evaluation", help="Performs cross-validation to ensure model performance is stable across different subsets of data."):
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
        st.success("Cross-Validation Complete")

    if 'act3_results' in st.session_state:
        # 
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, 
                              help="Higher thresholds make the model 'stricter' (more Specificity), while lower thresholds make it 'cautious' (more Sensitivity).")
        
        metrics = []
        for y_true, y_prob in st.session_state['act3_results']:
            y_pred = (y_prob > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.append([tp/(tp+fn), tn/(tn+fp), tp/(tp+fp)])
        
        avg_m = np.mean(metrics, axis=0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sensitivity (Recall)", f"{avg_m[0]:.3f}", help="The ability to correctly identify all patients who died.")
        c2.metric("Specificity", f"{avg_m[1]:.3f}", help="The ability to correctly identify all patients who survived.")
        c3.metric("Precision", f"{avg_m[2]:.3f}", help="Of those predicted to die, how many actually did?")
        
        st.info("Notebook Question: After observing sensitivity, specificity, and precision, how is the performance now?")

# ==========================================
# ACTIVITY 4: STRATEGIC COMPARISON
# ==========================================
elif activity == "Activity 4: Strategic Comparison":
    st.title("Activity 4: Model Comparison")
    
    with st.expander("Strategic Evaluation", expanded=True):
        st.write("Compare the CNN results with the Milestone 1 Decision Tree to determine the best choice for ICU deployment.")

    st.subheader("Model Comparison Matrix")
    # [Image comparing decision tree architecture to neural network architecture]
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decision Tree (Milestone 1)**")
        st.write("- Logic: Interpretable 'If-Then' branches.")
        st.write("- Feature Extraction: Manual.")
        st.write("- Transparency: High (White Box).")
    with col2:
        st.markdown("**1D CNN (Deep Learning)**")
        st.write("- Logic: Complex non-linear filters.")
        st.write("- Feature Extraction: Automated.")
        st.write("- Transparency: Low (Black Box).")

    st.markdown("---")
    
    st.subheader("Interactive Decision Tool")
    priority = st.select_slider("Select Hospital Priority:", options=["Interpretability", "Balanced", "Performance"], 
                                help="Which factor is most important for your ICU team?")
    
    if priority == "Interpretability":
        st.info("Recommendation: Decision Tree. Prioritize this if doctors need to explain every prediction to a family.")
    elif priority == "Performance":
        st.success("Recommendation: 1D CNN. Prioritize this if you want the highest possible detection of at-risk patients.")
    else:
        st.warning("Recommendation: Hybrid approach. Consider using the CNN for detection and a Tree for explanation.")
        
    st.bar_chart(pd.DataFrame({
        'Metric': ['Interpretability', 'Raw Performance', 'Automation'],
        'Decision Tree': [9, 5, 2],
        '1D CNN': [2, 9, 8]
    }).set_index('Metric'))
