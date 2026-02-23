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
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Watch this spike during training; it represents the computational cost of the CNN filters.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Total memory used by the dataset and the neural network model.")

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# --- TRACK SELECTION ---
st.sidebar.header("Select Focus Area")
track = st.sidebar.radio("Track", ["Clinical Science", "Foundational Science"], 
                         help="Clinical focuses on patient care impact; Foundational focuses on algorithmic logic.")
st.sidebar.markdown("---")

# --- NAVIGATION ---
activity = st.sidebar.radio("Navigation", [
    "Activity 1: Objective and Data",
    "Activity 2: Training and Base Metrics",
    "Activity 3: Evaluation Trade-offs",
    "Activity 4: Strategic Comparison"
], help="Switch between the different phases of the machine learning pipeline.")

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
    
    # CLINICAL LABEL MAPPING for BMI, BP, and S1-S6
    mapping = {
        'age': 'Age', 'sex': 'Sex', 'bmi': 'BMI', 'bp': 'Blood Pressure (MAP)',
        's1': 'Total Cholesterol (s1)', 's2': 'LDL Cholesterol (s2)', 's3': 'HDL Cholesterol (s3)',
        's4': 'Total/HDL Ratio (s4)', 's5': 'Serum Triglycerides (s5)', 's6': 'Blood Glucose (s6)'
    }
    df.rename(columns=mapping, inplace=True)
    return df

df = load_data()

# ==========================================
# ACTIVITY 1: OBJECTIVE AND DATA
# ==========================================
if activity == "Activity 1: Objective and Data":
    st.title("Activity 1: Applied Fundamentals")
    
    with st.expander("Activity Guide: How to Use This Page", expanded=True):
        st.write("1. **Read the Scenario:** Understand the clinical problem we are trying to solve.")
        st.write("2. **Explore the Distribution:** Check the 'Outcome Distribution' chart to see the mortality rate baseline.")
        st.write("3. **Compare Features:** Use the dropdown to see how specific lab results (BMI, BP, S1-S6) differ between survivors and non-survivors.")

    st.header("Project Scenario")
    # Placeholder for image: 
    if track == "Clinical Science":
        st.write("""
        You are a data scientist in an ICU. Your objective is to build a 1D CNN that identifies patients at high risk of 
        mortality using vital signs and lab results.
        """)
    else:
        st.write("""
        This task demonstrates 1D CNN architecture. The goal is to evaluate the model's ability to learn 
        from sequential or tabular clinical arrays without manual feature engineering.
        """)

    st.markdown("### Interactive Data Exploration")
    feature_to_view = st.selectbox("Select a Clinical Feature to Analyze:", df.columns[:-1], 
                                   help="Analyze how this clinical metric correlates with patient outcomes.")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Outcome Distribution**")
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival', 1: 'Death'})
        st.bar_chart(class_counts, color="#FF4B4B")
        st.caption("Class imbalance: Most records represent patient survival.")
    with col2:
        st.markdown(f"**Mean {feature_to_view} by Outcome**")
        feature_means = df.groupby('Outcome')[feature_to_view].mean()
        st.bar_chart(feature_means)

# ==========================================
# ACTIVITY 2: TRAINING AND BASE METRICS
# ==========================================
elif activity == "Activity 2: Training and Base Metrics":
    st.title("Activity 2: Training and Accuracy")
    
    with st.expander("Activity Guide: How to Train the Model", expanded=True):
        st.write("1. **Set Parameters:** Use the sidebar sliders to set Epochs and Batch Size.")
        st.write("2. **Train:** Click 'Execute Training' to start the neural network's optimization process.")
        st.write("3. **Observe:** Watch the performance chart live. Does accuracy keep improving or does it plateau?")

    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 5, 50, 20, help="More epochs allow the model to learn longer but increase CPU time.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32], value=16, help="Smaller batches make training more granular.")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("1D CNN Configuration")
        # Placeholder for image: 
        if st.button("Execute Training", help="Starts the automated learning process."):
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
            
            with st.spinner("Training model... Observe the System Monitor in the sidebar!"):
                history = model.fit(X_scaled, y, epochs=epochs, validation_split=0.2, verbose=0)
            
            st.session_state['act2_history'] = history.history
            st.success("Training Complete")
            
    with col2:
        if 'act2_history' in st.session_state:
            st.subheader("Model Learning Curve")
            st.line_chart(pd.DataFrame(st.session_state['act2_history'])['accuracy'])
            st.metric("Final Total Accuracy", f"{st.session_state['act2_history']['accuracy'][-1]:.2%}", 
                      help="The percentage of correct predictions (Survivals + Deaths).")
            st.warning("Note: In imbalanced mortality data, high accuracy can be misleading.")

# ==========================================
# ACTIVITY 3: EVALUATION TRADE-OFFS
# ==========================================
elif activity == "Activity 3: Evaluation Trade-offs":
    st.title("Activity 3: Advanced Clinical Metrics")
    
    with st.expander("Activity Guide: How to Evaluate the Model", expanded=True):
        st.write("1. **Generate Predictions:** Click 'Run 5-Fold Evaluation' to get cross-validated results.")
        st.write("2. **Adjust Threshold:** Move the slider. Notice how catching more deaths (Sensitivity) often creates more false alarms.")
        st.write("3. **Analyze:** Look for the 'Sweet Spot' between sensitivity and specificity.")

    # Placeholder for image: 
    if st.button("Run 5-Fold Evaluation", help="Runs the model 5 separate times on different data slices for rigor."):
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
        st.success("Full Evaluation Generated")

    if 'act3_results' in st.session_state:
        threshold = st.slider("Classification Sensitivity Threshold", 0.1, 0.9, 0.5, 
                              help="Lowering this makes the model 'cautious' (picks up more deaths); Raising it makes it 'strict'.")
        
        metrics = []
        for y_true, y_prob in st.session_state['act3_results']:
            y_pred = (y_prob > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.append([tp/(tp+fn), tn/(tn+fp), tp/(tp+fp)])
        
        avg_m = np.mean(metrics, axis=0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sensitivity (Recall)", f"{avg_m[0]:.3f}", help="What % of all deaths did we successfully flag?")
        c2.metric("Specificity", f"{avg_m[1]:.3f}", help="What % of all survivors did we successfully identify?")
        c3.metric("Precision", f"{avg_m[2]:.3f}", help="When we flag a death, how often are we actually correct?")

# ==========================================
# ACTIVITY 4: STRATEGIC COMPARISON
# ==========================================
elif activity == "Activity 4: Strategic Comparison":
    st.title("Activity 4: Model Strategy")
    
    with st.expander("Activity Guide: Final Assessment", expanded=True):
        st.write("1. **Compare Models:** Review the visual differences between Trees and CNNs.")
        st.write("2. **Select Priority:** Move the slider to reflect your specific deployment goals.")
        st.write("3. **Final Decision:** Compare Interpretability vs Performance.")

    # Placeholder for image: [Image comparing decision tree architecture to neural network architecture]
    st.subheader("Decision Matrix")
    
    priority = st.select_slider("Select Core Requirement:", options=["Interpretability", "Balanced", "Performance"], 
                                help="Interpretability favors Trees; Performance favors CNNs.")
    
    if priority == "Interpretability":
        st.info("Strategy: Use the Decision Tree. Clinician trust relies on understanding the exact 'If-Then' logic.")
    elif priority == "Performance":
        st.success("Strategy: Use the 1D CNN. Raw detection power is the highest priority for patient safety.")
    else:
        st.warning("Strategy: Hybrid approach required.")
        
    st.bar_chart(pd.DataFrame({
        'Metric': ['Interpretability', 'Raw Performance', 'Automation'],
        'Decision Tree': [9, 5, 2],
        '1D CNN': [2, 9, 8]
    }).set_index('Metric'))
