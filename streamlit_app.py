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
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Spikes during CNN training.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Current memory footprint.")

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# --- TRACK SELECTION ---
st.sidebar.header("Select Your Track")
track = st.sidebar.radio("Focus Area", ["Clinical Science", "Foundational Science"], 
                         help="Toggle between patient-care focus and algorithmic-mechanism focus.")
st.sidebar.markdown("---")

# --- NAVIGATION ---
activity = st.sidebar.radio("Navigation", [
    "Activity 1: Data Exploration",
    "Activity 2: Base Performance",
    "Activity 3: Advanced Metrics",
    "Activity 4: Model Comparison"
])

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
# ACTIVITY 1
# ==========================================
if activity == "Activity 1: Data Exploration":
    st.title("Activity 1: eICU Data Exploration")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("1. Examine the statistical distributions of ICU vital signs.")
        st.write("2. Identify how clinical features correlate with In-Hospital Mortality.")
        st.write("3. Determine the outcome variable and data types.")

    if track == "Clinical Science":
        st.info("Clinical Focus: Identify risk factors for mortality to support ICU triage teams.")
    else:
        st.info("Foundational Focus: Analyze class imbalance to understand bias in loss functions.")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### Class Distribution")
        # 
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(class_counts, color="#FF4B4B")
        st.caption("Binary Outcome Distribution (0: Survival, 1: Death)")
        
    with col2:
        st.markdown("### Feature Distribution")
        feature_to_plot = st.selectbox("Select Feature:", df.columns[:-1])
        feature_means = df.groupby('Outcome')[feature_to_plot].mean().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(feature_means)
        st.caption(f"Mean {feature_to_plot} per Outcome Group")

    st.markdown("### Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# ACTIVITY 2
# ==========================================
elif activity == "Activity 2: Base Performance":
    st.title("Activity 2: Training & Accuracy")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("1. Adjust Hyperparameters in the sidebar.")
        st.write("2. Click 'Train Single Fold' to run the 1D CNN.")
        st.write("3. Evaluate if 'Total Accuracy' is a safe metric for mortality prediction.")

    st.sidebar.header("Model Hyperparameters")
    epochs = st.sidebar.slider("Epochs", 10, 50, 20)
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("1D CNN Training")
        # 
        if st.button("Train Single Fold"):
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            
            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).reshape(len(X_train), X.shape[1], 1)
            X_val = scaler.transform(X_val).reshape(len(X_val), X.shape[1], 1)
            
            model = Sequential([
                Input(shape=(X.shape[1], 1)),
                Conv1D(32, kernel_size=2, activation='relu'),
                Flatten(),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            with st.spinner("Training..."):
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                    epochs=epochs, batch_size=batch_size, verbose=0)
            
            y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
            acc = np.mean(y_pred == y_val)
            
            st.session_state['history'] = history.history
            st.session_state['acc'] = acc
            st.success("Training Complete")
            
    with col2:
        if 'history' in st.session_state:
            st.subheader("Training Metrics")
            st.line_chart(pd.DataFrame(st.session_state['history'])[['accuracy', 'val_accuracy']])
            st.metric("Total Accuracy", f"{st.session_state['acc']*100:.2f}%")

# ==========================================
# ACTIVITY 3
# ==========================================
elif activity == "Activity 3: Advanced Metrics":
    st.title("Activity 3: Clinical Evaluation")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("1. Run the K-Fold Evaluation.")
        st.write("2. Move the threshold slider to see the trade-off in the confusion matrix.")

    if st.button("Run Full K-Fold Evaluation"):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx]).reshape(len(train_idx), X.shape[1], 1)
            X_val = scaler.transform(X[val_idx]).reshape(len(val_idx), X.shape[1], 1)
            
            model = Sequential([
                Input(shape=(X.shape[1], 1)),
                Conv1D(32, kernel_size=2, activation='relu'),
                Flatten(),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy')
            model.fit(X_train, y[train_idx], epochs=10, verbose=0)
            
            y_prob = model.predict(X_val, verbose=0)
            results.append((y[val_idx], y_prob))
        
        st.session_state['cv_results'] = results
        st.success("K-Fold Complete")

    if 'cv_results' in st.session_state:
        # 
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5)
        
        metrics = []
        for y_true, y_prob in st.session_state['cv_results']:
            y_pred = (y_prob > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.append([tp/(tp+fn), tn/(tn+fp), tp/(tp+fp)])
        
        avg_m = np.mean(metrics, axis=0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Sensitivity (Recall)", f"{avg_m[0]:.3f}")
        c2.metric("Specificity", f"{avg_m[1]:.3f}")
        c3.metric("Precision", f"{avg_m[2]:.3f}")

# ==========================================
# ACTIVITY 4
# ==========================================
elif activity == "Activity 4: Model Comparison":
    st.title("Activity 4: CNN vs Decision Tree")
    
    # 
    st.subheader("Model Characteristics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decision Tree (Milestone 1)**")
        st.write("- Logic: 'If Glucose > 180...'")
        st.write("- Advantage: Fully Interpretable")
    with col2:
        st.markdown("**1D CNN (Deep Learning)**")
        st.write("- Logic: Automated Feature Extraction")
        st.write("- Advantage: High Predictive Performance")

    st.markdown("---")
    st.subheader("Comparative Scaling Matrix")
    comp_df = pd.DataFrame({
        'Metric': ['Interpretability', 'Performance', 'Automation'],
        'Decision Tree': [9, 5, 2],
        '1D CNN': [2, 9, 8]
    }).set_index('Metric')
    st.bar_chart(comp_df)
