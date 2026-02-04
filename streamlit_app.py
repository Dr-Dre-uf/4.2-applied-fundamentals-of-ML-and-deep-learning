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
    # interval=0.1 is crucial during training to catch the peaks
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚀 Training Monitor")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Usage", f"{cpu_percent}%")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB")

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# Navigation
activity = st.sidebar.radio("Navigation", [
    "Activity 1: Scenario",
    "Activity 2 & 3: Interactive CNN",
    "Activity 4: Comparison"
])

# Display the monitor in the sidebar
display_system_monitor()

@st.cache_data
def load_data():
    try:
        # Attempt to load local file
        df = pd.read_csv("diabetes.csv")
    except:
        # Fallback to sklearn dataset if file not found
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        # Create a synthetic binary outcome for classification
        df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
        df.drop(columns='target', inplace=True)
    return df

df = load_data()

# Activity 1
if activity == "Activity 1: Scenario":
    st.title("Activity 1: Applied Fundamentals of ML")
    st.info("**Instructions:** Read the scenario below to understand the clinical context of this ML task.")
    
    st.header("Clinical Scenario")
    st.write("""
    You are part of a hospital’s clinical analytics team using a **CNN model** to predict **in-hospital mortality** using data from the eICU Collaborative Research Database. The dataset includes patient demographics 
    and selected lab results such as glucose, creatinine, and potassium.
    """)
    
    
    
    st.subheader("Task Details")
    st.write("- **Job Task:** Binary Classification (Predicting 0 for Survival, 1 for Death).")
    st.write("- **CNN Advantage:** 1D CNNs can identify complex relationships between different lab values and demographics without manual feature engineering.")

# Activity 2 & 3
elif activity == "Activity 2 & 3: Interactive CNN":
    st.title("Activity 2 & 3: Training and Metrics")
    
    st.success("""
    **Instructions:** 1. Adjust the **Hyperparameters** in the sidebar.
    2. Click **'Train Next Fold'** to execute one segment of 5-fold cross-validation.
    3. Watch the **Training Monitor** in the sidebar to see the CPU hit during training!
    """)

    # Interactive Sidebar for Hyperparameters
    st.sidebar.header("Model Hyperparameters")
    filt1 = st.sidebar.slider("Conv1D Layer 1 Filters", 16, 64, 32, help="Learns patterns from raw clinical data.")
    filt2 = st.sidebar.slider("Conv1D Layer 2 Filters", 32, 128, 64, help="Learns complex relationships from the first layer.")
    drop_val = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.3, help="Prevents the model from memorizing the data (overfitting).")
    lr = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.001, 0.0001], value=0.001)

    if 'cv_results' not in st.session_state:
        st.session_state.cv_results = []
    if 'current_fold' not in st.session_state:
        st.session_state.current_fold = 0

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("Train Next Fold"):
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kf.split(X))

            if st.session_state.current_fold < 5:
                train_idx, val_idx = folds[st.session_state.current_fold]
                
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X[train_idx]).reshape(len(train_idx), X.shape[1], 1)
                X_val = scaler.transform(X[val_idx]).reshape(len(val_idx), X.shape[1], 1)
                y_train, y_val = y[train_idx], y[val_idx]

                # CNN Architecture
                model = Sequential([
                    Input(shape=(X.shape[1], 1)),
                    Conv1D(filt1, kernel_size=2, activation='relu'),
                    Conv1D(filt2, kernel_size=2, activation='relu'),
                    Flatten(),
                    Dense(32, activation='relu'),
                    Dropout(drop_val),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
                
                with st.spinner(f"Training Fold {st.session_state.current_fold + 1}..."):
                    # The CPU usage will peak here
                    model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=0)
                
                y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
                
                st.session_state.cv_results.append({
                    'Fold': st.session_state.current_fold + 1,
                    'Accuracy': (tp+tn)/(tp+tn+fp+fn),
                    'Sensitivity': tp/(tp+fn) if (tp+fn)>0 else 0,
                    'Specificity': tn/(tn+fp) if (tn+fp)>0 else 0,
                    'Precision': tp/(tp+fp) if (tp+fp)>0 else 0
                })
                st.session_state.current_fold += 1
                st.rerun()
            else:
                st.success("5-Fold Cross Validation Complete.")

        if btn_col2.button("Reset Session"):
            st.session_state.cv_results = []
            st.session_state.current_fold = 0
            st.rerun()

    with col2:
        st.subheader("Performance Metrics")
        if st.session_state.cv_results:
            res_df = pd.DataFrame(st.session_state.cv_results).set_index('Fold')
            st.table(res_df.style.format("{:.3f}"))
            
            st.write("**Average Metrics Across Folds**")
            avg_df = res_df.mean().to_frame(name="Value").T
            st.dataframe(avg_df.style.format("{:.3f}"))
        else:
            st.info("No data yet. Click 'Train Next Fold' to begin.")
            
        

# Activity 4
elif activity == "Activity 4: Comparison":
    st.title("Activity 4: Model Comparison")
    st.info("**Instructions:** Review the trade-offs between model types for clinical settings.")
    
    st.subheader("CNN vs. Decision Tree")
    st.write("""
    - **Decision Trees:** Highly interpretable. Doctors can follow the logic ('If glucose > 200...').
    - **CNNs:** Superior performance on high-dimensional data but function as a 'Black Box'.
    """)
    
    st.warning("Interpretability is often a regulatory requirement in clinical AI deployment.")