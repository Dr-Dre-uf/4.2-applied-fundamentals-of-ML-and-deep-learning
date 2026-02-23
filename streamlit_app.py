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
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Spikes when the neural network is training.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Memory currently consumed by the application.")

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# --- TRACK SELECTION ---
st.sidebar.header("Select Your Track")
track = st.sidebar.radio("Focus Area", ["Clinical Science", "Foundational Science"], help="Changes the context and explanations provided throughout the app.")
st.sidebar.markdown("---")

# --- NAVIGATION ---
activity = st.sidebar.radio("Navigation", [
    "Activity 1: Data Exploration",
    "Activity 2: Base Performance",
    "Activity 3: Advanced Metrics",
    "Activity 4: Model Comparison"
], help="Navigate between the 4 core activities of this module.")

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
        st.write("1. Review the dataset class distribution to understand the mortality rate baseline.")
        st.write("2. Use the interactive dropdown to explore how different features correlate visually with the outcome.")
        st.write("3. Review the raw data table to understand the structure of the arrays feeding into the model.")

    if track == "Clinical Science":
        st.info("Clinical Focus: Understand the patient population and the prevalence of mortality within the ICU dataset.")
    else:
        st.info("Foundational Focus: Analyze the statistical distribution and class imbalance of the dataset.")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### Dataset Class Distribution")
        # 
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(class_counts, color="#FF4B4B")
        st.caption("Notice the heavy class imbalance (majority are Survivals).")
        
    with col2:
        st.markdown("### Interactive Feature Exploration")
        feature_to_plot = st.selectbox("Select a Feature:", df.columns[:-1], help="Visualize its distribution across survival and death.")
        
        feature_means = df.groupby('Outcome')[feature_to_plot].mean().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(feature_means)
        st.caption(f"Average {feature_to_plot} categorized by patient outcome.")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# ACTIVITY 2
# ==========================================
elif activity == "Activity 2: Base Performance":
    st.title("Activity 2: Evaluating Base Performance")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("1. Adjust the Epochs and Batch Size in the sidebar.")
        st.write("2. Click 'Train Single Fold' to train the 1D CNN.")
        st.write("3. Watch the Training Metrics chart populate live.")

    st.sidebar.header("Model Hyperparameters")
    epochs = st.sidebar.slider("Epochs", 10, 50, 20)
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Train the 1D CNN Model")
            
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
            
            with st.spinner("Training model..."):
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
            
            y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            acc = (tp+tn)/(tp+tn+fp+fn)
            
            st.session_state['act2_acc'] = acc
            st.session_state['act2_history'] = history.history
            st.session_state['act2_trained'] = True
            st.success("Training Complete!")
            
        with st.expander("View Network Architecture"):
            # 

[Image of 1D Convolutional Neural Network architecture]

            st.code("""
Sequential([
  Input(shape=(Features, 1)),
  Conv1D(32, kernel_size=2, activation='relu'),
  Flatten(),
  Dense(16, activation='relu'),
  Dense(1, activation='sigmoid')
])
            """, language='python')
            
    with col2:
        st.subheader("Training Metrics")
        if st.session_state.get('act2_trained', False):
            hist_df = pd.DataFrame({
                'Train Accuracy': st.session_state['act2_history']['accuracy'],
                'Val Accuracy': st.session_state['act2_history']['val_accuracy'],
            })
            
            st.line_chart(hist_df)
            st.metric("Final Total Accuracy", f"{st.session_state['act2_acc']*100:.1f}%")

# ==========================================
# ACTIVITY 3
# ==========================================
elif activity == "Activity 3: Advanced Metrics":
    st.title("Activity 3: Trade-off Metrics")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("1. Click 'Run Full K-Fold Evaluation'.")
        st.write("2. Once complete, slide the 'Probability Decision Threshold'.")

    st.subheader("5-Fold Cross Validation Evaluation")
    if st.button("Run Full K-Fold Evaluation"):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        progress_bar = st.progress(0)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx]).reshape(len(train_idx), X.shape[1], 1)
            X_val = scaler.transform(X[val_idx]).reshape(len(val_idx), X.shape[1], 1)
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = Sequential([
                Input(shape=(X.shape[1], 1)),
                Conv1D(32, kernel_size=2, activation='relu'),
                Flatten(),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)
            
            y_prob = model.predict(X_val, verbose=0)
            st.session_state[f'probs_fold_{fold}'] = (y_val, y_prob)
            progress_bar.progress((fold + 1) / 5)
            
        st.session_state['cv_done'] = True
        st.success("Evaluation complete.")

    if st.session_state.get('cv_done', False):
        st.markdown("---")
        threshold = st.slider("Probability Decision Threshold", 0.05, 0.95, 0.50, 0.05)
        
        metrics_list = []
        tn_total, fp_total, fn_total, tp_total = 0, 0, 0, 0
        
        for fold in range(5):
            y_val, y_prob = st.session_state[f'probs_fold_{fold}']
            y_pred = (y_prob > threshold).astype(int).flatten()
            
            if len(np.unique(y_val)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
                
            tn_total += tn
            fp_total += fp
            fn_total += fn
            tp_total += tp
                
            metrics_list.append({
                'Sensitivity (Recall)': tp/(tp+fn) if (tp+fn)>0 else 0,
                'Specificity': tn/(tn+fp) if (tn+fp)>0 else 0,
                'Precision': tp/(tp+fp) if (tp+fp)>0 else 0
            })
            
        avg_df = pd.DataFrame(metrics_list).mean()
        
        st.markdown("### Metrics Summary")
        # 
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Negatives (TN)", tn_total)
        c2.metric("False Positives (FP)", fp_total)
        c3.metric("False Negatives (FN)", fn_total)
        c4.metric("True Positives (TP)", tp_total)
        
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.metric("Avg Sensitivity (Recall)", f"{avg_df['Sensitivity (Recall)']:.3f}")
            st.metric("Avg Specificity", f"{avg_df['Specificity']:.3f}")
            st.metric("Avg Precision", f"{avg_df['Precision']:.3f}")
            
        with col2:
            chart_data = pd.DataFrame({
                "Score": [avg_df['Sensitivity (Recall)'], avg_df['Specificity'], avg_df['Precision']]
            }, index=["Sensitivity", "Specificity", "Precision"])
            st.bar_chart(chart_data)

# ==========================================
# ACTIVITY 4
# ==========================================
elif activity == "Activity 4: Model Comparison":
    st.title("Activity 4: CNN vs Decision Tree")
    
    with st.expander("Activity Instructions", expanded=True):
        st.write("Determine which model to deploy based on your priorities.")
    
    # 
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Decision Trees")
        if track == "Clinical Science":
            st.write("- **Interpretability:** High. Logic is easy to follow.")
        else:
            st.write("- **Architecture:** Orthogonal decision boundaries.")
            
    with col2:
        st.markdown("### 1D CNNs")
        if track == "Clinical Science":
            st.write("- **Interpretability:** Low (Black Box).")
        else:
            st.write("- **Architecture:** Automated high-dimensional feature extraction.")
    
    st.markdown("---")
    
    st.subheader("Interactive Model Selection Tool")
    weight_perf = st.slider("Importance of Raw Performance", 1, 10, 5)
    weight_interp = st.slider("Importance of Interpretability", 1, 10, 5)
    
    dt_score = (5 * weight_perf) + (9 * weight_interp)
    cnn_score = (9 * weight_perf) + (2 * weight_interp)
    
    if cnn_score > dt_score:
        st.success(f"Recommendation: 1D CNN (Score: {cnn_score})")
    elif dt_score > cnn_score:
        st.info(f"Recommendation: Decision Tree (Score: {dt_score})")
    else:
        st.warning("Recommendation: Tied!")
        
    st.markdown("---")
    
    comp_df = pd.DataFrame({
        'Metric': ['Interpretability', 'Raw Performance', 'Automated Extraction'],
        'Decision Tree': [9, 5, 2],
        '1D CNN': [2, 9, 8]
    }).set_index('Metric')
    
    st.bar_chart(comp_df)
