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
    c1.metric("CPU Usage", f"{cpu_percent}%")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB")

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# --- TRACK SELECTION ---
st.sidebar.header("Select Your Track")
track = st.sidebar.radio("Focus Area", ["Clinical Science", "Foundational Science"])
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
    
    if track == "Clinical Science":
        st.info("Clinical Focus: Understand the patient population and the prevalence of mortality within the ICU dataset to inform triage strategies.")
        # Placeholder for clinical architecture image
        # st.image("clinical_architecture.png", caption="Clinical decision support system architecture")
    else:
        st.info("Foundational Focus: Analyze the statistical distribution and class imbalance of the dataset to anticipate how it will affect the neural network's loss function.")
        # Placeholder for 1D CNN architecture image
        # st.image("cnn_architecture.png", caption="1D Convolutional Neural Network architecture")
    
    st.markdown("### Dataset Class Distribution")
    class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
    st.bar_chart(class_counts, color="#FF4B4B")
    
    if track == "Clinical Science":
        st.caption("Notice the heavy class imbalance. The vast majority of records represent 'Survival', which is typical in general ICU admissions.")
    else:
        st.caption("Notice the heavy class imbalance. This skew will heavily bias a standard binary cross-entropy loss function during gradient descent if not corrected.")

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ==========================================
# ACTIVITY 2
# ==========================================
elif activity == "Activity 2: Base Performance":
    st.title("Activity 2: Evaluating Base Performance")
    
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
            st.success("Training Complete")
            
    with col2:
        st.subheader("Training Metrics")
        if st.session_state.get('act2_trained', False):
            hist_df = pd.DataFrame({
                'Train Accuracy': st.session_state['act2_history']['accuracy'],
                'Val Accuracy': st.session_state['act2_history']['val_accuracy'],
            })
            
            st.line_chart(hist_df)
            st.metric("Final Total Accuracy", f"{st.session_state['act2_acc']*100:.1f}%")
            
            if track == "Clinical Science":
                st.warning("Clinical Context: A model that simply predicts 'Survival' for everyone will achieve high total accuracy but miss 100% of the at-risk patients. Accuracy alone is an insufficient metric for clinical deployment.")
            else:
                st.warning("Foundational Context: Due to the imbalanced prior probabilities of the classes, global accuracy masks the model's inability to establish a proper decision boundary for the minority class.")

# ==========================================
# ACTIVITY 3
# ==========================================
elif activity == "Activity 3: Advanced Metrics":
    st.title("Activity 3: Trade-off Metrics")
    
    if track == "Clinical Science":
        st.info("Adjust the threshold to balance catching at-risk patients (Sensitivity) against burdening staff with false alarms (Specificity).")
    else:
        st.info("Adjust the decision threshold to observe the mathematical trade-off in the confusion matrix space (Recall vs Precision/Specificity).")

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
        for fold in range(5):
            y_val, y_prob = st.session_state[f'probs_fold_{fold}']
            y_pred = (y_prob > threshold).astype(int).flatten()
            
            if len(np.unique(y_val)) > 1:
                tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
                
            metrics_list.append({
                'Sensitivity (Recall)': tp/(tp+fn) if (tp+fn)>0 else 0,
                'Specificity': tn/(tn+fp) if (tn+fp)>0 else 0,
                'Precision': tp/(tp+fp) if (tp+fp)>0 else 0
            })
            
        avg_df = pd.DataFrame(metrics_list).mean()
        
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
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Decision Trees")
        if track == "Clinical Science":
            st.write("- **Interpretability:** Highly interpretable. Doctors can easily follow the clinical logic.")
            st.write("- **Adoption:** Easier to get approved by hospital review boards.")
        else:
            st.write("- **Architecture:** Partitions feature space using orthogonal decision boundaries.")
            st.write("- **Feature Engineering:** Requires manual feature creation to capture complex interactions.")
            
    with col2:
        st.markdown("### 1D CNNs")
        if track == "Clinical Science":
            st.write("- **Interpretability:** Functions as a 'Black Box'. Hard to mathematically explain the prediction to a patient.")
            st.write("- **Adoption:** Strict regulatory hurdles for clinical AI deployment.")
        else:
            st.write("- **Architecture:** Uses convolutional filters to extract high-dimensional representations automatically.")
            st.write("- **Feature Engineering:** Learns non-linear patterns directly from raw data arrays.")
    
    st.markdown("---")
    
    comp_df = pd.DataFrame({
        'Metric': ['Interpretability', 'Raw Performance', 'Automated Feature Extraction'],
        'Decision Tree': [9, 5, 2],
        '1D CNN': [2, 9, 8]
    }).set_index('Metric')
    
    st.bar_chart(comp_df)
    
    if track == "Clinical Science":
        st.error("Clinical Reality: Interpretability is often a regulatory requirement. Hospitals may choose a slightly less accurate Decision Tree over a CNN if they cannot legally deploy a 'black box' model.")
    else:
        st.error("Foundational Reality: While CNNs boast superior automated feature extraction, their complex mapping limits algorithmic transparency, posing challenges for post-hoc error analysis.")
