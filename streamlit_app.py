import streamlit as st
import pandas as pd
import numpy as np
import psutil
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# --- MONITORING UTILITY ---
def display_system_monitor():
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Monitor")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Spikes during DNN training.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Total memory currently consumed.")

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
], help="Navigate through the core components of the ML pipeline.")

display_system_monitor()

@st.cache_data
def load_data():
    try:
        # Prioritize the local diabetes.csv file to match the notebook
        df = pd.read_csv("data/diabetes.csv")
    except:
        try:
            df = pd.read_csv("diabetes.csv")
        except:
            # Fallback if CSV is missing
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
    
    with st.expander("Activity Guide: Data Exploration", expanded=True):
        st.write("1. Review the outcome distribution to establish the mortality rate baseline.")
        st.write("2. Analyze how specific clinical metrics correlate with patient outcomes.")

    st.markdown("### Interactive Data Exploration")
    
    # Filter to only show feature columns (excluding Outcome)
    feature_cols = [col for col in df.columns if col != 'Outcome']
    feature_to_view = st.selectbox("Select a Clinical Feature to Analyze:", feature_cols, 
                                   help="Analyze how this clinical metric correlates with patient outcomes.")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Outcome Distribution**")
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(class_counts, color="#FF4B4B")
        
        # ADA Compliance Text Summary
        st.write(f"**Data Summary:** There are {class_counts.iloc[0]} Survival records and {class_counts.iloc[1]} Death records. This represents a significant class imbalance.")
        
    with col2:
        st.markdown(f"**Mean {feature_to_view} by Outcome**")
        feature_means = df.groupby('Outcome')[feature_to_view].mean()
        st.bar_chart(feature_means)
        
        # ADA Compliance Text Summary
        st.write(f"**Data Summary:** The average {feature_to_view} for Survivors is {feature_means.iloc[0]:.2f}, while the average for Deaths is {feature_means.iloc[1]:.2f}.")

    with st.expander("Reveal: Conceptual Insights for Activity 1"):
        if track == "Clinical Science":
            st.info("""
            **Understanding the Data Format:** You may notice that when the data is fed into the model, the numbers look very small (e.g., between -0.05 and 0.05). This is because the data is 'Standardized'. In predictive modeling, a large number like Glucose (e.g., 148) could overpower a small number like DiabetesPedigreeFunction (e.g., 0.627). Standardizing transforms the data so all features are treated equally.
            
            **The DNN Advantage:** A Deep Neural Network considers not only the individual parameters but the complex relationships among them. For example, a slightly lower blood pressure measurement may not be a problem for an otherwise healthy person, but if combined with other specific risk factors, it can be flagged as dangerous.
            """)
        else:
            st.info("""
            **Understanding the Data Format:** Prior to model ingestion, features are scaled to a mean of 0 and a variance of 1. This prevents features with larger numerical magnitudes from dominating the gradient updates during backpropagation.
            
            **The DNN Advantage:** The DNN provides automated, non-linear feature extraction across multiple hidden layers, eliminating the need for manual feature engineering required by traditional baseline models.
            """)

# ==========================================
# ACTIVITY 2: TRAINING AND BASE METRICS
# ==========================================
elif activity == "Activity 2: Training and Base Metrics":
    st.title("Activity 2: Training and Base Metrics")
    
    with st.expander("Activity Guide: Model Optimization", expanded=True):
        st.write("1. Set the Epochs and Batch Size parameters.")
        st.write("2. Execute the DNN training to observe the optimization process.")
        st.write("3. Review the learning curve and final accuracy score.")

    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 5, 50, 50, help="More epochs allow the model to learn longer.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32], value=16, help="Smaller batches make training updates more frequent.")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("DNN Configuration")
        
        st.code("""
model = Sequential([
    Input(shape=(X_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
        """, language='python')
        
        if st.button("Execute Training", help="Click to train the Deep Neural Network."):
            X = df.drop(columns=['Outcome']).values
            y = df['Outcome'].values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = Sequential([
                Input(shape=(X_scaled.shape[1],)),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            
            with st.spinner("Training model..."):
                history = model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
            
            st.session_state['act2_history'] = history.history
            st.success("Training Complete")
            
    with col2:
        if 'act2_history' in st.session_state:
            st.subheader("Model Learning Curve")
            st.line_chart(pd.DataFrame(st.session_state['act2_history'])['accuracy'])
            
            final_acc = st.session_state['act2_history']['accuracy'][-1]
            st.metric("Final Total Accuracy", f"{final_acc:.2%}", help="The overall percentage of correct predictions.")
            
            # ADA Compliance Text Summary
            st.write(f"**Data Summary:** The model achieved a final global training accuracy of {final_acc:.2%}.")
            
            with st.expander("Reveal: Conceptual Insights for Activity 2"):
                st.warning("""
                **Evaluating Performance:** Setting epochs to the maximum and batch size to the minimum often yields the highest training accuracy. However, in practice, this can lead to 'overfitting', where the model memorizes the training data but fails on new patients.
                
                **The Metric Problem:** In an imbalanced dataset (where the vast majority survive), total accuracy is misleading. A model could simply predict 'Survival' for everyone and achieve high accuracy without successfully detecting a single mortality case. For a true evaluation, metrics like Sensitivity, Specificity, and the F1 Score must be considered.
                """)

# ==========================================
# ACTIVITY 3: EVALUATION TRADE-OFFS
# ==========================================
elif activity == "Activity 3: Evaluation Trade-offs":
    st.title("Activity 3: Advanced Clinical Metrics")
    
    with st.expander("Activity Guide: Cross-Validation", expanded=True):
        st.write("1. Generate rigorous predictions using 5-Fold Evaluation.")
        st.write("2. Shift the classification threshold to observe the inverse relationship between Sensitivity and Specificity.")

    if st.button("Run 5-Fold Evaluation", help="Execute 5-fold cross-validation to rigorously test the model."):
        X = df.drop(columns=['Outcome']).values
        y = df['Outcome'].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            status_text.text(f"Training Fold {fold + 1} of 5...")
            
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[train_idx])
            X_val = scaler.transform(X[val_idx])
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = Sequential([
                Input(shape=(X_train.shape[1],)),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(64, activation='relu'),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
            
            model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
            
            y_prob = model.predict(X_val, verbose=0)
            results.append((y_val, y_prob))
            
            progress_bar.progress((fold + 1) / 5)
        
        status_text.text("Cross-validation complete!")
        st.session_state['act3_results'] = results
        st.success("Full Evaluation Generated")

    if 'act3_results' in st.session_state:
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, help="Adjusting this changes the definition of a positive prediction.")
        
        metrics = []
        for y_true, y_prob in st.session_state['act3_results']:
            y_pred = (y_prob > threshold).astype(int).flatten()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            metrics.append([acc, sens, spec, prec])
        
        avg_m = np.mean(metrics, axis=0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Accuracy", f"{avg_m[0]:.3f}", help="Total Correct / Total Predictions")
        c2.metric("Avg Sensitivity", f"{avg_m[1]:.3f}", help="True Positives / Actual Positives")
        c3.metric("Avg Specificity", f"{avg_m[2]:.3f}", help="True Negatives / Actual Negatives")
        c4.metric("Avg Precision", f"{avg_m[3]:.3f}", help="True Positives / Predicted Positives")
        
        with st.expander("Reveal: Conceptual Insights for Activity 3"):
            st.info("""
            **The ROC Curve Connection:** Adjusting the threshold above is the practical equivalent of analyzing a Receiver Operating Characteristic (ROC) curve. 
            
            Lowering the threshold increases Average Sensitivity (flagging more potential mortality cases) but decreases Average Specificity (generating more false positives). In a clinical environment, researchers must identify the optimal balance to ensure safety without overwhelming staff with false alarms.
            """)

# ==========================================
# ACTIVITY 4: STRATEGIC COMPARISON
# ==========================================
elif activity == "Activity 4: Strategic Comparison":
    st.title("Activity 4: Model Strategy")
    
    with st.expander("Activity Guide: Final Assessment", expanded=True):
        st.write("Analyze the architectural differences between the models to determine the best deployment strategy.")

    st.subheader("Decision Matrix")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decision Tree (Milestone 1)**")
        st.write("- **Logic:** Interpretable 'If-Then' branches.")
        st.write("- **Transparency:** High (White Box).")
    with col2:
        st.markdown("**Deep Neural Network (DNN)**")
        st.write("- **Logic:** Complex non-linear combinations across multiple dense layers.")
        st.write("- **Transparency:** Low (Black Box).")
        
    st.markdown("---")
    
    priority = st.select_slider("Select Core Requirement:", options=["Interpretability", "Balanced", "Performance"], help="Slide to reveal the recommended algorithm based on organizational goals.")
    
    if priority == "Interpretability":
        st.info("Strategy: Use the Decision Tree. Clinician trust relies on understanding the exact rules governing the prediction.")
    elif priority == "Performance":
        st.success("Strategy: Use the DNN. Raw detection power is the highest priority for accurate patient triage.")
    else:
        st.warning("Strategy: A hybrid or ensemble approach is required to balance power and transparency.")

    with st.expander("Reveal: Conceptual Insights for Activity 4"):
        st.success("""
        **The Core Trade-off:** The DNN yields superior performance because its layers can extract multidimensional representations of the data that a simple decision boundary cannot. However, this structure creates a 'Black Box' where the exact reasoning for a single prediction cannot be easily explained to a clinician or patient. 
        
        Organizations must choose between the high predictive power of the DNN and the interpretability of traditional decision trees.
        """)
