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
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Indicates the computational load during gradient optimization.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Total memory currently allocated to the dataset and model architecture.")

st.set_page_config(page_title="Applied Fundamentals of Deep Learning", layout="wide")

# --- GLOBAL INSTRUCTIONS ---
st.markdown("""
> **Module Instructions:** Complete each activity in order. In the sidebar, toggle between the Clinical Science and Foundational Science perspectives to observe how the identical pipeline is interpreted differently based on specific scientific goals. Record your responses to the module questions in your Canvas submission area or notebook.
""")
st.markdown("---")

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
], help="Navigate through the core components of the machine learning pipeline.")

display_system_monitor()

@st.cache_data
def load_data():
    try:
        # Prioritize the local data directory to match the notebook environment
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
    st.title("Activity 1: Data Exploration")
    
    with st.expander("Activity Guide: Data Exploration", expanded=True):
        st.write("1. Inspect the clinical features (inputs) and the binary outcome variable (mortality).")
        st.write("2. Analyze the data distribution to establish the baseline mortality rate and identify potential class imbalances.")

    st.markdown("### Interactive Data Exploration")
    
    feature_cols = [col for col in df.columns if col != 'Outcome']
    feature_to_view = st.selectbox("Select a Clinical Feature to Analyze:", feature_cols, 
                                   help="Analyze how this clinical metric correlates with patient outcomes.")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Outcome Distribution**")
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(class_counts, color="#FF4B4B")
        
        # ADA Compliance Text Summary
        st.write(f"**Data Summary:** The dataset contains {class_counts.iloc[0]} Survival records and {class_counts.iloc[1]} Death records, indicating a severe class imbalance.")
        
    with col2:
        st.markdown(f"**Mean {feature_to_view} by Outcome**")
        feature_means = df.groupby('Outcome')[feature_to_view].mean()
        st.bar_chart(feature_means)
        
        # ADA Compliance Text Summary
        st.write(f"**Data Summary:** The calculated mean {feature_to_view} for Survivors is {feature_means.iloc[0]:.2f}, whereas the mean for Mortality cases is {feature_means.iloc[1]:.2f}.")

    with st.expander("Reveal: Conceptual Insights for Activity 1"):
        if track == "Clinical Science":
            st.info("""
            **Understanding the Data Format:** Prior to ingestion, features are standardized (scaled). In predictive modeling, a feature with large numerical values like Glucose (e.g., 148) could mathematically overpower a feature with small values like DiabetesPedigreeFunction (e.g., 0.627). Standardization transforms the data so all features are evaluated proportionally.
            
            **The Algorithmic Advantage:** A Deep Neural Network considers not only the individual parameters but the complex interactions among them. For example, a slightly lower blood pressure measurement may not be critical in isolation, but when combined with other specific risk factors, the network can flag the interaction as highly dangerous.
            """)
        else:
            st.info("""
            **Understanding the Data Format:** Features are standardized to a mean of 0 and a variance of 1. This prevents features with larger numerical magnitudes from dominating the gradient updates during backpropagation, ensuring stable convergence.
            
            **The Algorithmic Advantage:** The Deep Neural Network provides automated, non-linear feature extraction across multiple dense layers. This eliminates the need for the manual feature engineering required by traditional baseline models.
            """)

# ==========================================
# ACTIVITY 2: TRAINING AND BASE METRICS
# ==========================================
elif activity == "Activity 2: Training and Base Metrics":
    st.title("Activity 2: Model Optimization")
    
    with st.expander("Activity Guide: Model Optimization", expanded=True):
        st.write("1. Configure the optimization parameters (Epochs and Batch Size).")
        st.write("2. Execute the Deep Neural Network training pipeline.")
        st.write("3. Evaluate the resulting learning curve and overall global accuracy.")

    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 5, 50, 50, help="Determines the number of complete passes through the training dataset.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32], value=16, help="Determines the number of samples processed before the model updates its internal weights.")

    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.subheader("Deep Neural Network Architecture")
        
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
        
        if st.button("Execute Training", help="Initialize weights and begin the gradient optimization process."):
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
            
            with st.spinner("Executing model training..."):
                history = model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
            
            st.session_state['act2_history'] = history.history
            st.success("Training Complete")
            
    with col2:
        if 'act2_history' in st.session_state:
            st.subheader("Model Learning Curve")
            st.line_chart(pd.DataFrame(st.session_state['act2_history'])['accuracy'])
            
            final_acc = st.session_state['act2_history']['accuracy'][-1]
            st.metric("Final Global Accuracy", f"{final_acc:.2%}", help="The overall proportion of correct predictions across all classes.")
            
            # ADA Compliance Text Summary
            st.write(f"**Data Summary:** The model converged with a final global training accuracy of {final_acc:.2%}.")
            
            with st.expander("Reveal: Conceptual Insights for Activity 2"):
                st.warning("""
                **Evaluating Performance:** Maximizing epochs and minimizing batch size frequently yields the highest training accuracy. However, this often leads to 'overfitting', a scenario where the model memorizes the training data but fails to generalize to new, unseen patient records.
                
                **The Metric Problem:** In an imbalanced dataset (where the vast majority of instances belong to the 'Survival' class), total accuracy is highly deceptive. A model could predict 'Survival' for every patient and achieve high accuracy without successfully detecting a single mortality case. Robust evaluation requires analyzing Sensitivity, Specificity, and the F1 Score.
                """)

# ==========================================
# ACTIVITY 3: EVALUATION TRADE-OFFS
# ==========================================
elif activity == "Activity 3: Evaluation Trade-offs":
    st.title("Activity 3: Cross-Validation Analysis")
    
    with st.expander("Activity Guide: Cross-Validation Analysis", expanded=True):
        st.write("1. Execute the 5-fold cross-validation analysis to generate robust evaluation metrics.")
        st.write("2. Adjust the classification threshold to observe the statistical trade-offs between Sensitivity (Recall), Specificity, and Precision.")

    if st.button("Run 5-Fold Evaluation", help="Execute 5-fold cross-validation to rigorously assess model generalization."):
        X = df.drop(columns=['Outcome']).values
        y = df['Outcome'].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            status_text.text(f"Processing Fold {fold + 1} of 5...")
            
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
        
        status_text.text("Cross-validation procedure complete.")
        st.session_state['act3_results'] = results
        st.success("Evaluation Metrics Generated")

    if 'act3_results' in st.session_state:
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5, help="Modifying this boundary alters the probability required to classify an instance as positive.")
        
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
        c1.metric("Avg Accuracy", f"{avg_m[0]:.3f}", help="Calculated as: (TP + TN) / Total Predictions")
        c2.metric("Avg Sensitivity", f"{avg_m[1]:.3f}", help="Calculated as: TP / (TP + FN)")
        c3.metric("Avg Specificity", f"{avg_m[2]:.3f}", help="Calculated as: TN / (TN + FP)")
        c4.metric("Avg Precision", f"{avg_m[3]:.3f}", help="Calculated as: TP / (TP + FP)")
        
        with st.expander("Reveal: Conceptual Insights for Activity 3"):
            st.info("""
            **The ROC Curve Connection:** Modifying the classification threshold practically simulates traversing a Receiver Operating Characteristic (ROC) curve. 
            
            Lowering the threshold effectively increases Average Sensitivity (ensuring more potential mortality cases are flagged) at the direct expense of Average Specificity (generating more false positives). In applied environments, researchers must identify the optimal operational threshold that maximizes safety without overwhelming clinical staff with false alarms.
            """)

# ==========================================
# ACTIVITY 4: STRATEGIC COMPARISON
# ==========================================
elif activity == "Activity 4: Strategic Comparison":
    st.title("Activity 4: Strategic Evaluation")
    
    with st.expander("Activity Guide: Final Assessment", expanded=True):
        st.write("1. Contrast the architectural capabilities of the Deep Neural Network with the Milestone 1 Decision Tree.")
        st.write("2. Determine the optimal algorithmic approach based on the organizational requirements for interpretability versus predictive performance.")

    st.subheader("Architectural Comparison Matrix")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decision Tree (Milestone 1)**")
        st.write("- **Logic:** Employs orthogonal, interpretable 'If-Then' splits.")
        st.write("- **Transparency:** High (White Box model).")
    with col2:
        st.markdown("**Deep Neural Network (Current)**")
        st.write("- **Logic:** Utilizes complex, non-linear combinations across multiple dense hidden layers.")
        st.write("- **Transparency:** Low (Black Box model).")
        
    st.markdown("---")
    
    priority = st.select_slider("Select Core Organizational Requirement:", options=["Interpretability", "Balanced", "Performance"], help="Select the primary requirement to reveal the recommended deployment strategy.")
    
    if priority == "Interpretability":
        st.info("Deployment Strategy: Utilize the Decision Tree. Clinical adoption often mandates high interpretability to ensure practitioners can validate the underlying logic of a prediction.")
    elif priority == "Performance":
        st.success("Deployment Strategy: Utilize the Deep Neural Network. Maximizing raw detection capability is prioritized to ensure high-risk patients are successfully triaged.")
    else:
        st.warning("Deployment Strategy: An ensemble approach or the application of post-hoc explainability frameworks (e.g., SHAP) is required to balance predictive power and transparency.")

    with st.expander("Reveal: Conceptual Insights for Activity 4"):
        st.success("""
        **The Core Algorithmic Trade-off:** The Deep Neural Network yields superior predictive performance because its hidden layers extract multidimensional representations of the input data that a linear decision boundary cannot capture. However, this complex structure creates a 'Black Box' phenomenon where the exact mathematical reasoning for a single prediction cannot be easily explained to a clinician or patient. 
        
        Data science teams must continuously evaluate the organizational trade-off between the high predictive capacity of deep learning models and the essential interpretability provided by traditional algorithms.
        """)
