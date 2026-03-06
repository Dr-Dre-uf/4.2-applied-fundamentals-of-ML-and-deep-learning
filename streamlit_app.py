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
        df = pd.read_csv("diabetes.csv")
    except:
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        df = data.frame.copy()
        df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
        df.drop(columns='target', inplace=True)
    
    mapping = {
        'age': 'Age', 'bmi': 'BMI', 'bp': 'BloodPressure', 
        'Pregnancies': 'Pregnancies', 'Glucose': 'Glucose', 
        'SkinThickness': 'SkinThickness', 'Insulin': 'Insulin',
        'DiabetesPedigreeFunction': 'DiabetesPedigreeFunction'
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
        st.write("1. **Explore the Distribution:** Check the 'Outcome Distribution' chart to see the mortality rate baseline.")
        st.write("2. **Compare Features:** Use the dropdown to see how specific features differ between outcomes.")

    st.markdown("### Interactive Data Exploration")
    feature_to_view = st.selectbox("Select a Clinical Feature to Analyze:", df.columns[:-1], 
                                   help="Analyze how this clinical metric correlates with patient outcomes.")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Outcome Distribution**")
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(class_counts, color="#FF4B4B")
        st.write(f"**Data Summary:** There are {class_counts.iloc[0]} Survival records and {class_counts.iloc[1]} Death records. This represents a significant class imbalance.")
        
    with col2:
        st.markdown(f"**Mean {feature_to_view} by Outcome**")
        feature_means = df.groupby('Outcome')[feature_to_view].mean()
        st.bar_chart(feature_means)
        st.write(f"**Data Summary:** The average {feature_to_view} for Survivors is {feature_means.iloc[0]:.2f}, while the average for Deaths is {feature_means.iloc[1]:.2f}.")

    with st.expander("Reveal: Why are the numbers on the Y-Axis so small?"):
        st.info("""
        **Standardized Data:** If you see numbers ranging from -0.05 to 0.05 for factors like Age or BMI, it means the data has been scaled. 
        In Machine Learning, if we don't scale the data, large numbers (like Glucose = 150) will overpower smaller numbers (like BMI = 25). 
        The data is transformed so the average is 0, allowing the neural network to treat all features equally.
        """)

# ==========================================
# ACTIVITY 2: TRAINING AND BASE METRICS
# ==========================================
elif activity == "Activity 2: Training and Base Metrics":
    st.title("Activity 2: Training and Accuracy")
    
    with st.expander("Activity Guide: How to Train the Model", expanded=True):
        st.write("1. **Set Parameters:** Use the sidebar sliders to set Epochs and Batch Size.")
        st.write("2. **Train:** Click 'Execute Training' to start the DNN optimization.")
        st.write("3. **Observe:** Does total accuracy reflect true performance in this scenario?")

    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 5, 50, 50, help="More epochs allow the model to learn longer.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32], value=16, help="Smaller batches make training more granular.")

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
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
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
            
            st.write(f"**Data Summary:** The model achieved a final global training accuracy of {final_acc:.2%}.")
            
            with st.expander("Reveal: Is Total Accuracy a good metric here?"):
                st.warning("""
                **No, it is highly misleading.** Because the dataset is imbalanced (the vast majority of patients survive), 
                a model could simply guess "Survival" for every single patient and still achieve high total accuracy. 
                In clinical settings, we must look at the F1 Score, Sensitivity, and Specificity instead.
                """)

# ==========================================
# ACTIVITY 3: EVALUATION TRADE-OFFS
# ==========================================
elif activity == "Activity 3: Evaluation Trade-offs":
    st.title("Activity 3: Advanced Clinical Metrics")
    
    with st.expander("Activity Guide: How to Evaluate the Model", expanded=True):
        st.write("1. **Generate Predictions:** Click 'Run 5-Fold Evaluation'.")
        st.write("2. **Adjust Threshold:** Move the slider to shift the balance between Sensitivity and Specificity.")

    if st.button("Run 5-Fold Evaluation", help="Execute 5-fold cross-validation to rigorously test the model."):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
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
        
        with st.expander("Reveal: Understanding the Trade-Off"):
            st.info("""
            **The ROC Curve Connection:** Moving the slider above is essentially manually traveling along an ROC (Receiver Operating Characteristic) curve. 
            If you lower the threshold to catch every single potential mortality case (High Sensitivity), you will inherently increase 
            the number of false alarms (Lower Specificity). Hospitals must decide where their 'sweet spot' is on this curve.
            """)

# ==========================================
# ACTIVITY 4: STRATEGIC COMPARISON
# ==========================================
elif activity == "Activity 4: Strategic Comparison":
    st.title("Activity 4: Model Strategy")
    
    with st.expander("Activity Guide: Final Assessment", expanded=True):
        st.write("Compare the DNN to the Decision Tree and determine which to deploy based on your priorities.")

    st.subheader("Decision Matrix")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decision Tree (MS1)**")
        st.write("- **Logic:** Interpretable 'If-Then' branches.")
        st.write("- **Transparency:** High (White Box).")
    with col2:
        st.markdown("**Deep Neural Network (DNN)**")
        st.write("- **Logic:** Complex non-linear combinations across layers.")
        st.write("- **Transparency:** Low (Black Box).")
        
    st.markdown("---")
    
    priority = st.select_slider("Select Core Requirement:", options=["Interpretability", "Balanced", "Performance"], help="Slide to reveal the recommended algorithm based on organizational goals.")
    
    if priority == "Interpretability":
        st.info("Strategy: Use the Decision Tree. Clinician trust relies on understanding the exact 'If-Then' logic.")
    elif priority == "Performance":
        st.success("Strategy: Use the DNN. Raw detection power is the highest priority for patient safety.")
    else:
        st.warning("Strategy: Hybrid approach required.")

    with st.expander("Reveal: Why does the DNN perform better?"):
        st.success("""
        **The Power of Hidden Layers:** A Decision Tree makes rigid, rectangular cuts in the data (e.g., "If Age > 60 and BP < 90"). 
        A Deep Neural Network evaluates the non-linear relationship *between* the variables. It can understand that a slightly low blood 
        pressure might be perfectly fine for an average patient, but extremely dangerous if combined with a specific BMI and Glucose level. 
        It trades human readability (the "Black Box") for immense mathematical predictive power.
        """)
