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
def display_performance_monitor():
    """Tracks CPU and RAM usage of the current Streamlit process."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Monitor")
    st.sidebar.caption("Tracks the resource usage of this app in real-time.")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Load", f"{cpu_percent}%", help="Current CPU usage of the Streamlit server.")
    c2.metric("RAM Usage", f"{mem_mb:.1f} MB", help="Current RAM memory allocated to this app.")

# ---------------------------------
# Page Config & Sidebar
# ---------------------------------
st.set_page_config(page_title="Applied Fundamentals of Deep Learning", layout="wide")

st.sidebar.markdown("### 1. Select Perspective")
perspective = st.sidebar.radio(
    "View demonstration through the lens of:",
    ["Clinical Science", "Foundational Science"],
    help="Toggle this to see how the same machine learning pipeline is interpreted differently depending on the scientific domain."
)

st.sidebar.markdown("### 2. Navigation")
activity = st.sidebar.radio(
    "Go to:",
    [
        "Activity 1 - Data Exploration",
        "Activity 2 - Model Optimization",
        "Activity 3 - Cross-Validation Analysis",
        "Activity 4 - Strategic Evaluation"
    ],
    help="Select an activity to interact with the corresponding stage of the pipeline."
)

display_performance_monitor()

# ---------------------------------
# Context Variables
# ---------------------------------
if perspective == "Clinical Science":
    app_desc = "Interactive demonstration of a clinical analytics pipeline. Observe how a Deep Neural Network learns to predict in-hospital mortality using data from the eICU Collaborative Research Database."
else:
    app_desc = "Interactive demonstration of a computational biology pipeline. Analyze how a Deep Neural Network maps continuous input features to a binary target on a highly imbalanced dataset."

st.title("Applied Fundamentals of Deep Learning")
st.write(app_desc)

# ---------------------------------
# Load Dataset 
# ---------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/diabetes.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("diabetes.csv")
        except FileNotFoundError:
            # Fallback dataset generation if file is missing
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

# --------------------
# Activity 1 - Data Exploration
# --------------------
if activity == "Activity 1 - Data Exploration":
    st.header("Activity 1: Exploring Data Types")
    
    st.markdown("### Instructions")
    st.write("Complete each activity in order. In the sidebar, toggle between the Clinical Science and Foundational Science perspectives. Record your responses to the module activities exclusively in your Canvas submission area.")
    st.write("Before training a model, researchers must inspect the raw data to understand feature distributions and identify class imbalances.")
    
    st.subheader("Data Preview")
    n_rows = st.slider(
        "Number of records to display", 
        min_value=1, max_value=20, value=5,
        help="Drag the slider to increase or decrease the number of rows visible in the table below."
    )
    st.dataframe(df.head(n_rows), use_container_width=True)
    
    with st.expander("View Data Types (.dtypes)"):
        st.write("These are the variable types the computer recognizes for each column:")
        st.write(df.dtypes)
    
    st.subheader("Feature Distributions")
    feature_cols = [col for col in df.columns if col != 'Outcome']
    feature_to_plot = st.selectbox(
        "Select a feature to visualize:", 
        feature_cols,
        help="Choose a specific metric to see a chart of its distribution."
    )
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.markdown("**Outcome Distribution**")
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival (0)', 1: 'Death (1)'})
        st.bar_chart(class_counts, color="#1f77b4")
        st.write(f"**Data Summary:** There are {class_counts.iloc[0]} Survival records and {class_counts.iloc[1]} Death records.")
        
    with col2:
        st.markdown(f"**Mean {feature_to_plot} by Outcome**")
        feature_means = df.groupby('Outcome')[feature_to_view if 'feature_to_view' in locals() else feature_to_plot].mean()
        st.bar_chart(feature_means, color="#ff7f0e")
        st.write(f"**Data Summary:** The mean {feature_to_plot} for Survivors is {feature_means.iloc[0]:.2f}, while for Deaths it is {feature_means.iloc[1]:.2f}.")

    st.markdown("---")
    with st.expander("Reveal Expected Insights for Data Exploration"):
        if perspective == "Clinical Science":
            st.info("""
            **The Job Task:** The objective is to predict in-hospital mortality using demographic and lab data to support ICU triage.
            **The Algorithmic Advantage:** A Deep Neural Network considers not only the individual clinical features but also the complex, non-linear relationships among them.
            **Understanding the Data Format:** Prior to ingestion, features are standardized (scaled) so that features with large numerical values do not mathematically overpower those with small values.
            """)
        else:
            st.info("""
            **The Job Task:** The task is binary classification, mapping continuous input arrays to a 0 or 1 target variable on a highly imbalanced dataset.
            **The Algorithmic Advantage:** The Deep Neural Network provides automated, non-linear feature extraction across multiple dense layers, eliminating manual feature engineering.
            **Understanding the Data Format:** Features are standardized to a mean of 0 and a variance of 1 to ensure stable gradient updates during backpropagation.
            """)

# --------------------
# Activity 2 - Model Optimization
# --------------------
elif activity == "Activity 2 - Model Optimization":
    st.header("Activity 2: Model Optimization")
    
    st.markdown("### Instructions")
    st.write("Configure the optimization parameters to dictate how the network updates its internal weights. Execute the training pipeline and evaluate the resulting learning curve.")
    
    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 5, 50, 50, help="Number of complete passes through the training dataset.")
    batch_size = st.sidebar.select_slider("Batch Size", options=[8, 16, 32], value=16, help="Samples processed before the model updates its weights.")

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
        
        if st.button("Execute Training"):
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
            st.metric("Final Global Accuracy", f"{final_acc:.4f}")
            st.write(f"**Data Summary:** The model converged with a final global training accuracy of {final_acc:.2%}.")

    st.markdown("---")
    with st.expander("Reveal Expected Insights for Model Optimization"):
        if perspective == "Clinical Science":
            st.warning("""
            **Comparison to MS1:** While the DNN may reach higher accuracy than the Decision Tree, global accuracy is deceptive here.
            **Metric Suitability:** Total accuracy is not a valid evaluation metric for this specific case because the vast majority of patients survive.
            """)
        else:
            st.warning("""
            **Comparison to MS1:** The DNN has higher capacity, but we must evaluate if it is optimizing for the minority class or defaulting to the majority.
            **Metric Suitability:** In imbalanced datasets, total accuracy is insufficient as the loss function is dominated by the majority class.
            """)

# --------------------
# Activity 3 - Cross-Validation Analysis
# --------------------
elif activity == "Activity 3 - Cross-Validation Analysis":
    st.header("Activity 3: Cross-Validation and Trade-Offs")
    
    st.markdown("### Instructions")
    st.write("Execute the 5-fold cross-validation analysis. Adjust the classification threshold to observe the statistical trade-offs between Sensitivity and Specificity.")

    if st.button("Run 5-Fold Evaluation"):
        X = df.drop(columns=['Outcome']).values
        y = df['Outcome'].values
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        results = []
        progress_bar = st.progress(0)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
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
        
        st.session_state['act3_results'] = results
        st.success("Evaluation Metrics Generated")

    if 'act3_results' in st.session_state:
        threshold = st.slider("Classification Threshold", 0.1, 0.9, 0.5)
        
        metrics = []
        for y_true, y_prob in st.session_state['act3_results']:
            y_pred = (y_prob > threshold).astype(int).flatten()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            acc = (tp + tn) / (tp + tn + fp + fn)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics.append([acc, sens, spec, prec])
        
        avg_m = np.mean(metrics, axis=0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Accuracy", f"{avg_m[0]:.3f}")
        c2.metric("Avg Sensitivity", f"{avg_m[1]:.3f}")
        c3.metric("Avg Specificity", f"{avg_m[2]:.3f}")
        c4.metric("Avg Precision", f"{avg_m[3]:.3f}")

    st.markdown("---")
    with st.expander("Reveal Expected Insights for Cross-Validation Analysis"):
        st.info("""
        **Performance Evaluation:** Lowering the threshold increases Sensitivity (catching more mortality cases) but decreases Specificity (generating more false alarms). This effectively simulates traversing an ROC curve.
        """)

# --------------------
# Activity 4 - Strategic Evaluation
# --------------------
elif activity == "Activity 4 - Strategic Evaluation":
    st.header("Activity 4: Strategic Evaluation")
    
    st.markdown("### Instructions")
    st.write("Determine the optimal algorithmic approach based on organizational requirements for interpretability versus performance.")

    st.subheader("Architectural Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Decision Tree (MS1)**")
        st.write("- Logic: Interpretable 'If-Then' splits.")
        st.write("- Transparency: High (White Box).")
    with col2:
        st.markdown("**Deep Neural Network (Current)**")
        st.write("- Logic: Complex non-linear combinations.")
        st.write("- Transparency: Low (Black Box).")
        
    st.markdown("---")
    priority = st.select_slider("Select Requirement:", options=["Interpretability", "Balanced", "Performance"])
    
    if priority == "Interpretability":
        st.info("Strategy: Utilize the Decision Tree for human-readable logic.")
    elif priority == "Performance":
        st.success("Strategy: Utilize the DNN for maximum predictive power.")
    else:
        st.warning("Strategy: Hybrid or post-hoc explainability required.")

    st.markdown("---")
    with st.expander("Reveal Expected Insights for Strategic Evaluation"):
        st.success("""
        **Model Selection:** The DNN trades human readability for mathematical capacity. One would choose the DNN for complex mapping, but default to the Decision Tree if structural transparency is an absolute requirement.
        """)
