import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import psutil
import os
from sklearn.model_selection import KFold, train_test_split
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
    outcome_label = "In-Hospital Mortality (0=Survival, 1=Death)"
else:
    app_desc = "Interactive demonstration of a computational biology pipeline. Analyze how a Deep Neural Network maps continuous input features to a binary target on a highly imbalanced dataset."
    outcome_label = "Binary Target (0=Majority Class, 1=Minority Class)"

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
    st.write("Before training a model, researchers must inspect the raw data to understand feature distributions and identify class imbalances. Use the controls below to preview the dataset and inspect the data structures.")
    
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
        help="Choose a specific metric to see a histogram of its distribution."
    )
    
    # ADA COMPLIANCE: Colorblind safe colors
    fig = px.histogram(
        df, x=feature_to_plot, color="Outcome", barmode="overlay",
        title=f"Distribution of {feature_to_plot} grouped by {outcome_label}",
        color_discrete_sequence=["#1f77b4", "#ff7f0e"] 
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View chart data as text (Accessible Alternative)"):
        st.dataframe(df.groupby("Outcome")[feature_to_plot].describe())

    st.markdown("---")
    with st.expander("Reveal Expected Insights for Data Exploration"):
        if perspective == "Clinical Science":
            st.write("The primary job task is to predict in-hospital mortality using demographic and lab data to support ICU triage.")
            st.write("The advantage of using a Deep Neural Network is its ability to consider complex, non-linear relationships among clinical features. A specific blood pressure measurement might only be flagged as dangerous when combined with specific BMI and Glucose levels.")
            st.write("Prior to ingestion, features must be standardized. This prevents metrics with large numerical values from overpowering metrics with small values, ensuring all clinical data is evaluated proportionally.")
        else:
            st.write("The primary job task is binary classification, mapping continuous input arrays to a discrete target variable on a highly imbalanced dataset.")
            st.write("The advantage of the Deep Neural Network is its automated, non-linear feature extraction across multiple dense layers, eliminating the need for the manual feature engineering required by baseline models.")
            st.write("Features are standardized to a mean of 0 and a variance of 1. This prevents features with large magnitudes from dominating gradient updates during backpropagation, ensuring stable convergence.")

# --------------------
# Activity 2 - Model Optimization
# --------------------
elif activity == "Activity 2 - Model Optimization":
    st.header("Activity 2: Model Optimization")
    
    st.markdown("### Instructions")
    st.write("Configure the optimization parameters to dictate how the network updates its internal weights. Execute the training pipeline and evaluate the resulting learning curve and global accuracy.")
    
    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider(
        "Epochs", 
        min_value=5, max_value=50, value=50, 
        help="Determines the number of complete passes through the training dataset."
    )
    batch_size = st.sidebar.select_slider(
        "Batch Size", 
        options=[8, 16, 32], value=16, 
        help="Determines the number of samples processed before the model updates its internal weights."
    )

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
            
            hist_df = pd.DataFrame(st.session_state['act2_history'])
            hist_df['Epoch'] = hist_df.index + 1
            
            fig = px.line(
                hist_df, x='Epoch', y='accuracy', 
                title="Training Accuracy per Epoch",
                color_discrete_sequence=["#1f77b4"]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            final_acc = hist_df['accuracy'].iloc[-1]
            st.metric("Final Global Accuracy", f"{final_acc:.4f}", help="The overall proportion of correct predictions across all classes.")
            
            with st.expander("View learning curve data as text (Accessible Alternative)"):
                st.dataframe(hist_df[['Epoch', 'accuracy']])

    st.markdown("---")
    with st.expander("Reveal Expected Insights for Model Optimization"):
        if perspective == "Clinical Science":
            st.write("While the DNN often reaches a higher raw accuracy score than the Milestone 1 Decision Tree due to its ability to find hidden patterns, global accuracy is highly deceptive in this clinical context.")
            st.write("Total accuracy is an invalid evaluation metric for this case. Because the vast majority of patients survive, a model could predict 'Survival' for every patient and achieve high accuracy without successfully detecting a single mortality case. Robust clinical evaluation requires analyzing Sensitivity, Specificity, and Precision.")
        else:
            st.write("The DNN has a higher capacity for feature extraction than the MS1 Decision Tree, but it is necessary to evaluate if the network is actively optimizing for the minority class or merely defaulting to the majority class distribution.")
            st.write("Total accuracy is an insufficient metric. In highly imbalanced datasets, the binary cross-entropy loss function is heavily dominated by the majority class, masking the model's true predictive capability and convergence on the minority class.")

# --------------------
# Activity 3 - Cross-Validation Analysis
# --------------------
elif activity == "Activity 3 - Cross-Validation Analysis":
    st.header("Activity 3: Cross-Validation and Trade-Offs")
    
    st.markdown("### Instructions")
    st.write("Execute the 5-fold cross-validation analysis to generate robust evaluation metrics. Adjust the classification threshold to observe the statistical trade-offs between Sensitivity (Recall), Specificity, and Precision.")

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
        for fold_idx, (y_true, y_prob) in enumerate(st.session_state['act3_results']):
            y_pred = (y_prob > threshold).astype(int).flatten()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            metrics.append([fold_idx + 1, acc, sens, spec, prec])
        
        metrics_df = pd.DataFrame(metrics, columns=['Fold', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision'])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Accuracy", f"{metrics_df['Accuracy'].mean():.4f}", help="Calculated as: (TP + TN) / Total Predictions")
        c2.metric("Avg Sensitivity", f"{metrics_df['Sensitivity'].mean():.4f}", help="Calculated as: TP / (TP + FN)")
        c3.metric("Avg Specificity", f"{metrics_df['Specificity'].mean():.4f}", help="Calculated as: TN / (TN + FP)")
        c4.metric("Avg Precision", f"{metrics_df['Precision'].mean():.4f}", help="Calculated as: TP / (TP + FP)")
        
        st.subheader("Fold Performance Overview")
        
        fig = px.bar(
            metrics_df, x="Fold", y="Accuracy", title="Accuracy per Fold", 
            text_auto=".3f", range_y=[0, 1], color_discrete_sequence=["#1f77b4"]
        )
        fig.add_hline(y=metrics_df['Accuracy'].mean(), line_dash="dash", line_color="#ff7f0e", annotation_text="Mean")
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View fold metrics as text (Accessible Alternative)"):
            st.dataframe(metrics_df)

    st.markdown("---")
    with st.expander("Reveal Expected Insights for Cross-Validation Analysis"):
        if perspective == "Clinical Science":
            st.write("Adjusting the threshold reveals a critical clinical trade-off. Lowering the threshold effectively increases Average Sensitivity (ensuring more potential mortality cases are flagged) at the direct expense of Average Specificity (generating more false alarms).")
            st.write("In applied hospital environments, analytical teams must identify the optimal operational balance that maximizes patient safety without creating alert fatigue among the nursing staff.")
        else:
            st.write("Shifting the decision boundary practically simulates traversing a Receiver Operating Characteristic (ROC) curve. This illustrates the model's behavior in the Precision-Recall space.")
            st.write("This interactive adjustment clearly demonstrates how effectively the network minimizes False Negatives versus False Positives during minority class optimization.")

# --------------------
# Activity 4 - Strategic Evaluation
# --------------------
elif activity == "Activity 4 - Strategic Evaluation":
    st.header("Activity 4: Strategic Evaluation")
    
    st.markdown("### Instructions")
    st.write("Contrast the architectural capabilities of the Deep Neural Network with the Milestone 1 Decision Tree. Determine the optimal algorithmic approach based on organizational requirements for interpretability versus predictive performance.")

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
    
    priority = st.select_slider(
        "Select Core Organizational Requirement:", 
        options=["Interpretability", "Balanced", "Performance"], 
        help="Select the primary requirement to reveal the recommended deployment strategy."
    )
    
    if priority == "Interpretability":
        st.info("Deployment Strategy: Utilize the Decision Tree. Clinical adoption often mandates high interpretability to ensure practitioners can validate the underlying logic of a prediction.")
    elif priority == "Performance":
        st.success("Deployment Strategy: Utilize the Deep Neural Network. Maximizing raw detection capability is prioritized to ensure high-risk patients are successfully triaged.")
    else:
        st.warning("Deployment Strategy: An ensemble approach or the application of post-hoc explainability frameworks is required to balance predictive power and transparency.")

    st.markdown("---")
    with st.expander("Reveal Expected Insights for Strategic Evaluation"):
        if perspective == "Clinical Science":
            st.write("You would likely deploy the DNN to maximize the detection of at-risk patients and allocate life-saving resources effectively. However, if hospital administrators or doctors refuse to use a 'Black Box' system because they cannot interpret the reasoning behind a prediction, the Decision Tree must be used to maintain clinician trust.")
        else:
            st.write("The DNN provides superior capacity for non-linear feature extraction compared to the orthogonal decision boundaries of the Decision Tree. You would choose the DNN for complex, high-dimensional mapping tasks, but default to the Decision Tree if structural transparency and model explainability are absolute organizational requirements.")
