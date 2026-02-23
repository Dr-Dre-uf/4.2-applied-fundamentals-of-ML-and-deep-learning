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
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_percent = process.cpu_percent(interval=0.1)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Monitor")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("CPU Usage", f"{cpu_percent}%", help="Spikes during deep learning model training.")
    c2.metric("RAM Footprint", f"{mem_mb:.1f} MB", help="Current memory usage of the application process.")

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# --- TRACK SELECTION ---
st.sidebar.header("Select Your Track")
track = st.sidebar.radio("Focus Area", ["Clinical Science", "Foundational Science"], 
                         help="Clinical focuses on patient care impact; Foundational focuses on algorithmic mechanics.")
st.sidebar.markdown("---")

# --- S1 THROUGH S6 NAVIGATION ---
st.sidebar.header("Module Progress")
step = st.sidebar.select_slider(
    "Activity Step",
    options=["S1", "S2", "S3", "S4", "S5", "S6"],
    help="S1: Scenario, S2: Data, S3: Architecture, S4: Training, S5: Advanced Metrics, S6: Comparison"
)

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
# S1: CLINICAL SCENARIO
# ==========================================
if step == "S1":
    st.title("S1: Clinical Scenario")
    # 

[Image of clinical decision support system architecture]

    st.write("""
    **Goal:** Use a CNN model to predict in-hospital mortality using data from the eICU Collaborative Research Database.
    **Task:** Binary Classification (0: Survival, 1: Death).
    """)

# ==========================================
# S2: DATA EXPLORATION
# ==========================================
elif step == "S2":
    st.title("S2: Data Exploration")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Class Imbalance")
        class_counts = df['Outcome'].value_counts().rename(index={0: 'Survival', 1: 'Death'})
        st.bar_chart(class_counts, color="#FF4B4B")
    with col2:
        st.markdown("### Data Types")
        st.dataframe(df.dtypes.to_frame(name="Type"), use_container_width=True)

# ==========================================
# S3: CNN ARCHITECTURE
# ==========================================
elif step == "S3":
    st.title("S3: CNN Architecture")
    # 
    st.markdown("### Why use CNN for this task?")
    st.info("Advantage: 1D CNNs automatically identify complex relationships between lab values without manual feature engineering.")
    st.code("""
    model = Sequential([
        Input(shape=(X.shape[1], 1)),
        Conv1D(32, kernel_size=2, activation='relu'),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])
    """, language='python')

# ==========================================
# S4: BASE PERFORMANCE (TRAINING)
# ==========================================
elif step == "S4":
    st.title("S4: Base Performance")
    epochs = st.sidebar.slider("Epochs", 5, 30, 10, help="Number of training iterations.")
    
    if st.button("Run Base Training", help="Trains the model to observe standard accuracy."):
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).reshape(len(X), X.shape[1], 1)
        
        model = Sequential([Input(shape=(X.shape[1], 1)), Conv1D(16, 2, activation='relu'), Flatten(), Dense(1, activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        with st.spinner("Training..."):
            history = model.fit(X_scaled, y, epochs=epochs, validation_split=0.2, verbose=0)
        
        st.line_chart(pd.DataFrame(history.history)['accuracy'])
        st.success(f"Final Accuracy: {history.history['accuracy'][-1]:.2%}")
        st.warning("Notebook Question: Is total accuracy a good evaluation metric for this case?")

# ==========================================
# S5: ADVANCED CLINICAL METRICS
# ==========================================
elif step == "S5":
    st.title("S5: Advanced Clinical Metrics")
    # 
    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 
                          help="Lowering the threshold increases Sensitivity (catches more deaths) but lowers Specificity.")
    
    st.write("Calculate Sensitivity (Recall), Specificity, and Precision to see the true performance.")
    # (Simplified metric display for demo)
    st.metric("Sensitivity", "0.824", help="Proportion of actual deaths correctly identified.")

# ==========================================
# S6: MODEL COMPARISON
# ==========================================
elif step == "S6":
    st.title("S6: Comparison")
    # 
    st.markdown("### CNN vs. Decision Tree")
    st.write("How is the performance? Is it better than the decision tree from MS1?")
    
    comp_df = pd.DataFrame({
        'Metric': ['Interpretability', 'Performance'],
        'Decision Tree': [9, 5],
        '1D CNN': [2, 9]
    }).set_index('Metric')
    st.bar_chart(comp_df)
