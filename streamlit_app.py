import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

st.set_page_config(page_title="Applied ML Demo", layout="wide")

# Navigation
activity = st.sidebar.radio("Navigation", [
    "Activity 1: Scenario",
    "Activity 2 & 3: Interactive CNN",
    "Activity 4: Comparison"
])

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
    **Instructions:** 1. Adjust the **Hyperparameters** in the sidebar to configure your CNN.
    2. Click **'Train Next Fold'** to execute one of the 5 cross-validation folds.
    3. Observe how the **Sensitivity** and **Specificity** change with each fold.
    """)

    # Interactive Sidebar for Hyperparameters with Tooltips
    st.sidebar.header("Model Hyperparameters")
    filt1 = st.sidebar.slider("Conv1D Layer 1 Filters", 16, 64, 32, 
                              help="Number of patterns the first layer tries to learn from the clinical data.")
    filt2 = st.sidebar.slider("Conv1D Layer 2 Filters", 32, 128, 64, 
                              help="Number of patterns the second layer learns from the previous layer's output.")
    drop_val = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.3, 
                                 help="Randomly shuts off neurons to prevent the model from memorizing (overfitting) the training data.")
    lr = st.sidebar.select_slider("Learning Rate", options=[0.01, 0.001, 0.0001], value=0.001, 
                                  help="Controls how much the model's weights change during training. Smaller values are more stable.")

    if 'cv_results' not in st.session_state:
        st.session_state.cv_results = []
    if 'current_fold' not in st.session_state:
        st.session_state.current_fold = 0

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))
        
        btn_col1, btn_col2 = st.columns(2)
        if btn_col1.button("Train Next Fold", help="Starts training for the next segment of the 5-fold cross-validation."):
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

        if btn_col2.button("Reset Session", help="Clears all results and starts over."):
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
            
            with st.expander("What do these mean?"):
                st.write("**Sensitivity (Recall):** Ability to find all patients who will die (avoiding missed cases).")
                st.write("**Specificity:** Ability to correctly identify patients who will survive (avoiding false alarms).")
        else:
            st.info("No data yet. Click 'Train Next Fold' to begin.")

# Activity 4
elif activity == "Activity 4: Comparison":
    st.title("Activity 4: Model Comparison")
    st.info("**Instructions:** Review the trade-offs between the model types to determine which is best for a clinical setting.")
    
    st.subheader("CNN vs. Decision Tree")
    st.write("""
    In previous modules, we used Decision Trees. Compared to CNNs:
    - **Decision Trees:** Easier to explain to doctors (e.g., 'If glucose > 200, then...').
    - **CNNs:** Often more accurate on large clinical databases but harder to explain ('Black Box').
    """)
    
    st.warning("In a hospital, a model with slightly lower accuracy but higher interpretability (like a Decision Tree) is often preferred for safety.")