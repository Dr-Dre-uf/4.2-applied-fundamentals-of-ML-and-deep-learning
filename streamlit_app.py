# streamlit_app_interactive_metrics.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Applied ML & Deep Learning", layout="wide")

# ----------------------------
# Sidebar navigation
# ----------------------------
st.sidebar.title("Steps")
activity = st.sidebar.radio("Navigate to:", [
    "Load & Explore Data",
    "CNN Training"
])

# ----------------------------
# Reset session button
# ----------------------------
if st.sidebar.button("Reset CNN Training", help="Clear all fold progress and start training over."):
    for key in ['cnn_metrics', 'cv_results', 'current_fold']:
        if key in st.session_state:
            del st.session_state[key]
    st.sidebar.success("Session reset. You can start CNN training again.")

# ----------------------------
# Privacy alert
# ----------------------------
st.warning("⚠️ Please do NOT upload any personal or private data. Use anonymized or sample data only.")

# ----------------------------
# Sample data
# ----------------------------
def load_sample_data():
    from sklearn.datasets import load_diabetes
    diabetes = load_diabetes(as_frame=True)
    df = diabetes.frame.copy()
    # Make a binary target for demonstration (simulate Outcome)
    df['Outcome'] = (df['target'] > df['target'].median()).astype(int)
    df.drop(columns='target', inplace=True)
    return df

uploaded_file = st.file_uploader(
    "Upload CSV (optional, anonymized only)", 
    type=["csv"], 
    help="Upload anonymized CSV data. Do NOT upload personal or private data."
)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = load_sample_data()

# Store CNN metrics and fold info
if 'cnn_metrics' not in st.session_state:
    st.session_state.cnn_metrics = {}
if 'cv_results' not in st.session_state:
    st.session_state.cv_results = []
if 'current_fold' not in st.session_state:
    st.session_state.current_fold = 0
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = df.columns[:-1].tolist()
    st.session_state.target_column = df.columns[-1]

# Prepare X and y
feature_columns = st.session_state.feature_columns
target_column = st.session_state.target_column
X = df[feature_columns].values
y = df[target_column].values

# ----------------------------
# Load & Explore Data
# ----------------------------
if activity == "Load & Explore Data":
    st.title("Data Preview & Feature Selection")
    st.subheader("Dataset Preview")
    st.dataframe(df.head())
    st.subheader("Basic Statistics")
    st.dataframe(df.describe())
    st.subheader("Select Features and Target")
    all_columns = df.columns.tolist()
    st.session_state.target_column = st.selectbox(
        "Target column:", 
        all_columns, 
        index=all_columns.index(st.session_state.target_column),
        help="Choose the column you want the CNN to predict."
    )
    st.session_state.feature_columns = st.multiselect(
        "Feature columns:", 
        [c for c in all_columns if c != st.session_state.target_column], 
        default=st.session_state.feature_columns,
        help="Select the columns to be used as input features for the CNN."
    )
    st.write(f"Features shape: {len(st.session_state.feature_columns)}, Target: {st.session_state.target_column}")

# ----------------------------
# CNN Training
# ----------------------------
elif activity == "CNN Training":
    st.title("CNN Training")

    st.subheader("CNN Parameters")
    n_splits = st.slider("Number of CV folds:", 2, 10, 5, help="Number of folds for cross-validation.")
    epochs = st.slider("Epochs:", 10, 100, 50, help="Number of training epochs for each fold.")
    batch_size = st.slider("Batch size:", 8, 64, 16, help="Number of samples per gradient update.")
    filters1 = st.number_input("Filters for Conv1D layer 1:", 8, 128, 32, help="Number of convolutional filters in the first Conv1D layer.")
    filters2 = st.number_input("Filters for Conv1D layer 2:", 8, 128, 64, help="Number of convolutional filters in the second Conv1D layer.")
    dropout_rate = st.slider("Dropout rate:", 0.0, 0.5, 0.3, help="Dropout rate for the fully connected layer to prevent overfitting.")

    start_training = st.button("Start or Continue Training", help="Click to train the next fold of the CNN. Progress is saved across folds.")
    if start_training:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        folds = list(kf.split(X))

        if st.session_state.current_fold < n_splits:
            train_index, val_index = folds[st.session_state.current_fold]
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

            model = Sequential([
                Input(shape=(X_train.shape[1],1)),
                Conv1D(filters1, kernel_size=3, activation='relu'),
                Conv1D(filters2, kernel_size=3, activation='relu'),
                Flatten(),
                Dense(32, activation='relu'),
                Dropout(dropout_rate),
                Dense(1, activation='sigmoid')
            ])
            model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_val, y_val))

            # Plot accuracy
            fig, ax = plt.subplots()
            ax.plot(history.history['accuracy'], label='Train Acc')
            ax.plot(history.history['val_accuracy'], label='Val Acc')
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Fold {st.session_state.current_fold + 1} Accuracy")
            ax.legend()
            st.pyplot(fig)

            # Confusion matrix
            y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).flatten()
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            cm = np.array([[tn, fp], [fn, tp]])
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            ax_cm.set_title(f"Fold {st.session_state.current_fold + 1} Confusion Matrix")
            st.pyplot(fig_cm)

            acc = (tp+tn)/(tp+tn+fp+fn)
            sens = tp/(tp+fn)
            spec = tn/(tn+fp)
            prec = tp/(tp+fp)
            st.session_state.cv_results.append({'Fold': st.session_state.current_fold + 1, 'Accuracy': acc, 'Sensitivity': sens, 'Specificity': spec, 'Precision': prec})

            # Display fold-by-fold metrics
            st.subheader("Fold Metrics")
            df_metrics = pd.DataFrame(st.session_state.cv_results)
            st.dataframe(df_metrics.style.format({
                'Accuracy': "{:.3f}",
                'Sensitivity': "{:.3f}",
                'Specificity': "{:.3f}",
                'Precision': "{:.3f}"
            }))

            st.session_state.current_fold += 1
            if st.session_state.current_fold < n_splits:
                st.info(f"✅ Fold {st.session_state.current_fold} completed. Press 'Start or Continue Training' to train next fold.")
            else:
                st.success("✅ All folds trained. Average metrics saved.")
                results = pd.DataFrame(st.session_state.cv_results)
                st.session_state.cnn_metrics = results.drop(columns='Fold').mean().to_dict()
                st.write("Average Metrics Across Folds:", {k: f"{v:.3f}" for k, v in st.session_state.cnn_metrics.items()})
        else:
            st.warning("All folds have been trained. Reset session to train again.")
