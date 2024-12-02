import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

def main():
    st.title("Model Evaluation")
    st.write("Upload your predictions and ground truth to evaluate the model.")

    # Initialize session state variables
    if "uploaded_data" not in st.session_state:
        st.session_state["uploaded_data"] = None
    if "accuracy" not in st.session_state:
        st.session_state["accuracy"] = None
    if "conf_matrix" not in st.session_state:
        st.session_state["conf_matrix"] = None

    # File upload for predictions and ground truth
    uploaded_file = st.file_uploader(
        "Upload a CSV file with 'y_true' and 'y_pred' columns", 
        type=["csv"]
    )

    if uploaded_file:
        try:
            # Read CSV file
            data = pd.read_csv(uploaded_file)

            # Check if required columns are present
            if "y_true" in data.columns and "y_pred" in data.columns:
                y_true = data["y_true"]
                y_pred = data["y_pred"]

                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                conf_matrix = confusion_matrix(y_true, y_pred)

                # Save results to session state
                st.session_state["uploaded_data"] = data
                st.session_state["accuracy"] = accuracy
                st.session_state["conf_matrix"] = conf_matrix

                st.success("Model evaluation results have been saved!")
            else:
                st.error("The CSV file must contain 'y_true' and 'y_pred' columns.")
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

    # Display saved results from session state if they exist
    if st.session_state["uploaded_data"] is not None:
        st.subheader("Evaluation Results")
        st.write(f"**Accuracy:** {st.session_state['accuracy']:.2f}")

        st.subheader("Confusion Matrix")
        st.write(st.session_state["conf_matrix"])
    else:
        st.info("No data has been uploaded yet.")
