from pathlib import Path
import hashlib
import streamlit as st
import pandas as pd

from modules import prediction_app, query_app, visualisation_app

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

PAGES = {
    "Home": "🏠 Home",
    "Query": "💬 Ask Dataset",
    "Visualization": "📈 Visualization",
    "Prediction": "🎯 Prediction",
}


def fingerprint(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reset_dataset_state():
    # Module-specific outputs belong to the previous dataset and must not leak forward.
    for key in ("query_messages", "viz_plan", "viz_data", "viz_dataset_id", "viz_verified",
                "prediction_target"):
        st.session_state.pop(key, None)


if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.caption("Your dataset workspace")
    st.divider()
    st.subheader("Navigation")
    for page, label in PAGES.items():
        if st.button(label, use_container_width=True, key=f"nav_{page}",
                     type="primary" if st.session_state.active_page == page else "secondary"):
            st.session_state.active_page = page
            st.rerun()
    st.divider()
    if st.session_state.dataset is not None:
        st.success(f"Dataset loaded: {st.session_state.dataset_name}")
        st.caption(f"{st.session_state.dataset.shape[0]:,} rows × {st.session_state.dataset.shape[1]} columns")
    else:
        st.info("No dataset uploaded yet.")

page = st.session_state.active_page

if page == "Home":
    st.title("📊 AI Data Analyst")
    st.write("Upload one dataset and use all three analytical modules from a single workspace.")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"], key="home_dataset_uploader")
    if uploaded_file is not None:
        raw = uploaded_file.getvalue()
        new_id = fingerprint(raw)
        if new_id != st.session_state.dataset_id:
            try:
                loaded_df = pd.read_csv(uploaded_file)
                if loaded_df.empty:
                    st.error("The uploaded CSV is empty. Please choose a dataset containing rows.")
                elif loaded_df.columns.duplicated().any():
                    st.error("Duplicate column names are not supported. Please correct the CSV and upload again.")
                else:
                    st.session_state.dataset = loaded_df
                    st.session_state.dataset_id = new_id
                    st.session_state.dataset_name = uploaded_file.name
                    reset_dataset_state()
                    st.success("New dataset uploaded. Module-specific history has been reset.")
            except Exception as exc:
                st.error(f"Unable to read this CSV: {exc}")
        elif st.session_state.dataset is None:
            try:
                st.session_state.dataset = pd.read_csv(uploaded_file)
                st.session_state.dataset_name = uploaded_file.name
            except Exception as exc:
                st.error(f"Unable to read this CSV: {exc}")

    if st.session_state.dataset is not None:
        df = st.session_state.dataset
        st.divider()
        st.subheader("Dataset overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
        st.dataframe(df.head(10), use_container_width=True)
        st.divider()
        st.subheader("Choose a module")
        cols = st.columns(3)
        for col, target, label, desc in zip(
            cols,
            ("Query", "Visualization", "Prediction"),
            ("💬 Ask Dataset", "📈 Visualization", "🎯 Prediction"),
            ("Ask questions in natural language.", "Generate charts from analytical questions.", "Train a model and make predictions.")
        ):
            with col:
                st.markdown(f"### {label}")
                st.write(desc)
                if st.button("Open module", key=f"home_open_{target}", use_container_width=True):
                    st.session_state.active_page = target
                    st.rerun()
    else:
        st.info("Upload a CSV file above. Dataset upload is managed only from this Home page.")

else:
    df = st.session_state.dataset
    if df is None:
        st.warning("Please upload a dataset from the Home page first.")
        if st.button("Go to Home", type="primary"):
            st.session_state.active_page = "Home"
            st.rerun()
    elif page == "Query":
        query_app.render(df)
    elif page == "Visualization":
        visualisation_app.render(df)
    elif page == "Prediction":
        prediction_app.render(df)
