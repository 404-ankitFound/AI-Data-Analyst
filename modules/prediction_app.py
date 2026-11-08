import hashlib
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, root_mean_squared_error


MODEL_DIR = Path("saved_prediction_models")
MODEL_DIR.mkdir(exist_ok=True)


def dataset_fingerprint(data: pd.DataFrame) -> str:
    """Create a fingerprint so a model is reused only with the same uploaded dataset."""
    digest = hashlib.sha256()
    digest.update("|".join(map(str, data.columns)).encode("utf-8"))
    digest.update("|".join(map(str, data.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(data, index=True).values.tobytes())
    return digest.hexdigest()[:20]


def find_predictable_columns(data: pd.DataFrame) -> list:
    """
    Follows the target-column selection logic from the notebook:
    float columns are candidates
    object columns are candidates when unique_values / rows < 0.2
    other columns are candidates when unique_values < rows
    """
    rows = len(data)
    if rows == 0:
        return []

    candidates = []
    for col in data.columns:
        series = data[col]
        unique_count = series.nunique(dropna=True)

        if pd.api.types.is_float_dtype(series):
            candidates.append(col)
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            if unique_count / rows < 0.2:
                candidates.append(col)
        else:
            if unique_count < rows:
                candidates.append(col)

    return candidates



def verify_prediction_dataset(data: pd.DataFrame, target: str | None = None):
    """Validate dataset structure and values before training a prediction model."""
    if not isinstance(data, pd.DataFrame):
        return False, ["Input is not a pandas DataFrame."]
    if data.empty:
        return False, ["Dataset contains no rows."]
    if data.columns.duplicated().any():
        return False, ["Duplicate column names are not supported."]
    if not all(isinstance(col, str) and col.strip() for col in data.columns):
        return False, ["All column names must be non-empty strings."]
    if target is None:
        return True, ["Dataset structure verified."]
    if target not in data.columns:
        return False, ["Selected target does not exist in the dataset."]
    if data.shape[1] < 2:
        return False, ["At least one feature column and one target column are required."]

    target_series = data[target].dropna()
    if len(target_series) < 5:
        return False, ["At least 5 non-missing target values are required."]
    if target_series.nunique() < 2:
        return False, ["Target must contain at least two distinct values."]
    if pd.api.types.is_datetime64_any_dtype(target_series):
        return False, ["Datetime target columns are not supported. Convert the target to a numeric or categorical value first."]
    if not (pd.api.types.is_numeric_dtype(target_series) or
            pd.api.types.is_object_dtype(target_series) or
            pd.api.types.is_string_dtype(target_series) or
            pd.api.types.is_bool_dtype(target_series) or
            isinstance(target_series.dtype, pd.CategoricalDtype)):
        return False, ["Target must be numeric or categorical."]

    problems = []
    for col in data.columns:
        series = data[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if np.isinf(numeric.dropna()).any():
                problems.append(f"Column '{col}' contains infinite numeric values.")
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_null = series.dropna()
            if non_null.map(lambda value: isinstance(value, (dict, list, tuple, set))).any():
                problems.append(f"Column '{col}' contains nested Python objects; use scalar values only.")
    if problems:
        return False, problems
    return True, ["Dataset and selected target passed validation."]


def verify_prediction_bundle(bundle: dict, data: pd.DataFrame):
    """Validate a loaded/trained model bundle before it is used for prediction."""
    required = {"model", "target", "task", "metric_name", "metric_value",
                "feature_columns", "categorical_columns", "category_levels"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        return False, "Model bundle is missing required fields."
    if not hasattr(bundle["model"], "predict"):
        return False, "Loaded model does not expose a prediction method."
    features = bundle["feature_columns"]
    if not isinstance(features, list) or not features or len(features) != len(set(features)):
        return False, "Model feature list is invalid."
    if any(col not in data.columns for col in features):
        return False, "Model expects feature columns that are absent from the current dataset."
    if bundle["target"] not in data.columns:
        return False, "Model target does not match the current dataset."
    categorical = bundle["categorical_columns"]
    if not isinstance(categorical, list) or any(col not in features for col in categorical):
        return False, "Model categorical feature metadata is invalid."
    if bundle["task"] not in {"Classification", "Regression"}:
        return False, "Unknown prediction task in model metadata."
    return True, "Model metadata and feature schema verified."

def model_path(dataset_id: str, target: str) -> Path:
    safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target))[:80]
    target_hash = hashlib.sha256(str(target).encode("utf-8")).hexdigest()[:8]
    return MODEL_DIR / f"{dataset_id}_{safe_target}_{target_hash}.joblib"


def is_categorical(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
    )


def prepare_features(data: pd.DataFrame, target: str):
    X = data.drop(columns=[target]).copy()
    categorical_columns = []

    for col in X.columns:
        if is_categorical(X[col]):
            # LightGBM handles pandas category columns natively.
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].astype(str)
            X[col] = X[col].astype("category")
            categorical_columns.append(col)

    return X, categorical_columns


def train_model(data: pd.DataFrame, target: str):
    valid, messages = verify_prediction_dataset(data, target)
    if not valid:
        raise ValueError("Security verification failed: " + " ".join(messages))

    # Rows with a missing target cannot be used for supervised training.
    train_data = data.dropna(subset=[target]).copy()
    if len(train_data) < 5:
        raise ValueError("At least 5 rows with a non-missing target are required.")

    X, categorical_columns = prepare_features(train_data, target)
    y = train_data[target]

    if y.nunique(dropna=True) < 2:
        raise ValueError("The selected target has only one distinct value. A model cannot learn from it.")

    # Same task-selection rule as the notebook.
    is_classification = (y.nunique(dropna=True) < 10) or (
        pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if is_classification:
        model = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X_train, y_train, categorical_feature=categorical_columns)
        predictions = model.predict(X_test)
        metric_name = "Accuracy"
        metric_value = float(accuracy_score(y_test, predictions))
        task = "Classification"
    else:
        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            random_state=42,
            verbosity=-1,
        )
        model.fit(X_train, y_train, categorical_feature=categorical_columns)
        predictions = model.predict(X_test)
        metric_name = "RMSE"
        metric_value = float(root_mean_squared_error(y_test, predictions))
        task = "Regression"

    # Save category levels so form inputs have exactly the same category mapping.
    category_levels = {
        col: list(X[col].cat.categories)
        for col in categorical_columns
    }

    return {
        "model": model,
        "target": target,
        "task": task,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "feature_columns": list(X.columns),
        "categorical_columns": categorical_columns,
        "category_levels": category_levels,
        "target_dtype": str(y.dtype),
    }


def load_or_train_model(data: pd.DataFrame, dataset_id: str, target: str):
    path = model_path(dataset_id, target)
    if path.exists():
        return joblib.load(path), True

    bundle = train_model(data, target)
    joblib.dump(bundle, path)
    return bundle, False


def make_prediction_form(bundle: dict, data: pd.DataFrame):
    st.subheader("Enter values for prediction")
    st.caption("Categorical columns use dropdowns. Numeric columns use number inputs.")

    values = {}
    with st.form("prediction_form"):
        cols = st.columns(2)

        for i, col in enumerate(bundle["feature_columns"]):
            series = data[col]
            with cols[i % 2]:
                if col in bundle["categorical_columns"]:
                    options = list(bundle["category_levels"].get(col, []))
                    # Keep a missing-value choice if the training data contained nulls.
                    has_missing = series.isna().any()
                    display_options = (["(Missing)"] if has_missing else []) + options

                    if not display_options:
                        st.warning(f"No available values for {col}; using missing value.")
                        values[col] = np.nan
                    else:
                        selected = st.selectbox(
                            col,
                            options=display_options,
                            format_func=lambda value: str(value),
                            key=f"input_{col}",
                        )
                        values[col] = np.nan if selected == "(Missing)" else selected
                else:
                    numeric = pd.to_numeric(series, errors="coerce").dropna()
                    if numeric.empty:
                        default = 0.0
                    else:
                        default = float(numeric.median())

                    if pd.api.types.is_integer_dtype(series):
                        values[col] = st.number_input(
                            col, value=int(round(default)), step=1, key=f"input_{col}"
                        )
                    else:
                        values[col] = st.number_input(
                            col, value=float(default), key=f"input_{col}"
                        )

        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = pd.DataFrame([values], columns=bundle["feature_columns"])

        # Restore the exact category sets used at training time.
        for col in bundle["categorical_columns"]:
            categories = bundle["category_levels"].get(col, [])
            row[col] = pd.Categorical(row[col], categories=categories)

        # Match the model's expected feature order, then verify the final input.
        row = row[bundle["feature_columns"]]
        if list(row.columns) != bundle["feature_columns"] or row.shape != (1, len(bundle["feature_columns"])):
            st.error("Security verification failed: prediction input schema mismatch.")
            return
        if row.select_dtypes(include=["number"]).isin([np.inf, -np.inf]).any().any():
            st.error("Security verification failed: infinite numeric input detected.")
            return
        prediction = bundle["model"].predict(row)[0]

        st.success("Prediction generated")
        if bundle["task"] == "Classification":
            st.metric("Predicted class", str(prediction))
            if hasattr(bundle["model"], "predict_proba"):
                probabilities = bundle["model"].predict_proba(row)[0]
                confidence = float(np.max(probabilities)) * 100
                st.metric("Model confidence", f"{confidence:.2f}%")
        else:
            st.metric(f"Predicted {bundle['target']}", f"{float(prediction):,.4f}")

        st.info(
            f"Test-set {bundle['metric_name']}: "
            f"{bundle['metric_value']:.4f}"
            + (f" ({bundle['metric_value'] * 100:.2f}%)" if bundle["metric_name"] == "Accuracy" else "")
        )



def render(df: pd.DataFrame):
    """Render prediction module using the dataset managed by app.py."""
    st.title("📊 Prediction")
    st.caption("Choose a column to predict, train a LightGBM model, and generate predictions.")
    if df is None or df.empty:
        st.info("Upload a CSV from Home to start.")
        return

    dataset_ok, dataset_messages = verify_prediction_dataset(df)
    if not dataset_ok:
        st.error("Security verification failed for this dataset.")
        for message in dataset_messages:
            st.write(f"- {message}")
        return
    st.success("Security verification passed: dataset structure is valid.")

    st.subheader("Dataset preview")
    st.write(f"Rows: **{df.shape[0]:,}** | Columns: **{df.shape[1]}**")
    with st.expander("Preview dataset"):
        st.dataframe(df.head(10), use_container_width=True)

    dp = df.copy(deep=True)
    dataset_id = dataset_fingerprint(dp)
    candidates = find_predictable_columns(dp)
    st.divider()
    st.header("Prediction")
    if not candidates:
        st.warning("No predictable target columns were found using the notebook's feature-selection rule.")
        return

    target = st.selectbox("Select the column you want to predict", candidates,
                          help="The target-column selection follows the logic in your notebook.",
                          key="prediction_target")
    target_ok, target_messages = verify_prediction_dataset(dp, target)
    if not target_ok:
        st.error("Security verification failed for the selected target.")
        for message in target_messages:
            st.write(f"- {message}")
        return
    st.success("Security verification passed: selected target and training data are valid.")

    current_model_path = model_path(dataset_id, target)
    if current_model_path.exists():
        st.success(f"Model already trained and loaded for **{target}**.")
    else:
        st.warning(f"No trained model found for **{target}** in this dataset.")
        if st.button("Train your model", type="primary", key="train_prediction_model"):
            with st.spinner(f"Training a model to predict '{target}'..."):
                try:
                    bundle = train_model(dp, target)
                    joblib.dump(bundle, current_model_path)
                    st.success("Training completed. Model saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")
                    return

    if current_model_path.exists():
        try:
            bundle = joblib.load(current_model_path)
        except Exception as exc:
            st.error(f"Could not load the saved model: {exc}")
            return
        bundle_ok, bundle_message = verify_prediction_bundle(bundle, dp)
        if not bundle_ok:
            st.error(f"Security verification failed: {bundle_message}")
            return
        st.success(f"Security verification passed: {bundle_message}")
        metric_text = (f"{bundle['metric_value'] * 100:.2f}%" if bundle["metric_name"] == "Accuracy"
                       else f"{bundle['metric_value']:.4f}")
        left, middle, right = st.columns(3)
        left.metric("Model type", bundle["task"])
        middle.metric(bundle["metric_name"], metric_text)
        right.metric("Target column", target)

        if st.button("Retrain model", key="retrain_prediction_model"):
            with st.spinner("Retraining model..."):
                try:
                    bundle = train_model(dp, target)
                    joblib.dump(bundle, current_model_path)
                    st.success("Model retrained and saved.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Retraining failed: {exc}")
        make_prediction_form(bundle, dp)
