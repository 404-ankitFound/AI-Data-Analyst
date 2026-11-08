import json
import re
import io

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import ollama



# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "llama3.2:3b"

ALLOWED_CHARTS = {
    "histogram", "count", "scatter", "box", "violin",
    "bar", "line", "heatmap", "kde"
}

ALLOWED_AGGREGATIONS = {
    "none", "count", "sum", "mean", "median", "min", "max"
}

ALLOWED_OPERATORS = {
    "eq", "ne", "in", "not_in", "gt", "gte", "lt", "lte", "contains"
}

NUMERIC_OPERATORS = {"gt", "gte", "lt", "lte"}

MAX_FILTERS = 8
MAX_PLOT_ROWS = 100000


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

def get_dataset_info(df):

    columns = []

    for col in df.columns:

        series = df[col]

        info = {
            "name": str(col),
            "dtype": str(series.dtype),
            "unique_values": int(series.nunique()),
            "missing_values": int(series.isna().sum())
        }

        if pd.api.types.is_numeric_dtype(series):

            clean = series.dropna()

            if not clean.empty:
                info["min"] = float(clean.min())
                info["max"] = float(clean.max())

        else:

            info["common_values"] = (
                series.dropna()
                .astype(str)
                .value_counts()
                .head(15)
                .index.tolist()
            )

        columns.append(info)

    return {
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },

        "columns": columns,

        "sample_data": (
            df.head(5)
            .replace({np.nan: None})
            .to_dict(orient="records")
        )
    }


# --------------------------------------------------
# JSON EXTRACTION
# --------------------------------------------------

def extract_json(response):

    response = response.strip()

    response = re.sub(
        r"^```(?:json)?\s*",
        "",
        response,
        flags=re.IGNORECASE
    )

    response = re.sub(r"\s*```$", "", response)

    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("Invalid JSON response from Ollama.")

    result = json.loads(response[start:end + 1])

    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object.")

    return result


# --------------------------------------------------
# GENERATE VISUALIZATION PLAN
# --------------------------------------------------

def generate_visualization_plan(question, df):

    dataset_info = get_dataset_info(df)

    system_prompt = """
You are an expert data analyst and visualization planner.

Your job is to understand the user's analytical question
and generate a correct visualization plan.

IMPORTANT:
You must NEVER generate Python code.
You must NEVER generate Pandas expressions.
You must NEVER generate Seaborn expressions.
You must NEVER generate executable code.

Return ONLY valid JSON.

EXPECTED FORMAT:

{
    "chart": "bar",
    "x": "column_name",
    "y": "column_name",
    "hue": null,
    "aggregation": "mean",
    "filters": [],
    "title": "Chart title",
    "explanation": "Chart explanation"
}

SUPPORTED CHARTS:
histogram, count, scatter, box, violin, bar, line, heatmap, kde

SUPPORTED AGGREGATIONS:
none, count, sum, mean, median, min, max

FILTER FORMAT:
{"column": "column_name", "operator": "eq", "value": "value"}

SUPPORTED OPERATORS:
eq, ne, in, not_in, gt, gte, lt, lte, contains

ANALYTICAL RULES:

1. Use exact column names from the dataset.
2. Never invent column names.
3. Never invent category values.
4. Understand whether the user wants distribution, comparison, relationship, correlation or trend.
5. Use histogram for numerical distributions.
6. Use count for categorical frequencies.
7. Use scatter for relationships between two numerical columns.
8. Use box or violin for numerical distributions across categories.
9. Use bar for categorical comparisons.
10. Use line for ordered trends.
11. Use heatmap for numerical correlations.
12. Use mean for average.
13. Use sum for total.
14. Use count for number of records.
15. Use median when median is requested.
16. Include filters only when explicitly requested.
17. Do not confuse a column name with a filter value.
18. For bar and line charts, use aggregation=count when user requests record counts.
19. For histogram and KDE, x must be numerical.
20. For scatter, both x and y must be numerical.
21. For box and violin, y must be numerical.
22. If ambiguous or impossible, return {"error":"Explain the problem"}.
23. Do not guess.
24. Return JSON only.
"""

    user_prompt = f"""
DATASET INFORMATION:

{json.dumps(dataset_info, default=str)}

USER QUESTION:

{question}

Generate the most appropriate visualization plan.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        format="json",
        options={
            "temperature": 0,
            "num_predict": 350
        }
    )

    return extract_json(response["message"]["content"])


# --------------------------------------------------
# SECURITY VERIFICATION
# --------------------------------------------------

def verify_visualization_plan(plan, df):

    if not isinstance(df, pd.DataFrame) or df.empty:
        return False, "Dataset is empty or invalid."

    if df.columns.duplicated().any():
        return False, "Dataset contains duplicate column names."

    if not isinstance(plan, dict):
        return False, "Invalid visualization plan."

    allowed_keys = {
        "chart", "x", "y", "hue", "aggregation",
        "filters", "title", "explanation", "error"
    }

    if set(plan) - allowed_keys:
        return False, "Unexpected fields detected."

    if plan.get("error"):
        if not isinstance(plan.get("error"), str):
            return False, "Invalid error field in visualization plan."
        return False, plan["error"][:500]

    for text_key, max_length in (("title", 120), ("explanation", 1000)):
        value = plan.get(text_key, "")
        if not isinstance(value, str) or len(value) > max_length:
            return False, f"Invalid or excessively long {text_key}."

    chart = plan.get("chart")

    if chart not in ALLOWED_CHARTS:
        return False, "Unsupported chart type."

    columns = set(df.columns)

    for key in ["x", "y", "hue"]:

        value = plan.get(key)

        if value is not None:

            if not isinstance(value, str):
                return False, f"Invalid {key} column."

            if value not in columns:
                return False, f"Unknown column: {value}"

    aggregation = plan.get("aggregation", "none")

    if not isinstance(aggregation, str) or aggregation not in ALLOWED_AGGREGATIONS:
        return False, "Invalid aggregation."

    filters = plan.get("filters", [])

    if not isinstance(filters, list):
        return False, "Invalid filters."

    if len(filters) > MAX_FILTERS:
        return False, "Too many filters."

    for item in filters:

        if not isinstance(item, dict):
            return False, "Invalid filter structure."

        if set(item) != {"column", "operator", "value"}:
            return False, "Unexpected filter fields."

        col = item["column"]
        op = item["operator"]
        value = item["value"]

        if not isinstance(col, str) or col not in columns:
            return False, f"Unknown filter column: {col}"

        if not isinstance(op, str) or op not in ALLOWED_OPERATORS:
            return False, "Invalid filter operator."

        if op in NUMERIC_OPERATORS:

            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, "Numeric operator on non-numeric column."

        if op == "contains":

            if pd.api.types.is_numeric_dtype(df[col]):
                return False, "Contains requires a text column."

        if op in {"in", "not_in"}:

            if not isinstance(value, list):
                return False, "Expected a list for filter."

        elif isinstance(value, (dict, list)):
            return False, "Invalid filter value."

        if isinstance(value, str) and len(value) > 500:
            return False, "Filter value too long."

    if chart in {"histogram", "kde"}:

        if not plan.get("x"):
            return False, "Numeric x column required."

        if not pd.api.types.is_numeric_dtype(df[plan["x"]]):
            return False, "Numeric x column required."

    elif chart == "count":

        if not plan.get("x"):
            return False, "Categorical x column required."

    elif chart == "scatter":

        if not plan.get("x") or not plan.get("y"):
            return False, "Both x and y are required."

        if not pd.api.types.is_numeric_dtype(df[plan["x"]]):
            return False, "Scatter requires numeric x."

        if not pd.api.types.is_numeric_dtype(df[plan["y"]]):
            return False, "Scatter requires numeric y."

    elif chart in {"box", "violin"}:

        if not plan.get("y"):
            return False, "Numeric y column required."

        if not pd.api.types.is_numeric_dtype(df[plan["y"]]):
            return False, "Numeric y column required."

    elif chart in {"bar", "line"}:

        if not plan.get("x") or not plan.get("y"):
            return False, "Both x and y are required."

        if aggregation != "count" and not pd.api.types.is_numeric_dtype(df[plan["y"]]):
            return False, "Numeric y column required."

    elif chart == "heatmap":

        if df.select_dtypes(include="number").empty:
            return False, "No numerical columns found."

    return True, "Security verification passed."


# --------------------------------------------------
# APPLY FILTERS
# --------------------------------------------------

def apply_filters(df, filters):

    result = df.copy()

    for item in filters:

        col = item["column"]
        op = item["operator"]
        value = item["value"]

        series = result[col]

        if op == "eq":

            result = result[series.astype(str) == str(value)]

        elif op == "ne":

            result = result[series.astype(str) != str(value)]

        elif op == "in":

            result = result[
                series.astype(str).isin([str(v) for v in value])
            ]

        elif op == "not_in":

            result = result[
                ~series.astype(str).isin([str(v) for v in value])
            ]

        elif op in NUMERIC_OPERATORS:

            numeric_series = pd.to_numeric(series, errors="coerce")
            numeric_value = pd.to_numeric(value, errors="raise")

            if op == "gt":
                result = result[numeric_series > numeric_value]

            elif op == "gte":
                result = result[numeric_series >= numeric_value]

            elif op == "lt":
                result = result[numeric_series < numeric_value]

            elif op == "lte":
                result = result[numeric_series <= numeric_value]

        elif op == "contains":

            result = result[
                series.astype("string").str.contains(
                    re.escape(str(value)),
                    case=False,
                    na=False
                )
            ]

    return result


# --------------------------------------------------
# DATA AGGREGATION
# --------------------------------------------------

def aggregate_data(df, plan):

    aggregation = plan.get("aggregation", "none")

    x = plan.get("x")
    y = plan.get("y")
    hue = plan.get("hue")

    if aggregation == "none":
        return df

    group_columns = [
        col for col in [x, hue]
        if col is not None
    ]

    if aggregation == "count":

        if not group_columns:
            return pd.DataFrame({"_count": [len(df)]})

        return (
            df.groupby(group_columns, dropna=False)
            .size()
            .reset_index(name="_count")
        )

    if y is None:
        raise ValueError("Aggregation requires a y column.")

    if not group_columns:

        return pd.DataFrame({
            y: [getattr(df[y], aggregation)()]
        })

    return (
        df.groupby(group_columns, dropna=False)[y]
        .agg(aggregation)
        .reset_index()
    )


# --------------------------------------------------
# GENERATE CHART + DOWNLOAD IMAGE
# --------------------------------------------------

def generate_chart(df, plan):

    chart = plan["chart"]

    x = plan.get("x")
    y = plan.get("y")
    hue = plan.get("hue")

    aggregation = plan.get("aggregation", "none")

    if df.empty:

        st.warning("No records found after applying filters.")
        return

    if len(df) > MAX_PLOT_ROWS:

        df = df.sample(
            MAX_PLOT_ROWS,
            random_state=42
        )

        st.info(
            f"Showing a sample of {MAX_PLOT_ROWS:,} records."
        )

    fig, ax = plt.subplots(figsize=(11, 5.5))

    if chart == "histogram":

        sns.histplot(
            data=df,
            x=x,
            kde=True,
            ax=ax
        )

    elif chart == "kde":

        sns.kdeplot(
            data=df,
            x=x,
            fill=True,
            ax=ax
        )

    elif chart == "count":

        sns.countplot(
            data=df,
            x=x,
            hue=hue,
            ax=ax
        )

    elif chart == "scatter":

        sns.scatterplot(
            data=df,
            x=x,
            y=y,
            hue=hue,
            alpha=0.75,
            ax=ax
        )

    elif chart in {"box", "violin"}:

        plot_function = (
            sns.boxplot if chart == "box"
            else sns.violinplot
        )

        plot_function(
            data=df,
            x=x,
            y=y,
            hue=hue,
            ax=ax
        )

    elif chart in {"bar", "line"}:

        plot_df = aggregate_data(df, plan)

        plot_y = (
            "_count" if aggregation == "count"
            else y
        )

        if chart == "bar":

            sns.barplot(
                data=plot_df,
                x=x,
                y=plot_y,
                hue=hue,
                errorbar=None,
                ax=ax
            )

        else:

            sns.lineplot(
                data=plot_df,
                x=x,
                y=plot_y,
                hue=hue,
                marker="o",
                errorbar=None,
                ax=ax
            )

    elif chart == "heatmap":

        correlation = (
            df.select_dtypes(include="number")
            .corr()
        )

        sns.heatmap(
            correlation,
            annot=len(correlation.columns) <= 12,
            cmap="coolwarm",
            center=0,
            ax=ax
        )

    # Chart title and layout

    ax.set_title(
        str(plan.get("title", "Data Visualization"))[:120],
        fontsize=15,
        pad=15
    )

    ax.grid(
        axis="y",
        alpha=0.2
    )

    fig.tight_layout()

    # Display visualization

    st.pyplot(
        fig,
        use_container_width=True
    )

    # ----------------------------------------------
    # DOWNLOAD GENERATED VISUALIZATION AS PNG
    # ----------------------------------------------

    image_buffer = io.BytesIO()

    fig.savefig(
        image_buffer,
        format="png",
        dpi=300,
        bbox_inches="tight"
    )

    image_buffer.seek(0)

    st.download_button(
        label="📥 Download Visualization",
        data=image_buffer.getvalue(),
        file_name="visualization.png",
        mime="image/png",
        key="download_visualization"
    )

    plt.close(fig)



def render(df: pd.DataFrame):
    """Render visualization module using the app-level uploaded dataframe."""
    st.title("📈 Visualization")
    st.caption("Ask analytical questions and generate intelligent visualizations.")
    if df is None or df.empty:
        st.info("Upload a CSV from Home to start visualization.")
        return
    if df.columns.duplicated().any():
        st.error("Duplicate column names are not supported.")
        return

    st.success("Security verification passed: dataset structure is valid.")
    st.caption(f"Working with {df.shape[0]:,} rows and {df.shape[1]} columns.")
    with st.expander("👀 Preview Dataset"):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 of {len(df):,} rows and {len(df.columns)} columns.")

    dv = df.copy(deep=True)
    question = st.text_area(
        "Ask your visualization question",
        placeholder="Example: Show average sales across different regions",
        height=100,
        key="visualization_question"
    )

    if st.button("Generate Visualization", type="primary", key="generate_visualization"):
        if not question.strip():
            st.warning("Please enter a visualization question.")
        else:
            with st.spinner("Understanding your question and generating visualization..."):
                try:
                    plan = generate_visualization_plan(question, dv)
                    valid, message = verify_visualization_plan(plan, dv)
                    if not valid:
                        st.error(f"Visualization rejected: {message}")
                        st.caption("The generated plan was not executed.")
                    else:
                        st.success(f"Security verification passed: {message}")
                        filtered_df = apply_filters(dv, plan.get("filters", []))
                        st.session_state["viz_plan"] = plan
                        st.session_state["viz_data"] = filtered_df
                        st.session_state["viz_dataset_id"] = st.session_state.get("dataset_id")
                        st.session_state["viz_verified"] = True
                except Exception as exc:
                    st.error(f"Visualization failed: {exc}")

    if (st.session_state.get("viz_verified", False)
            and st.session_state.get("viz_dataset_id") == st.session_state.get("dataset_id")):
        plan = st.session_state["viz_plan"]
        filtered_df = st.session_state["viz_data"]
        st.divider()
        st.subheader("Graphical Analysis")
        st.write(plan.get("explanation", "Visualization generated successfully."))
        st.success("Security verification passed: the stored visualization plan is approved for rendering.")
        st.caption(f"Chart: {plan['chart']} | Records: {len(filtered_df):,}")
        if plan.get("filters"):
            with st.expander("Applied Filters"):
                st.json(plan["filters"])
        generate_chart(filtered_df, plan)
        with st.expander("Visualization Details"):
            st.json(plan)
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered Dataset", data=csv,
                           file_name="filtered_data.csv", mime="text/csv",
                           key="download_filtered_dataset")
