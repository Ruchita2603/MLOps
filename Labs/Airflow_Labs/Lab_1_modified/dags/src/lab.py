import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64
import json
import logging

def load_data():
    """
    Loads data from a CSV file, serializes it, and returns the serialized data.
    Returns:
        str: Base64-encoded serialized data (JSON-safe).
    """
    print("We are here")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    serialized_data = pickle.dumps(df)                    # bytes
    return base64.b64encode(serialized_data).decode("ascii")  # JSON-safe string

def data_preprocessing(data_b64: str):
    """
    Deserializes base64-encoded pickled data, performs preprocessing,
    and returns base64-encoded pickled clustered data.
    """
    # decode -> bytes -> DataFrame
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    min_max_scaler = MinMaxScaler()
    clustering_data_minmax = min_max_scaler.fit_transform(clustering_data)

    # bytes -> base64 string for XCom
    clustering_serialized_data = pickle.dumps(clustering_data_minmax)
    return base64.b64encode(clustering_serialized_data).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a KMeans model on the preprocessed data and saves it.
    Returns the SSE list (JSON-serializable).
    """
    # decode -> bytes -> numpy array
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
    sse = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # NOTE: This saves the last-fitted model (k=49), matching your original intent.
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(kmeans, f)

    return sse  # list is JSON-safe


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved model and uses the elbow method to report k.
    Returns the first prediction (as a plain int) for test.csv.
    """
    # load the saved (last-fitted) model
    output_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(output_path, "rb"))

    # elbow for information/logging
    kl = KneeLocator(range(1, 50), sse, curve="convex", direction="decreasing")
    print(f"Optimal no. of clusters: {kl.elbow}")

    # predict on raw test data (matches your original code)
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    pred = loaded_model.predict(df)[0]

    # ensure JSON-safe return
    try:
        return int(pred)
    except Exception:
        return pred.item() if hasattr(pred, "item") else pred

# ---- Data validation settings ----
REQUIRED_COLUMNS = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]
MIN_ROWS = 100            
MAX_MISSING_RATIO = 0.05


def _to_dataframe(data):
    """Turn the output of load_data() back into a DataFrame (handles bytes or base64 text)."""
    if isinstance(data, str):
        data = base64.b64decode(data)
    return pickle.loads(data)


def validate_data(data, **kwargs):
    """
    Checks the loaded data and decides which branch of the DAG runs next.
    Returns the task_id of the next task.
    """
    df = _to_dataframe(data)
    problems = []

    # Check 1: required columns exist
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        problems.append(f"Missing required columns: {missing_cols}")

    # Check 2: enough rows
    if len(df) < MIN_ROWS:
        problems.append(f"Only {len(df)} rows, need at least {MIN_ROWS}")

    # Check 3: not too many missing values
    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            ratio = df[col].isna().mean()
            if ratio > MAX_MISSING_RATIO:
                problems.append(f"Column {col} has {ratio:.1%} missing values (limit {MAX_MISSING_RATIO:.0%})")

    # Share the results with the next task via XCom
    kwargs["ti"].xcom_push(key="validation_problems", value=problems)

    if problems:
        logging.warning("Data validation FAILED: %s", problems)
        return "report_bad_data_task"

    logging.info("Data validation PASSED: %d rows, all checks OK", len(df))
    return "data_preprocessing_task"


def report_bad_data(**kwargs):
    """Runs only when validation fails. Logs the problems and saves a report file."""
    problems = kwargs["ti"].xcom_pull(task_ids="validate_data_task", key="validation_problems")

    report_dir = "/opt/airflow/working_data"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "validation_report.json")
    with open(report_path, "w") as f:
        json.dump({"status": "failed", "problems": problems}, f, indent=2)

    logging.error("Pipeline stopped due to bad data. Report saved to %s", report_path)
    for p in problems:
        logging.error(" - %s", p)