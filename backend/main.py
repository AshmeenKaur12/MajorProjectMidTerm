import os
import json
import warnings

import numpy as np
import pandas as pd
import joblib
import shap

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Kidney Compatibility Prediction API",
    version="2.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# BASE DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# MODEL FOLDER
#
# KEEP THIS FOLDER EXACTLY AS IT IS
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "final_xgboost_model.joblib"
)


PREPROCESSOR_PATH = os.path.join(
    MODEL_DIR,
    "final_preprocessor.joblib"
)


CALIBRATOR_PATH = os.path.join(
    MODEL_DIR,
    "platt_calibrator.joblib"
)


METADATA_PATH = os.path.join(
    MODEL_DIR,
    "model_metadata.json"
)


# ============================================================
# OBJECTIVE 2 XAI FOLDER
# ============================================================

XAI_DIR = os.path.join(
    BASE_DIR,
    "objective_2_xai"
)


OBJECTIVE_2_METADATA_PATH = os.path.join(
    XAI_DIR,
    "objective_2_metadata.json"
)


OBJECTIVE_2_PREDICTIONS_PATH = os.path.join(
    XAI_DIR,
    "objective_2_test_predictions.csv"
)


PATIENT_SHAP_PATH = os.path.join(
    XAI_DIR,
    "patient_level_shap_values.csv"
)


IMMUNOLOGICAL_SHAP_PATH = os.path.join(
    XAI_DIR,
    "immunological_feature_shap.csv"
)


IMMUNOLOGICAL_SUMMARY_PATH = os.path.join(
    XAI_DIR,
    "immunological_summary.csv"
)


LIME_CASE_SUMMARY_PATH = os.path.join(
    XAI_DIR,
    "lime_case_summary.csv"
)


LIME_HIGH_PRED_PATH = os.path.join(
    XAI_DIR,
    "lime_highest_predicted_incompatibility.csv"
)


LIME_HIGH_TRUE_PATH = os.path.join(
    XAI_DIR,
    "lime_highest_probability_among_true.csv"
)


LOCAL_CASE_SUMMARY_PATH = os.path.join(
    XAI_DIR,
    "local_case_summary.csv"
)


LOCAL_SHAP_HIGH_PRED_PATH = os.path.join(
    XAI_DIR,
    "local_shap_highest_predicted_incompatibility.csv"
)


LOCAL_SHAP_HIGH_TRUE_PATH = os.path.join(
    XAI_DIR,
    "local_shap_highest_probability_among_true.csv"
)


SHAP_GLOBAL_ORIGINAL_PATH = os.path.join(
    XAI_DIR,
    "shap_global_original_features.csv"
)


SHAP_GLOBAL_PROCESSED_PATH = os.path.join(
    XAI_DIR,
    "shap_global_processed_features.csv"
)


# ============================================================
# VERIFY MODEL FILES
#
# These files MUST exist.
# ============================================================

required_model_files = {
    "model": MODEL_PATH,
    "preprocessor": PREPROCESSOR_PATH,
    "calibrator": CALIBRATOR_PATH,
    "metadata": METADATA_PATH,
}


for name, path in required_model_files.items():

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"Required {name} file not found: {path}"
        )


# ============================================================
# LOAD MODEL ARTIFACTS
# ============================================================

model = joblib.load(
    MODEL_PATH
)


preprocessor = joblib.load(
    PREPROCESSOR_PATH
)


calibrator = joblib.load(
    CALIBRATOR_PATH
)


with open(
    METADATA_PATH,
    "r"
) as f:

    metadata = json.load(f)


# ============================================================
# OBJECTIVE 2 XAI FILES
#
# These are OPTIONAL.
#
# The API will still start if one of these files is missing.
# ============================================================

objective_2_metadata = {}

if os.path.isfile(
    OBJECTIVE_2_METADATA_PATH
):

    try:

        with open(
            OBJECTIVE_2_METADATA_PATH,
            "r"
        ) as f:

            objective_2_metadata = json.load(f)

    except Exception as exc:

        warnings.warn(
            f"Could not load objective_2_metadata.json: {exc}"
        )


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "yrtx",
    "prevtx",
    "ageattx",
    "sex",
    "ethnicity",
    "prd",
    "rdm",
    "rbmi",
    "premt_tx",
    "dtype",
    "dageattx",
    "dcmv",
    "debv",
    "rcmv",
    "rebv",
    "mma",
    "mmb",
    "mmdr",
    "totalmm",
    "abomm",
    "crf",
    "cit",
    "induction",
    "pra",
    "age60",
    "citcat",
    "age10",
]


# ============================================================
# ACTIVE MODEL THRESHOLD
# ============================================================

try:

    ACTIVE_THRESHOLD = float(
        metadata[
            "threshold_selection"
        ][
            "active_threshold"
        ]
    )

except Exception:

    ACTIVE_THRESHOLD = 0.50


# ============================================================
# REQUEST MODEL
# ============================================================

class PatientData(BaseModel):

    yrtx: int | None = None
    prevtx: int | None = None

    ageattx: float | None = None

    sex: str | None = None
    ethnicity: str | None = None

    prd: str | None = None
    rdm: str | None = None

    rbmi: float | None = None

    premt_tx: int | None = None

    dtype: str | None = None

    dageattx: float | None = None

    dcmv: str | None = None
    debv: str | None = None

    rcmv: str | None = None
    rebv: str | None = None

    mma: float | None = None
    mmb: float | None = None
    mmdr: float | None = None

    totalmm: float | None = None
    abomm: float | None = None

    crf: float | None = None
    cit: float | None = None

    induction: str | None = None

    pra: float | None = None

    age60: str | None = None
    citcat: str | None = None
    age10: str | None = None


# ============================================================
# LOAD PREPROCESSOR COLUMN INFORMATION
# ============================================================

def get_preprocessor_columns():

    numeric_columns = []
    categorical_columns = []

    if not hasattr(
        preprocessor,
        "transformers_"
    ):

        return (
            numeric_columns,
            categorical_columns
        )

    for (
        name,
        transformer,
        columns
    ) in preprocessor.transformers_:

        if name == "remainder":
            continue

        if columns is None:
            continue

        columns = list(
            columns
        )

        transformer_name = str(
            transformer
        ).lower()

        pipeline_name = str(
            name
        ).lower()

        # ----------------------------------------------------
        # Numeric
        # ----------------------------------------------------

        if (
            "numeric" in pipeline_name
            or (
                "simpleimputer"
                in transformer_name
                and (
                    "median"
                    in transformer_name
                    or
                    "mean"
                    in transformer_name
                )
            )
        ):

            numeric_columns.extend(
                columns
            )

        # ----------------------------------------------------
        # Categorical
        # ----------------------------------------------------

        elif (
            "categor"
            in pipeline_name
            or
            "onehot"
            in transformer_name
        ):

            categorical_columns.extend(
                columns
            )

    numeric_columns = list(
        dict.fromkeys(
            numeric_columns
        )
    )

    categorical_columns = list(
        dict.fromkeys(
            categorical_columns
        )
    )

    return (
        numeric_columns,
        categorical_columns
    )


# ============================================================
# PROCESSED FEATURE NAMES
# ============================================================

try:

    PROCESSED_FEATURE_NAMES = list(
        preprocessor.get_feature_names_out()
    )

except Exception:

    PROCESSED_FEATURE_NAMES = None


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(data: dict):

    # --------------------------------------------------------
    # Exact training feature order
    # --------------------------------------------------------

    row = {
        feature: data.get(feature)
        for feature in FEATURES
    }

    X = pd.DataFrame(
        [row],
        columns=FEATURES
    )

    (
        numeric_columns,
        categorical_columns
    ) = get_preprocessor_columns()

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for column in numeric_columns:

        if column not in X.columns:
            continue

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    for column in categorical_columns:

        if column not in X.columns:
            continue

        X[column] = X[column].astype(
            "object"
        )

    return X


# ============================================================
# TRANSFORM INPUT
# ============================================================

def transform_input(data: dict):

    X = prepare_input(
        data
    )

    X_processed = preprocessor.transform(
        X
    )

    X_processed = np.asarray(
        X_processed
    )

    return (
        X,
        X_processed
    )


# ============================================================
# CALIBRATION
# ============================================================

def calibrate_probability(
    raw_probability: float
):

    eps = 1e-7

    probability = np.clip(
        raw_probability,
        eps,
        1.0 - eps
    )

    logit = np.log(
        probability /
        (
            1.0 - probability
        )
    )

    calibrated = calibrator.predict_proba(
        np.array(
            logit
        ).reshape(
            -1,
            1
        )
    )[:, 1]

    return float(
        calibrated[0]
    )


# ============================================================
# RISK CATEGORY
# ============================================================

def risk_category(
    probability: float
):

    if probability < 0.10:

        return (
            "Lower predicted incompatibility risk"
        )

    if probability < ACTIVE_THRESHOLD:

        return (
            "Moderate predicted incompatibility risk"
        )

    return (
        "Higher predicted incompatibility risk"
    )


# ============================================================
# SHAP EXPLAINER
# ============================================================

try:

    shap_explainer = shap.TreeExplainer(
        model
    )

    SHAP_AVAILABLE = True

except Exception as exc:

    shap_explainer = None

    SHAP_AVAILABLE = False

    warnings.warn(
        f"SHAP initialization failed: {exc}"
    )


# ============================================================
# SHAP VALUE EXTRACTION
# ============================================================

def get_shap_values(
    X_processed
):

    if shap_explainer is None:

        raise RuntimeError(
            "SHAP explainer is not available."
        )

    shap_output = (
        shap_explainer.shap_values(
            X_processed
        )
    )

    # --------------------------------------------------------
    # Older SHAP versions
    # --------------------------------------------------------

    if isinstance(
        shap_output,
        list
    ):

        shap_values = (
            shap_output[-1][0]
        )

    else:

        shap_values = np.asarray(
            shap_output
        )

        # ----------------------------------------------------
        # Newer SHAP versions
        # ----------------------------------------------------

        if shap_values.ndim == 3:

            shap_values = (
                shap_values[
                    0,
                    :,
                    -1
                ]
            )

        elif shap_values.ndim == 2:

            shap_values = (
                shap_values[0]
            )

        elif shap_values.ndim == 1:

            shap_values = (
                shap_values
            )

        else:

            raise RuntimeError(
                "Unexpected SHAP output shape: "
                f"{shap_values.shape}"
            )

    return np.asarray(
        shap_values
    ).ravel()


# ============================================================
# SHAP FEATURE NAME
# ============================================================

def get_feature_name(
    index
):

    if (
        PROCESSED_FEATURE_NAMES
        and
        index <
        len(PROCESSED_FEATURE_NAMES)
    ):

        return PROCESSED_FEATURE_NAMES[
            index
        ]

    return f"feature_{index}"


# ============================================================
# CREATE SHAP RESULTS
# ============================================================

def create_shap_results(
    shap_values
):

    results = []

    for index, value in enumerate(
        shap_values
    ):

        feature_name = get_feature_name(
            index
        )

        value = float(
            value
        )

        results.append({

            "feature": feature_name,

            "shap_value": round(
                value,
                6
            ),

            "abs_shap_value": round(
                abs(value),
                6
            ),

            "direction": (
                "increases predicted incompatibility"
                if value > 0
                else
                "decreases predicted incompatibility"
                if value < 0
                else
                "no contribution"
            )
        })

    results.sort(
        key=lambda x:
        x["abs_shap_value"],
        reverse=True
    )

    return results


# ============================================================
# LIME TRAINING DATA
#
# Objective 2 folder is checked first.
# ============================================================

LIME_TRAINING_DATA_PATHS = [

    os.path.join(
        XAI_DIR,
        "lime_training_data.npy"
    ),

    os.path.join(
        MODEL_DIR,
        "lime_training_data.npy"
    )
]


lime_training_data = None

for path in LIME_TRAINING_DATA_PATHS:

    if os.path.isfile(path):

        try:

            lime_training_data = np.load(
                path
            )

            break

        except Exception as exc:

            warnings.warn(
                f"Could not load LIME data {path}: {exc}"
            )


# ============================================================
# LIME EXPLAINER
# ============================================================

lime_explainer = None

if (
    lime_training_data is not None
    and
    PROCESSED_FEATURE_NAMES is not None
):

    try:

        lime_explainer = LimeTabularExplainer(

            training_data=lime_training_data,

            feature_names=(
                PROCESSED_FEATURE_NAMES
            ),

            class_names=[
                "Compatible",
                "Incompatible"
            ],

            mode="classification",

            discretize_continuous=True,

            random_state=42
        )

    except Exception as exc:

        warnings.warn(
            f"LIME initialization failed: {exc}"
        )

        lime_explainer = None


# ============================================================
# LIME PREDICT PROBA
# ============================================================



# ============================================================
# LIME EXPLANATION
# ============================================================

def create_lime_results(
    X_processed
):

    if lime_explainer is None:

        return []

    explanation = (
        lime_explainer.explain_instance(

            X_processed[0],

            lime_predict_proba,

            num_features=15,

            top_labels=2
        )
    )

    try:

        lime_list = explanation.as_list(
            label=1
        )

    except Exception:

        lime_list = explanation.as_list()

    results = []

    for feature, weight in lime_list:

        weight = float(
            weight
        )

        results.append({

            "feature": feature,

            "lime_weight": round(
                weight,
                6
            ),

            "abs_lime_weight": round(
                abs(weight),
                6
            ),

            "direction": (
                "increases predicted incompatibility"
                if weight > 0
                else
                "decreases predicted incompatibility"
                if weight < 0
                else
                "no contribution"
            )
        })

    return results


# ============================================================
# IMMUNOLOGICAL FEATURES
# ============================================================

IMMUNOLOGICAL_FEATURES = [
    "pra",
    "mma",
    "mmb",
    "mmdr",
    "totalmm",
    "abomm"
]


# ============================================================
# GROUP SHAP BY ORIGINAL FEATURE
# ============================================================

def get_original_feature_name(
    processed_name
):

    name = str(
        processed_name
    )

    # --------------------------------------------------------
    # Handle:
    # num__pra
    # cat__sex_M
    # --------------------------------------------------------

    if "__" in name:

        name = name.split(
            "__",
            1
        )[1]

    # --------------------------------------------------------
    # Direct feature
    # --------------------------------------------------------

    for feature in FEATURES:

        if name == feature:

            return feature

    # --------------------------------------------------------
    # One-hot categorical feature
    #
    # Examples:
    # sex_M
    # ethnicity_White
    # --------------------------------------------------------

    for feature in FEATURES:

        if name.startswith(
            feature + "_"
        ):

            return feature

    return name


# ============================================================
# IMMUNOLOGICAL SHAP ANALYSIS
# ============================================================

def create_immunological_results(
    shap_results
):

    results = []

    for feature in IMMUNOLOGICAL_FEATURES:

        matching = []

        for item in shap_results:

            original_name = (
                get_original_feature_name(
                    item["feature"]
                )
            )

            if original_name == feature:

                matching.append(
                    item
                )

        importance = sum(
            item["abs_shap_value"]
            for item in matching
        )

        signed_contribution = sum(
            item["shap_value"]
            for item in matching
        )

        results.append({

            "feature": feature,

            "mean_abs_shap": round(
                float(importance),
                6
            ),

            "shap_contribution": round(
                float(signed_contribution),
                6
            ),

            "direction": (
                "increases predicted incompatibility"
                if signed_contribution > 0
                else
                "decreases predicted incompatibility"
                if signed_contribution < 0
                else
                "no contribution"
            )
        })

    results.sort(
        key=lambda x:
        x["mean_abs_shap"],
        reverse=True
    )

    return results


# ============================================================
# LOAD OBJECTIVE 2 CSV
# ============================================================

def safe_read_csv(
    path
):

    if not os.path.isfile(
        path
    ):

        return None

    try:

        return pd.read_csv(
            path
        )

    except Exception as exc:

        warnings.warn(
            f"Could not read {path}: {exc}"
        )

        return None


# ============================================================
# OBJECTIVE 2 SUMMARY
# ============================================================

def get_objective_2_summary():

    files = {

        "objective_2_metadata":
            OBJECTIVE_2_METADATA_PATH,

        "test_predictions":
            OBJECTIVE_2_PREDICTIONS_PATH,

        "patient_level_shap":
            PATIENT_SHAP_PATH,

        "immunological_feature_shap":
            IMMUNOLOGICAL_SHAP_PATH,

        "immunological_summary":
            IMMUNOLOGICAL_SUMMARY_PATH,

        "lime_case_summary":
            LIME_CASE_SUMMARY_PATH,

        "lime_highest_predicted_incompatibility":
            LIME_HIGH_PRED_PATH,

        "lime_highest_probability_among_true":
            LIME_HIGH_TRUE_PATH,

        "local_case_summary":
            LOCAL_CASE_SUMMARY_PATH,

        "local_shap_highest_predicted_incompatibility":
            LOCAL_SHAP_HIGH_PRED_PATH,

        "local_shap_highest_probability_among_true":
            LOCAL_SHAP_HIGH_TRUE_PATH,

        "shap_global_original_features":
            SHAP_GLOBAL_ORIGINAL_PATH,

        "shap_global_processed_features":
            SHAP_GLOBAL_PROCESSED_PATH,
    }

    result = {}

    for name, path in files.items():

        result[name] = {

            "exists":
                os.path.isfile(path),

            "path":
                os.path.relpath(
                    path,
                    BASE_DIR
                )
        }

    return result


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "status": "online",

        "service":
            "Kidney Compatibility Prediction API",

        "version":
            "2.0.0",

        "model_loaded":
            True,

        "objective_2_xai":
            os.path.isdir(XAI_DIR)
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    (
        numeric_columns,
        categorical_columns
    ) = get_preprocessor_columns()

    return {

        "status":
            "healthy",

        "model_loaded":
            True,

        "threshold":
            ACTIVE_THRESHOLD,

        "features":
            len(FEATURES),

        "numeric_features":
            numeric_columns,

        "categorical_features":
            categorical_columns,

        "xai": {

            "shap":
                SHAP_AVAILABLE,

            "lime":
                lime_explainer is not None,

            "immunological_analysis":
                True,

            "objective_2_folder":
                os.path.isdir(XAI_DIR),

            "objective_2_metadata":
                os.path.isfile(
                    OBJECTIVE_2_METADATA_PATH
                ),

            "patient_level_shap":
                os.path.isfile(
                    PATIENT_SHAP_PATH
                ),

            "objective_2_predictions":
                os.path.isfile(
                    OBJECTIVE_2_PREDICTIONS_PATH
                )
        }
    }


# ============================================================
# OBJECTIVE 2 INFO
# ============================================================

@app.get("/xai/info")
def xai_info():

    return {

        "objective_2_folder":
            os.path.relpath(
                XAI_DIR,
                BASE_DIR
            ),

        "files":
            get_objective_2_summary(),

        "metadata":
            objective_2_metadata
    }


# ============================================================
# PREDICT
# ============================================================

@app.post("/predict")
def predict(
    patient: PatientData
):

    try:

        # ----------------------------------------------------
        # Pydantic -> dictionary
        # ----------------------------------------------------

        data = patient.model_dump()

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        X, X_processed = (
            transform_input(data)
        )

        # ----------------------------------------------------
        # Raw XGBoost probability
        # ----------------------------------------------------

        raw_probability = float(
            model.predict_proba(
                X_processed
            )[0, 1]
        )

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        calibrated_probability = (
            calibrate_probability(
                raw_probability
            )
        )

        # ----------------------------------------------------
        # Classification
        #
        # Keep your original threshold logic.
        # ----------------------------------------------------

        prediction = int(
            calibrated_probability >= ACTIVE_THRESHOLD
        )


        classification = (
            "Incompatible"
            if prediction == 1
            else "Compatible"
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "prediction":
                prediction,

            "classification":
                classification,

            "raw_probability":
                round(
                    raw_probability,
                    4
                ),

            "incompatibility_probability":
                round(
                    calibrated_probability,
                    4
                ),

            "compatibility_probability":
                round(
                    1.0 -
                    calibrated_probability,
                    4
                ),

            "threshold":
                ACTIVE_THRESHOLD,

            "risk_category":
                risk_category(
                    calibrated_probability
                )
        }

    except Exception as exc:

        print(
            "Prediction error:",
            repr(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(exc)}"
            )
        )


# ============================================================
# EXPLAIN
#
# Live patient-level SHAP + LIME
# ============================================================
def lime_predict_proba(
    X
):

    X = np.asarray(
        X
    )

    raw_probability = (
        model.predict_proba(X)[:, 1]
    )

    return np.column_stack([
        1.0 - raw_probability,
        raw_probability
    ])

@app.post("/explain")
def explain(
    patient: PatientData
):

    try:

        # ----------------------------------------------------
        # Pydantic -> dictionary
        # ----------------------------------------------------

        data = patient.model_dump()

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        X, X_processed = (
            transform_input(data)
        )

        # ----------------------------------------------------
        # RAW MODEL PREDICTION
        # ----------------------------------------------------

        raw_probability = float(
            model.predict_proba(
                X_processed
            )[0, 1]
        )

        # ----------------------------------------------------
        # CALIBRATED PROBABILITY
        # ----------------------------------------------------

        calibrated_probability = (
            calibrate_probability(
                raw_probability
            )
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        prediction = int(
            calibrated_probability >=
            ACTIVE_THRESHOLD
        )

        classification = (
            "Incompatible"
            if prediction == 1
            else "Compatible"
        )

        # ====================================================
        # SHAP
        # ====================================================

        shap_results = []

        shap_error = None

        if SHAP_AVAILABLE:

            try:

                shap_values = (
                    get_shap_values(
                        X_processed
                    )
                )

                shap_results = (
                    create_shap_results(
                        shap_values
                    )
                )

            except Exception as exc:

                shap_error = str(
                    exc
                )

        # ====================================================
        # LIME
        # ====================================================

        lime_results = []

        lime_error = None

        if lime_explainer is not None:

            try:

                lime_results = (
                    create_lime_results(
                        X_processed
                    )
                )

            except Exception as exc:

                lime_error = str(
                    exc
                )

        # ====================================================
        # IMMUNOLOGICAL SHAP
        # ====================================================

        immunological_results = (
            create_immunological_results(
                shap_results
            )
        )

        # ====================================================
        # TOP POSITIVE SHAP
        #
        # Positive SHAP -> higher model output
        # Negative SHAP -> lower model output
        # ====================================================

        top_incompatibility_factors = [

            item

            for item in shap_results

            if item["shap_value"] > 0

        ][:10]

        # ====================================================
        # TOP NEGATIVE SHAP
        # ====================================================

        top_compatibility_factors = [

            item

            for item in shap_results

            if item["shap_value"] < 0

        ][:10]

        # ====================================================
        # RESPONSE
        # ====================================================

        return {

            "prediction":
                prediction,

            "classification":
                classification,

            "raw_probability":
                round(
                    raw_probability,
                    6
                ),

            "calibrated_probability":
                round(
                    calibrated_probability,
                    6
                ),

            "incompatibility_probability":
                round(
                    calibrated_probability,
                    6
                ),

            "compatibility_probability":
                round(
                    1.0 -
                    calibrated_probability,
                    6
                ),

            "threshold":
                ACTIVE_THRESHOLD,

            "risk_category":
                risk_category(
                    calibrated_probability
                ),

            "xai": {

                "shap_available":
                    SHAP_AVAILABLE,

                "lime_available":
                    lime_explainer is not None,

                "shap_error":
                    shap_error,

                "lime_error":
                    lime_error,

                "shap":
                    shap_results[:20],

                "lime":
                    lime_results[:15],

                "immunological_factors":
                    immunological_results,

                "top_incompatibility_factors":
                    top_incompatibility_factors,

                "top_compatibility_factors":
                    top_compatibility_factors
            },

            "objective_2": {

                "folder":
                    os.path.relpath(
                        XAI_DIR,
                        BASE_DIR
                    ),

                "metadata_available":
                    os.path.isfile(
                        OBJECTIVE_2_METADATA_PATH
                    ),

                "patient_level_shap_available":
                    os.path.isfile(
                        PATIENT_SHAP_PATH
                    ),

                "global_shap_available":
                    (
                        os.path.isfile(
                            SHAP_GLOBAL_ORIGINAL_PATH
                        )
                        or
                        os.path.isfile(
                            SHAP_GLOBAL_PROCESSED_PATH
                        )
                    ),

                "immunological_analysis_available":
                    (
                        os.path.isfile(
                            IMMUNOLOGICAL_SHAP_PATH
                        )
                        or
                        os.path.isfile(
                            IMMUNOLOGICAL_SUMMARY_PATH
                        )
                    ),

                "lime_analysis_available":
                    (
                        os.path.isfile(
                            LIME_CASE_SUMMARY_PATH
                        )
                    )
            }
        }

    except Exception as exc:

        print(
            "XAI error:",
            repr(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "XAI explanation failed: "
                f"{str(exc)}"
            )
        )


# ============================================================
# OBJECTIVE 2 GLOBAL SHAP
# ============================================================

@app.get("/xai/global")
def xai_global():

    try:

        original_df = safe_read_csv(
            SHAP_GLOBAL_ORIGINAL_PATH
        )

        processed_df = safe_read_csv(
            SHAP_GLOBAL_PROCESSED_PATH
        )

        immunological_df = safe_read_csv(
            IMMUNOLOGICAL_SHAP_PATH
        )

        return {

            "global_original_features":
                (
                    original_df
                    .to_dict(
                        orient="records"
                    )
                    if original_df is not None
                    else []
                ),

            "global_processed_features":
                (
                    processed_df
                    .to_dict(
                        orient="records"
                    )
                    if processed_df is not None
                    else []
                ),

            "immunological_features":
                (
                    immunological_df
                    .to_dict(
                        orient="records"
                    )
                    if immunological_df is not None
                    else []
                )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Global XAI data failed: "
                f"{str(exc)}"
            )
        )


# ============================================================
# OBJECTIVE 2 LIME RESULTS
# ============================================================

@app.get("/xai/lime")
def xai_lime():

    try:

        case_df = safe_read_csv(
            LIME_CASE_SUMMARY_PATH
        )

        high_pred_df = safe_read_csv(
            LIME_HIGH_PRED_PATH
        )

        high_true_df = safe_read_csv(
            LIME_HIGH_TRUE_PATH
        )

        return {

            "case_summary":
                (
                    case_df
                    .to_dict(
                        orient="records"
                    )
                    if case_df is not None
                    else []
                ),

            "highest_predicted_incompatibility":
                (
                    high_pred_df
                    .to_dict(
                        orient="records"
                    )
                    if high_pred_df is not None
                    else []
                ),

            "highest_probability_among_true":
                (
                    high_true_df
                    .to_dict(
                        orient="records"
                    )
                    if high_true_df is not None
                    else []
                )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "LIME XAI data failed: "
                f"{str(exc)}"
            )
        )


# ============================================================
# OBJECTIVE 2 LOCAL SHAP RESULTS
# ============================================================

@app.get("/xai/local")
def xai_local():

    try:

        case_df = safe_read_csv(
            LOCAL_CASE_SUMMARY_PATH
        )

        high_pred_df = safe_read_csv(
            LOCAL_SHAP_HIGH_PRED_PATH
        )

        high_true_df = safe_read_csv(
            LOCAL_SHAP_HIGH_TRUE_PATH
        )

        return {

            "case_summary":
                (
                    case_df
                    .to_dict(
                        orient="records"
                    )
                    if case_df is not None
                    else []
                ),

            "highest_predicted_incompatibility":
                (
                    high_pred_df
                    .to_dict(
                        orient="records"
                    )
                    if high_pred_df is not None
                    else []
                ),

            "highest_probability_among_true":
                (
                    high_true_df
                    .to_dict(
                        orient="records"
                    )
                    if high_true_df is not None
                    else []
                )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Local SHAP data failed: "
                f"{str(exc)}"
            )
        )


# ============================================================
# OBJECTIVE 2 IMMUNOLOGICAL ANALYSIS
# ============================================================

@app.get("/xai/immunological")
def xai_immunological():

    try:

        shap_df = safe_read_csv(
            IMMUNOLOGICAL_SHAP_PATH
        )

        summary_df = safe_read_csv(
            IMMUNOLOGICAL_SUMMARY_PATH
        )

        return {

            "feature_shap":
                (
                    shap_df
                    .to_dict(
                        orient="records"
                    )
                    if shap_df is not None
                    else []
                ),

            "summary":
                (
                    summary_df
                    .to_dict(
                        orient="records"
                    )
                    if summary_df is not None
                    else []
                )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Immunological XAI data failed: "
                f"{str(exc)}"
            )
        )


# ============================================================
# OBJECTIVE 2 TEST PREDICTIONS
# ============================================================

@app.get("/xai/test-predictions")
def xai_test_predictions():

    try:

        df = safe_read_csv(
            OBJECTIVE_2_PREDICTIONS_PATH
        )

        if df is None:

            return {
                "available": False,
                "data": []
            }

        return {

            "available":
                True,

            "rows":
                len(df),

            "data":
                df.to_dict(
                    orient="records"
                )
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Objective 2 predictions failed: "
                f"{str(exc)}"
            )
        )


# ============================================================
# OBJECTIVE 2 METADATA
# ============================================================

@app.get("/xai/metadata")
def xai_metadata():

    return {

        "available":
            os.path.isfile(
                OBJECTIVE_2_METADATA_PATH
            ),

        "metadata":
            objective_2_metadata
    }


# ============================================================
# RUN SERVER
#
# Start with:
#
# uvicorn main:app --reload
#
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )