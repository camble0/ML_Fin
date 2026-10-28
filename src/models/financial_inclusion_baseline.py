import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "financial_inclusion_household.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "analytical_dataset_report.json"
MODEL_ARTIFACT_DIR = PROJECT_ROOT / "models" / "financial_inclusion_baseline"
TARGET_COLUMN = "hh_has_bank_or_momo_account"


def json_scalar(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def format_household_id(value) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return str(value)


def training_reference_profile(
    x_train: pd.DataFrame, categorical: list[str], numeric: list[str]
) -> dict:
    reference = {}
    for feature in numeric:
        reference[feature] = json_scalar(x_train[feature].median())
    for feature in categorical:
        modes = x_train[feature].mode(dropna=True)
        reference[feature] = json_scalar(modes.iloc[0]) if not modes.empty else None
    return reference


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    has_both_classes = y_test.nunique() == 2
    return {
        "precision_positive_class": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_positive_class": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_positive_class": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if has_both_classes else None,
        "pr_auc_average_precision": float(average_precision_score(y_test, probabilities)) if has_both_classes else None,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "test_households": int(len(y_test)),
        "test_class_counts": {
            str(key): int(value)
            for key, value in y_test.value_counts().sort_index().items()
        },
    }


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    target = frame[TARGET_COLUMN]
    labeled = target.notna()
    eligible = frame.loc[labeled].reset_index(drop=True)
    y = eligible[TARGET_COLUMN].astype(int)
    groups = eligible["clust"].fillna("missing-cluster").astype(str)
    features = report["model_predictor_columns"]
    categorical = report["categorical_predictors"]
    numeric = report["numeric_predictors"]
    x = eligible[features]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, test_indices = next(splitter.split(x, y, groups))
    x_train, x_test = x.iloc[train_indices], x.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    train_groups = set(groups.iloc[train_indices])
    test_groups = set(groups.iloc[test_indices])
    if train_groups & test_groups:
        raise ValueError("Cluster groups overlap between train and test sets.")

    MODEL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    reference_profile = training_reference_profile(x_train, categorical, numeric)
    reference_profile_path = MODEL_ARTIFACT_DIR / "training_reference_profile.json"
    reference_profile_path.write_text(
        json.dumps(
            {
                "method": "Training-fold median for numeric features and training-fold mode for categorical features.",
                "values": reference_profile,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    estimators = {
        "majority_class_baseline": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=2,
            random_state=42,
        ),
    }
    model_results = {}
    model_artifacts = {}
    for name, estimator in estimators.items():
        model = Pipeline(
            [
                ("preprocessor", make_preprocessor(categorical, numeric)),
                ("classifier", estimator),
            ]
        )
        model.fit(x_train, y_train)
        model_results[name] = evaluate_model(model, x_test, y_test)
        if name != "majority_class_baseline":
            artifact_path = MODEL_ARTIFACT_DIR / f"{name}.joblib"
            joblib.dump(model, artifact_path)
            model_artifacts[name] = str(artifact_path.relative_to(PROJECT_ROOT))

    demo_household_id = format_household_id(
        eligible.iloc[int(test_indices[0])]["hhid"]
    )

    evaluation = {
        "status": "completed",
        "target": TARGET_COLUMN,
        "eligible_labeled_households": int(len(eligible)),
        "unresolved_target_households_excluded_from_fit": int((~labeled).sum()),
        "split": {
            "method": "First fold of StratifiedGroupKFold (5 folds, shuffle=True, random_state=42).",
            "group_variable": "clust",
            "grouping_assumption": "clust is treated as a survey-cluster candidate; confirm its PSU role in survey documentation.",
            "group_overlap": False,
            "train_households": int(len(train_indices)),
            "test_households": int(len(test_indices)),
            "train_clusters": int(len(train_groups)),
            "test_clusters": int(len(test_groups)),
            "train_class_counts": {
                str(key): int(value)
                for key, value in y_train.value_counts().sort_index().items()
            },
        },
        "class_imbalance_handling": "class_weight='balanced' for logistic regression; balanced_subsample for random forest. Actual household class distribution is in the dataset target section.",
        "survey_weights": "weight and pop_wt were preserved but not passed to model fitting; see the documented survey-weight decision.",
        "threshold": 0.5,
        "metrics": model_results,
        "model_artifacts": model_artifacts,
        "training_reference_profile": str(reference_profile_path.relative_to(PROJECT_ROOT)),
        "interpretation_demo_household_id": demo_household_id,
        "interpretation_limit": "Predictive association only; cross-sectional survey data and this split do not establish causality.",
    }
    report["baseline_model_evaluation"] = evaluation
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
