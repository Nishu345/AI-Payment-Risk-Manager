import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# AI PAYMENT RISK MANAGER
# MACHINE LEARNING MODEL TRAINING
# ============================================================


RANDOM_STATE = 42


def load_data():
    """
    Load the processed/raw transaction data.
    """

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    data_file = os.path.join(
        project_root,
        "data",
        "raw",
        "transactions.csv"
    )

    if not os.path.exists(data_file):
        raise FileNotFoundError(
            "transactions.csv not found. "
            "Please generate the dataset first."
        )

    return pd.read_csv(data_file)


def create_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Create preprocessing pipeline.
    """

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


def evaluate_model(
    model_name,
    model,
    X_test,
    y_test
):
    """
    Evaluate a trained classification model.
    """

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print()
    print("=" * 70)
    print(f"{model_name} EVALUATION")
    print("=" * 70)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Genuine",
                "Fraud"
            ],
            zero_division=0
        )
    )

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }


def main():

    print()
    print("=" * 70)
    print("AI PAYMENT RISK MANAGER")
    print("MACHINE LEARNING MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = load_data()

    print(
        f"\nDataset loaded successfully."
    )

    print(
        f"Total transactions: {len(df):,}"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = [
        "amount",
        "transaction_hour",
        "payment_method",
        "location",
        "device_type",
        "is_new_device",
        "location_mismatch",
        "previous_failed_transactions",
        "transactions_last_24h",
        "account_age_days",
        "customer_avg_transaction_amount",
        "is_weekend"
    ]

    target_column = "fraud_label"

    X = df[feature_columns]

    y = df[target_column]

    # --------------------------------------------------------
    # Feature types
    # --------------------------------------------------------

    numerical_features = [
        "amount",
        "transaction_hour",
        "is_new_device",
        "location_mismatch",
        "previous_failed_transactions",
        "transactions_last_24h",
        "account_age_days",
        "customer_avg_transaction_amount",
        "is_weekend"
    ]

    categorical_features = [
        "payment_method",
        "location",
        "device_type"
    ]

    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(
        f"\nTraining samples: {len(X_train):,}"
    )

    print(
        f"Testing samples : {len(X_test):,}"
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    # ========================================================
    # BASELINE MODEL
    # ========================================================

    baseline_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    print(
        "\nTraining baseline Logistic Regression..."
    )

    baseline_model.fit(
        X_train,
        y_train
    )

    baseline_results = evaluate_model(
        "LOGISTIC REGRESSION BASELINE",
        baseline_model,
        X_test,
        y_test
    )

    # ========================================================
    # RANDOM FOREST MODEL
    # ========================================================

    rf_preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    random_forest_model = Pipeline(
        steps=[
            (
                "preprocessor",
                rf_preprocessor
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )

    print(
        "\nTraining Random Forest..."
    )

    random_forest_model.fit(
        X_train,
        y_train
    )

    rf_results = evaluate_model(
        "RANDOM FOREST",
        random_forest_model,
        X_test,
        y_test
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results = pd.DataFrame(
        [
            baseline_results,
            rf_results
        ]
    )

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select Random Forest
    # --------------------------------------------------------

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    model_directory = os.path.join(
        project_root,
        "model"
    )

    os.makedirs(
        model_directory,
        exist_ok=True
    )

    model_file = os.path.join(
        model_directory,
        "fraud_model.pkl"
    )

    joblib.dump(
        random_forest_model,
        model_file
    )

    # --------------------------------------------------------
    # Save model evaluation
    # --------------------------------------------------------

    evaluation_file = os.path.join(
        model_directory,
        "model_evaluation.csv"
    )

    results.to_csv(
        evaluation_file,
        index=False
    )

    print()
    print("=" * 70)
    print("MODEL SAVING")
    print("=" * 70)

    print(
        f"\nRandom Forest saved to:"
    )

    print(
        model_file
    )

    print(
        f"\nEvaluation results saved to:"
    )

    print(
        evaluation_file
    )

    print()
    print("=" * 70)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()