import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


# ============================================================
# AI PAYMENT RISK MANAGER
# DATA PREPROCESSING
# ============================================================


def load_raw_data():
    """
    Load the raw transaction dataset.
    """

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    input_file = os.path.join(
        project_root,
        "data",
        "raw",
        "transactions.csv"
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            "transactions.csv was not found. "
            "Please generate the dataset first."
        )

    return pd.read_csv(input_file)


def create_preprocessor(
    numerical_features,
    categorical_features
):
    """
    Create preprocessing pipeline for numerical
    and categorical features.
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


def preprocess_dataset():
    """
    Prepare transaction data for machine learning.
    """

    df = load_raw_data()

    print()
    print("=" * 70)
    print("DATA PREPROCESSING")
    print("=" * 70)

    # --------------------------------------------------------
    # Remove duplicate rows
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print(
        f"\nDuplicate rows found: {duplicate_count}"
    )

    df = df.drop_duplicates()

    # --------------------------------------------------------
    # Define target
    # --------------------------------------------------------

    target_column = "fraud_label"

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
    # Create preprocessing pipeline
    # --------------------------------------------------------

    preprocessor = create_preprocessor(
        numerical_features,
        categorical_features
    )

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # Fit preprocessing only on training data
    # --------------------------------------------------------

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    # --------------------------------------------------------
    # Convert processed data into DataFrames
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index
    )

    # --------------------------------------------------------
    # Save processed complete dataset
    # --------------------------------------------------------

    processed_df = X.copy()

    processed_df[target_column] = y.values

    output_directory = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "processed"
    )

    output_directory = os.path.abspath(
        output_directory
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    output_file = os.path.join(
        output_directory,
        "processed_transactions.csv"
    )

    processed_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print(
        f"\nOriginal rows       : {len(df)}"
    )

    print(
        f"Training rows       : {len(X_train)}"
    )

    print(
        f"Testing rows        : {len(X_test)}"
    )

    print(
        f"Original features   : {len(feature_columns)}"
    )

    print(
        f"Processed features  : {X_train_processed.shape[1]}"
    )

    print(
        f"\nTraining shape      : "
        f"{X_train_processed.shape}"
    )

    print(
        f"Testing shape       : "
        f"{X_test_processed.shape}"
    )

    print(
        f"\nProcessed dataset saved to:\n"
        f"{output_file}"
    )

    print()
    print("=" * 70)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 70)
    print()

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    preprocess_dataset()