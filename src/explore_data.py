import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# AI PAYMENT RISK MANAGER
# DATA EXPLORATION AND ANALYTICS
# ============================================================


def load_data():
    """
    Load the raw transaction dataset.
    """

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    file_path = os.path.join(
        project_root,
        "data",
        "raw",
        "transactions.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "transactions.csv not found. "
            "Please run data_generator.py first."
        )

    return pd.read_csv(file_path)


def main():

    df = load_data()

    print()
    print("=" * 70)
    print("AI PAYMENT RISK MANAGER - DATA EXPLORATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("\n1. DATASET SIZE")
    print("-" * 70)

    print(
        f"Number of transactions : {len(df):,}"
    )

    print(
        f"Number of columns      : {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Fraud statistics
    # --------------------------------------------------------

    fraud_count = int(
        df["fraud_label"].sum()
    )

    genuine_count = int(
        (df["fraud_label"] == 0).sum()
    )

    fraud_percentage = (
        fraud_count / len(df)
    ) * 100

    print("\n2. FRAUD STATISTICS")
    print("-" * 70)

    print(
        f"Fraud transactions   : {fraud_count:,}"
    )

    print(
        f"Genuine transactions : {genuine_count:,}"
    )

    print(
        f"Fraud percentage     : {fraud_percentage:.2f}%"
    )

    # --------------------------------------------------------
    # Transaction amount
    # --------------------------------------------------------

    print("\n3. TRANSACTION AMOUNT")
    print("-" * 70)

    print(
        f"Average amount : ₹{df['amount'].mean():,.2f}"
    )

    print(
        f"Maximum amount : ₹{df['amount'].max():,.2f}"
    )

    print(
        f"Minimum amount : ₹{df['amount'].min():,.2f}"
    )

    # --------------------------------------------------------
    # Payment method
    # --------------------------------------------------------

    print("\n4. PAYMENT METHOD DISTRIBUTION")
    print("-" * 70)

    payment_distribution = (
        df["payment_method"]
        .value_counts()
    )

    print(payment_distribution)

    # --------------------------------------------------------
    # Fraud by payment method
    # --------------------------------------------------------

    print("\n5. FRAUD BY PAYMENT METHOD")
    print("-" * 70)

    fraud_by_payment = (
        df.groupby("payment_method")["fraud_label"]
        .agg(
            total_transactions="count",
            fraud_transactions="sum"
        )
    )

    fraud_by_payment["fraud_rate_percent"] = (
        fraud_by_payment["fraud_transactions"]
        / fraud_by_payment["total_transactions"]
        * 100
    )

    print(
        fraud_by_payment.sort_values(
            "fraud_rate_percent",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Fraud by location
    # --------------------------------------------------------

    print("\n6. FRAUD BY LOCATION")
    print("-" * 70)

    fraud_by_location = (
        df.groupby("location")["fraud_label"]
        .agg(
            total_transactions="count",
            fraud_transactions="sum"
        )
    )

    fraud_by_location["fraud_rate_percent"] = (
        fraud_by_location["fraud_transactions"]
        / fraud_by_location["total_transactions"]
        * 100
    )

    print(
        fraud_by_location.sort_values(
            "fraud_rate_percent",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Fraud by device
    # --------------------------------------------------------

    print("\n7. FRAUD BY DEVICE TYPE")
    print("-" * 70)

    fraud_by_device = (
        df.groupby("device_type")["fraud_label"]
        .agg(
            total_transactions="count",
            fraud_transactions="sum"
        )
    )

    fraud_by_device["fraud_rate_percent"] = (
        fraud_by_device["fraud_transactions"]
        / fraud_by_device["total_transactions"]
        * 100
    )

    print(
        fraud_by_device.sort_values(
            "fraud_rate_percent",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Risk signals
    # --------------------------------------------------------

    print("\n8. RISK SIGNAL ANALYSIS")
    print("-" * 70)

    print(
        "Average amount - Genuine : "
        f"₹{df[df['fraud_label'] == 0]['amount'].mean():,.2f}"
    )

    print(
        "Average amount - Fraud   : "
        f"₹{df[df['fraud_label'] == 1]['amount'].mean():,.2f}"
    )

    print(
        "New device rate - Fraud  : "
        f"{df[df['fraud_label'] == 1]['is_new_device'].mean() * 100:.2f}%"
    )

    print(
        "Location mismatch - Fraud: "
        f"{df[df['fraud_label'] == 1]['location_mismatch'].mean() * 100:.2f}%"
    )

    print(
        "Avg failed transactions - Fraud: "
        f"{df[df['fraud_label'] == 1]['previous_failed_transactions'].mean():.2f}"
    )

    print(
        "Avg transactions/24h - Fraud: "
        f"{df[df['fraud_label'] == 1]['transactions_last_24h'].mean():.2f}"
    )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\n9. MISSING VALUES")
    print("-" * 70)

    missing_values = df.isnull().sum()

    print(missing_values)

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    print("\n10. DUPLICATE ROWS")
    print("-" * 70)

    print(
        f"Duplicate rows: {df.duplicated().sum()}"
    )

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    os.makedirs(
        os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            "..",
            "screenshots"
        ),
        exist_ok=True
    )

    screenshots_directory = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "screenshots"
        )
    )

    # --------------------------------------------------------
    # Chart 1: Fraud vs Genuine
    # --------------------------------------------------------

    fraud_counts = df["fraud_label"].value_counts()

    labels = [
        "Genuine",
        "Fraud"
    ]

    values = [
        fraud_counts.get(0, 0),
        fraud_counts.get(1, 0)
    ]

    plt.figure(figsize=(7, 5))

    plt.bar(
        labels,
        values
    )

    plt.title(
        "Genuine vs Fraud Transactions"
    )

    plt.ylabel(
        "Number of Transactions"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            screenshots_directory,
            "fraud_vs_genuine.png"
        )
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 2: Payment Method
    # --------------------------------------------------------

    payment_distribution.plot(
        kind="bar",
        figsize=(8, 5)
    )

    plt.title(
        "Transaction Distribution by Payment Method"
    )

    plt.xlabel(
        "Payment Method"
    )

    plt.ylabel(
        "Number of Transactions"
    )

    plt.xticks(
        rotation=30,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            screenshots_directory,
            "payment_method_distribution.png"
        )
    )

    plt.close()

    # --------------------------------------------------------
    # Chart 3: Transaction Amount Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["amount"],
        bins=50
    )

    plt.title(
        "Transaction Amount Distribution"
    )

    plt.xlabel(
        "Transaction Amount (₹)"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            screenshots_directory,
            "transaction_amount_distribution.png"
        )
    )

    plt.close()

    print()
    print("=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nCharts saved inside the screenshots folder."
    )

    print()


if __name__ == "__main__":
    main()