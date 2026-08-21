import os
import numpy as np
import pandas as pd


# ============================================================
# AI PAYMENT RISK MANAGER
# Synthetic Payment Transaction Dataset Generator
# ============================================================

RANDOM_SEED = 42
NUM_TRANSACTIONS = 10000

np.random.seed(RANDOM_SEED)


def generate_dataset(num_transactions=NUM_TRANSACTIONS):
    """
    Generate a realistic synthetic digital payment dataset.
    """

    # --------------------------------------------------------
    # Transaction and customer IDs
    # --------------------------------------------------------

    transaction_ids = [
        f"TXN{str(i).zfill(7)}"
        for i in range(1, num_transactions + 1)
    ]

    customer_ids = [
        f"CUST{str(np.random.randint(1, 2501)).zfill(5)}"
        for _ in range(num_transactions)
    ]

    # --------------------------------------------------------
    # Transaction amount
    # --------------------------------------------------------

    amounts = np.random.lognormal(
        mean=np.log(1800),
        sigma=1.0,
        size=num_transactions
    )

    amounts = np.clip(amounts, 50, 150000)
    amounts = np.round(amounts, 2)

    # --------------------------------------------------------
    # Transaction hour
    # --------------------------------------------------------

    # 24 probabilities that sum EXACTLY to 1.0
    hour_probabilities = np.array([
        0.025,
        0.018,
        0.015,
        0.012,
        0.012,
        0.015,
        0.025,
        0.045,
        0.065,
        0.065,
        0.060,
        0.055,
        0.055,
        0.055,
        0.055,
        0.060,
        0.065,
        0.070,
        0.075,
        0.075,
        0.065,
        0.055,
        0.040,
        0.032
    ])

    # Normalize to guarantee total = 1.0
    hour_probabilities = (
        hour_probabilities / hour_probabilities.sum()
    )

    transaction_hours = np.random.choice(
        np.arange(24),
        size=num_transactions,
        p=hour_probabilities
    )

    # --------------------------------------------------------
    # Payment method
    # --------------------------------------------------------

    payment_methods = np.random.choice(
        [
            "UPI",
            "Credit Card",
            "Debit Card",
            "Wallet",
            "Net Banking"
        ],
        size=num_transactions,
        p=[
            0.48,
            0.18,
            0.20,
            0.08,
            0.06
        ]
    )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    locations = np.random.choice(
        [
            "Delhi",
            "Mumbai",
            "Bengaluru",
            "Hyderabad",
            "Chennai",
            "Pune",
            "Kolkata",
            "Noida",
            "Gurugram",
            "Jaipur"
        ],
        size=num_transactions
    )

    # --------------------------------------------------------
    # Device type
    # --------------------------------------------------------

    device_types = np.random.choice(
        [
            "Android",
            "iOS",
            "Web"
        ],
        size=num_transactions,
        p=[
            0.55,
            0.25,
            0.20
        ]
    )

    # --------------------------------------------------------
    # New device
    # --------------------------------------------------------

    is_new_device = np.random.choice(
        [0, 1],
        size=num_transactions,
        p=[
            0.82,
            0.18
        ]
    )

    # --------------------------------------------------------
    # Location mismatch
    # --------------------------------------------------------

    location_mismatch = np.random.choice(
        [0, 1],
        size=num_transactions,
        p=[
            0.90,
            0.10
        ]
    )

    # --------------------------------------------------------
    # Previous failed transactions
    # --------------------------------------------------------

    previous_failed_transactions = np.random.poisson(
        lam=0.7,
        size=num_transactions
    )

    previous_failed_transactions = np.clip(
        previous_failed_transactions,
        0,
        8
    )

    # --------------------------------------------------------
    # Transactions in last 24 hours
    # --------------------------------------------------------

    transactions_last_24h = np.random.poisson(
        lam=3.5,
        size=num_transactions
    )

    transactions_last_24h = np.clip(
        transactions_last_24h,
        1,
        25
    )

    # --------------------------------------------------------
    # Account age
    # --------------------------------------------------------

    account_age_days = np.random.exponential(
        scale=600,
        size=num_transactions
    )

    account_age_days = np.clip(
        account_age_days,
        10,
        3000
    ).astype(int)

    # --------------------------------------------------------
    # Customer average transaction amount
    # --------------------------------------------------------

    customer_avg_transaction_amount = np.random.lognormal(
        mean=np.log(1600),
        sigma=0.65,
        size=num_transactions
    )

    customer_avg_transaction_amount = np.clip(
        customer_avg_transaction_amount,
        100,
        50000
    )

    customer_avg_transaction_amount = np.round(
        customer_avg_transaction_amount,
        2
    )

    # --------------------------------------------------------
    # Weekend indicator
    # --------------------------------------------------------

    is_weekend = np.random.choice(
        [0, 1],
        size=num_transactions,
        p=[
            0.71,
            0.29
        ]
    )

    # ========================================================
    # RISK SIGNAL GENERATION
    # ========================================================

    risk_score = np.zeros(num_transactions)

    # --------------------------------------------------------
    # Transaction amount compared with customer average
    # --------------------------------------------------------

    amount_ratio = (
        amounts / customer_avg_transaction_amount
    )

    risk_score += np.where(
        amount_ratio > 4,
        2.0,
        np.where(
            amount_ratio > 2.5,
            0.8,
            0
        )
    )

    # --------------------------------------------------------
    # Very high transaction amount
    # --------------------------------------------------------

    risk_score += np.where(
        amounts > 50000,
        2.0,
        0
    )

    risk_score += np.where(
        amounts > 100000,
        1.0,
        0
    )

    # --------------------------------------------------------
    # New device risk
    # --------------------------------------------------------

    risk_score += (
        is_new_device * 1.2
    )

    # --------------------------------------------------------
    # Location mismatch risk
    # --------------------------------------------------------

    risk_score += (
        location_mismatch * 1.5
    )

    # --------------------------------------------------------
    # Failed transaction risk
    # --------------------------------------------------------

    risk_score += np.minimum(
        previous_failed_transactions * 0.45,
        2.5
    )

    # --------------------------------------------------------
    # High transaction frequency
    # --------------------------------------------------------

    risk_score += np.where(
        transactions_last_24h >= 10,
        1.8,
        np.where(
            transactions_last_24h >= 6,
            0.7,
            0
        )
    )

    # --------------------------------------------------------
    # New account risk
    # --------------------------------------------------------

    risk_score += np.where(
        account_age_days < 30,
        1.0,
        np.where(
            account_age_days < 90,
            0.4,
            0
        )
    )

    # --------------------------------------------------------
    # Unusual transaction hour
    # --------------------------------------------------------

    unusual_hour = (
        (transaction_hours >= 0)
        & (transaction_hours <= 4)
    )

    risk_score += (
        unusual_hour * 1.0
    )

    # --------------------------------------------------------
    # Weekend effect
    # --------------------------------------------------------

    risk_score += (
        is_weekend * 0.15
    )

    # --------------------------------------------------------
    # Random noise
    # --------------------------------------------------------

    risk_score += np.random.normal(
        loc=0,
        scale=1.2,
        size=num_transactions
    )

    # ========================================================
    # FRAUD PROBABILITY
    # ========================================================

    fraud_probability = 1 / (
        1 + np.exp(
            -(risk_score - 5.0)
        )
    )

    # Keep the dataset reasonably balanced
    fraud_probability = (
        fraud_probability * 0.22
        + 0.005
    )

    fraud_probability = np.clip(
        fraud_probability,
        0.001,
        0.85
    )

    # --------------------------------------------------------
    # Fraud label
    # 0 = Genuine
    # 1 = Suspicious/Fraud
    # --------------------------------------------------------

    fraud_label = np.random.binomial(
        1,
        fraud_probability
    )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    df = pd.DataFrame({
        "transaction_id": transaction_ids,
        "customer_id": customer_ids,
        "amount": amounts,
        "transaction_hour": transaction_hours,
        "payment_method": payment_methods,
        "location": locations,
        "device_type": device_types,
        "is_new_device": is_new_device,
        "location_mismatch": location_mismatch,
        "previous_failed_transactions":
            previous_failed_transactions,
        "transactions_last_24h":
            transactions_last_24h,
        "account_age_days":
            account_age_days,
        "customer_avg_transaction_amount":
            customer_avg_transaction_amount,
        "is_weekend":
            is_weekend,
        "fraud_label":
            fraud_label
    })

    return df


def main():
    """
    Generate and save the synthetic transaction dataset.
    """

    # Project root directory
    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    # Output directory
    output_directory = os.path.join(
        project_root,
        "data",
        "raw"
    )

    # Create directory if it does not exist
    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # Output CSV path
    output_file = os.path.join(
        output_directory,
        "transactions.csv"
    )

    # Generate dataset
    df = generate_dataset()

    # Save dataset
    df.to_csv(
        output_file,
        index=False
    )

    # ========================================================
    # DISPLAY SUMMARY
    # ========================================================

    fraud_count = int(
        df["fraud_label"].sum()
    )

    genuine_count = int(
        (df["fraud_label"] == 0).sum()
    )

    fraud_percentage = (
        fraud_count / len(df)
    ) * 100

    print()
    print("=" * 60)
    print("SYNTHETIC PAYMENT DATASET GENERATED")
    print("=" * 60)

    print(
        f"Total transactions     : {len(df)}"
    )

    print(
        f"Total columns          : {len(df.columns)}"
    )

    print(
        f"Fraud transactions     : {fraud_count}"
    )

    print(
        f"Genuine transactions   : {genuine_count}"
    )

    print(
        f"Fraud percentage       : "
        f"{fraud_percentage:.2f}%"
    )

    print(
        f"Dataset saved to       : {output_file}"
    )

    print("=" * 60)
    print()


if __name__ == "__main__":
    main()