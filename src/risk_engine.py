import os
import joblib
import pandas as pd


# ============================================================
# AI PAYMENT RISK MANAGER
# RISK SCORING ENGINE
# ============================================================


class RiskEngine:
    """
    Explainable risk scoring engine.

    Combines:
    1. ML fraud probability
    2. Explainable transaction risk signals

    This is a prototype scoring mechanism and
    not a production fraud decision system.
    """

    def __init__(self, model_path):
        """
        Load the trained ML pipeline.
        """

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        self.model = joblib.load(model_path)

    # ========================================================
    # ML PREDICTION
    # ========================================================

    def get_model_prediction(self, transaction):
        """
        Get fraud probability and model prediction.
        """

        transaction_df = pd.DataFrame(
            [transaction]
        )

        fraud_probability = (
            self.model.predict_proba(
                transaction_df
            )[0][1]
        )

        prediction = self.model.predict(
            transaction_df
        )[0]

        return (
            float(fraud_probability),
            int(prediction)
        )

    # ========================================================
    # EXPLAINABLE RISK FACTORS
    # ========================================================

    def identify_risk_factors(self, transaction):
        """
        Identify human-readable risk signals.
        """

        factors = []

        amount = float(
            transaction["amount"]
        )

        customer_average = float(
            transaction[
                "customer_avg_transaction_amount"
            ]
        )

        new_device = int(
            transaction["is_new_device"]
        )

        location_mismatch = int(
            transaction["location_mismatch"]
        )

        failed_transactions = int(
            transaction[
                "previous_failed_transactions"
            ]
        )

        transactions_24h = int(
            transaction[
                "transactions_last_24h"
            ]
        )

        account_age = int(
            transaction["account_age_days"]
        )

        transaction_hour = int(
            transaction["transaction_hour"]
        )

        # ----------------------------------------------------
        # Amount analysis
        # ----------------------------------------------------

        amount_ratio = (
            amount / max(
                customer_average,
                1
            )
        )

        if amount > 100000:
            factors.append(
                "Extremely high transaction amount"
            )

        elif amount > 50000:
            factors.append(
                "Very high transaction amount"
            )

        elif amount_ratio > 4:
            factors.append(
                "Transaction amount is unusually high "
                "compared with customer history"
            )

        elif amount_ratio > 2.5:
            factors.append(
                "Transaction amount is significantly "
                "above customer average"
            )

        # ----------------------------------------------------
        # New device
        # ----------------------------------------------------

        if new_device == 1:
            factors.append(
                "New device detected"
            )

        # ----------------------------------------------------
        # Location mismatch
        # ----------------------------------------------------

        if location_mismatch == 1:
            factors.append(
                "Location mismatch detected"
            )

        # ----------------------------------------------------
        # Failed transactions
        # ----------------------------------------------------

        if failed_transactions >= 4:
            factors.append(
                "Multiple previous failed transactions"
            )

        elif failed_transactions >= 2:
            factors.append(
                "Previous failed transactions detected"
            )

        # ----------------------------------------------------
        # Transaction frequency
        # ----------------------------------------------------

        if transactions_24h >= 10:
            factors.append(
                "Very high transaction frequency "
                "in the last 24 hours"
            )

        elif transactions_24h >= 6:
            factors.append(
                "High transaction frequency "
                "in the last 24 hours"
            )

        # ----------------------------------------------------
        # Account age
        # ----------------------------------------------------

        if account_age < 30:
            factors.append(
                "Very new customer account"
            )

        elif account_age < 90:
            factors.append(
                "Relatively new customer account"
            )

        # ----------------------------------------------------
        # Unusual transaction hour
        # ----------------------------------------------------

        if 0 <= transaction_hour <= 4:
            factors.append(
                "Transaction occurred during "
                "an unusual late-night period"
            )

        # ----------------------------------------------------
        # Default
        # ----------------------------------------------------

        if not factors:
            factors.append(
                "No major risk signals detected"
            )

        return factors

    # ========================================================
    # EXPLAINABLE SIGNAL SCORE
    # ========================================================

    def calculate_signal_score(self, transaction):
        """
        Calculate an explainable risk signal score from 0-100.
        """

        score = 0.0

        amount = float(
            transaction["amount"]
        )

        customer_average = float(
            transaction[
                "customer_avg_transaction_amount"
            ]
        )

        new_device = int(
            transaction["is_new_device"]
        )

        location_mismatch = int(
            transaction["location_mismatch"]
        )

        failed_transactions = int(
            transaction[
                "previous_failed_transactions"
            ]
        )

        transactions_24h = int(
            transaction[
                "transactions_last_24h"
            ]
        )

        account_age = int(
            transaction["account_age_days"]
        )

        transaction_hour = int(
            transaction["transaction_hour"]
        )

        # ----------------------------------------------------
        # Amount
        # ----------------------------------------------------

        amount_ratio = (
            amount / max(
                customer_average,
                1
            )
        )

        if amount > 100000:
            score += 25

        elif amount > 50000:
            score += 18

        elif amount_ratio > 4:
            score += 18

        elif amount_ratio > 2.5:
            score += 10

        # ----------------------------------------------------
        # New device
        # ----------------------------------------------------

        if new_device == 1:
            score += 15

        # ----------------------------------------------------
        # Location mismatch
        # ----------------------------------------------------

        if location_mismatch == 1:
            score += 20

        # ----------------------------------------------------
        # Failed transactions
        # ----------------------------------------------------

        if failed_transactions >= 4:
            score += 15

        elif failed_transactions >= 2:
            score += 8

        # ----------------------------------------------------
        # Transaction frequency
        # ----------------------------------------------------

        if transactions_24h >= 10:
            score += 15

        elif transactions_24h >= 6:
            score += 8

        # ----------------------------------------------------
        # Account age
        # ----------------------------------------------------

        if account_age < 30:
            score += 8

        elif account_age < 90:
            score += 4

        # ----------------------------------------------------
        # Unusual hour
        # ----------------------------------------------------

        if 0 <= transaction_hour <= 4:
            score += 7

        return min(
            round(score, 2),
            100
        )

    # ========================================================
    # FINAL RISK SCORE
    # ========================================================

    def calculate_risk_score(
        self,
        fraud_probability,
        signal_score
    ):
        """
        Combine ML probability and explainable
        signal score into final 0-100 risk score.
        """

        ml_score = (
            fraud_probability * 100
        )

        # 60% ML + 40% explainable signals
        final_score = (
            ml_score * 0.60
            + signal_score * 0.40
        )

        return min(
            max(
                round(final_score, 2),
                0
            ),
            100
        )

    # ========================================================
    # RISK LEVEL
    # ========================================================

    def get_risk_level(self, risk_score):
        """
        Convert score into LOW / MEDIUM / HIGH.
        """

        if risk_score <= 30:
            return "LOW RISK"

        elif risk_score <= 70:
            return "MEDIUM RISK"

        return "HIGH RISK"

    # ========================================================
    # RECOMMENDED ACTION
    # ========================================================

    def get_recommended_action(
        self,
        risk_level
    ):
        """
        Generate recommended action.
        """

        if risk_level == "LOW RISK":

            return (
                "Allow transaction and continue "
                "normal monitoring."
            )

        elif risk_level == "MEDIUM RISK":

            return (
                "Apply additional verification "
                "and enhanced monitoring."
            )

        return (
            "Require additional authentication "
            "and consider manual review before processing."
        )

    # ========================================================
    # COMPLETE ANALYSIS
    # ========================================================

    def analyze_transaction(
        self,
        transaction
    ):
        """
        Perform complete transaction risk analysis.
        """

        (
            fraud_probability,
            model_prediction
        ) = self.get_model_prediction(
            transaction
        )

        signal_score = (
            self.calculate_signal_score(
                transaction
            )
        )

        risk_score = (
            self.calculate_risk_score(
                fraud_probability,
                signal_score
            )
        )

        risk_level = (
            self.get_risk_level(
                risk_score
            )
        )

        risk_factors = (
            self.identify_risk_factors(
                transaction
            )
        )

        recommended_action = (
            self.get_recommended_action(
                risk_level
            )
        )

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "fraud_probability": round(
                fraud_probability * 100,
                2
            ),
            "model_prediction": model_prediction,
            "risk_factors": risk_factors,
            "recommended_action":
                recommended_action
        }


# ============================================================
# TEST THE RISK ENGINE
# ============================================================


def main():

    project_root = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )

    model_path = os.path.join(
        project_root,
        "model",
        "fraud_model.pkl"
    )

    print()
    print("=" * 70)
    print("AI PAYMENT RISK MANAGER")
    print("RISK ENGINE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Example transaction
    # --------------------------------------------------------

    test_transaction = {
        "amount": 85000,
        "transaction_hour": 2,
        "payment_method": "UPI",
        "location": "Delhi",
        "device_type": "Android",
        "is_new_device": 1,
        "location_mismatch": 1,
        "previous_failed_transactions": 3,
        "transactions_last_24h": 9,
        "account_age_days": 45,
        "customer_avg_transaction_amount": 2500,
        "is_weekend": 1
    }

    # --------------------------------------------------------
    # Create engine
    # --------------------------------------------------------

    engine = RiskEngine(
        model_path
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    result = engine.analyze_transaction(
        test_transaction
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print()
    print(
        f"Risk Score       : "
        f"{result['risk_score']}/100"
    )

    print(
        f"Risk Level       : "
        f"{result['risk_level']}"
    )

    print(
        f"Fraud Probability: "
        f"{result['fraud_probability']}%"
    )

    print("\nRisk Factors:")

    for factor in result["risk_factors"]:
        print(
            f"  • {factor}"
        )

    print(
        "\nRecommended Action:"
    )

    print(
        f"  {result['recommended_action']}"
    )

    print()
    print("=" * 70)
    print("RISK ENGINE TEST COMPLETED")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()