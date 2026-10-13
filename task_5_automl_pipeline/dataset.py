import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

DATASET_METADATA: Dict[str, Dict[str, Any]] = {
    "customer_churn": {
        "title": "Telco Customer Churn",
        "description": "Predict whether a telecommunications subscriber will cancel service based on tenure, monthly spend, contracts, and service subscriptions.",
        "target_col": "churn",
        "positive_label": "Churn (1)",
        "negative_label": "Retained (0)",
        "domain": "Telecommunications & Customer Retention"
    },
    "credit_risk": {
        "title": "Credit Loan Default Risk",
        "description": "Forecast the likelihood of credit delinquency or default based on applicant credit score, debt-to-income ratio, and loan terms.",
        "target_col": "default_risk",
        "positive_label": "High Default Risk (1)",
        "negative_label": "Low Risk / Approved (0)",
        "domain": "FinTech & Banking Risk"
    },
    "employee_attrition": {
        "title": "Enterprise Employee Attrition",
        "description": "Analyze workplace satisfaction, compensation, overtime, and tenure to identify employees at high risk of resignation.",
        "target_col": "attrition",
        "positive_label": "Resigned (1)",
        "negative_label": "Retained (0)",
        "domain": "Human Resources & Talent Analytics"
    }
}


def load_customer_churn_dataset(n_samples: int = 1200, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates a realistic benchmark Customer Churn tabular dataset with
    interdependent numerical and categorical features.
    """
    rng = np.random.RandomState(random_state)

    tenure = rng.randint(1, 73, size=n_samples)
    monthly_charges = np.round(rng.uniform(18.5, 118.5, size=n_samples), 2)
    total_charges = np.round(tenure * monthly_charges * rng.uniform(0.95, 1.05, size=n_samples), 2)
    
    contract_choices = np.array(["Month-to-month", "One year", "Two year"])
    contract_probs = [0.55, 0.25, 0.20]
    contract_type = rng.choice(contract_choices, size=n_samples, p=contract_probs)

    internet_choices = np.array(["DSL", "Fiber optic", "No"])
    internet_service = rng.choice(internet_choices, size=n_samples, p=[0.34, 0.44, 0.22])

    online_security = []
    tech_support = []
    for internet in internet_service:
        if internet == "No":
            online_security.append("No internet service")
            tech_support.append("No internet service")
        else:
            online_security.append(rng.choice(["Yes", "No"], p=[0.38, 0.62]))
            tech_support.append(rng.choice(["Yes", "No"], p=[0.37, 0.63]))

    payment_choices = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    payment_method = rng.choice(payment_choices, size=n_samples, p=[0.35, 0.20, 0.23, 0.22])

    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])

    # Realistic logistic churn propensity formula
    z = (
        -0.8
        - (tenure / 18.0)
        + (monthly_charges / 45.0)
        + (1.2 * (contract_type == "Month-to-month"))
        - (1.0 * (contract_type == "Two year"))
        + (0.6 * (internet_service == "Fiber optic"))
        + (0.5 * (payment_method == "Electronic check"))
        - (0.6 * (np.array(tech_support) == "Yes"))
        + (0.3 * (np.array(online_security) == "No"))
        + rng.normal(0, 0.45, size=n_samples)
    )
    probs = 1.0 / (1.0 + np.exp(-z))
    churn = (probs > 0.48).astype(int)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "senior_citizen": senior_citizen,
        "churn": churn
    })

    X = df.drop(columns=["churn"])
    y = df["churn"]
    return X, y


def load_credit_risk_dataset(n_samples: int = 1200, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates a realistic benchmark Credit Risk tabular dataset.
    """
    rng = np.random.RandomState(random_state)

    applicant_age = rng.randint(21, 68, size=n_samples)
    annual_income = np.round(rng.lognormal(mean=10.8, sigma=0.55, size=n_samples), -2)
    annual_income = np.clip(annual_income, 22000, 240000)

    credit_score = rng.normal(loc=675, scale=75, size=n_samples).astype(int)
    credit_score = np.clip(credit_score, 450, 850)

    loan_amount = np.round(rng.uniform(3000, 45000, size=n_samples), -2)
    debt_to_income = np.round(rng.beta(2, 5, size=n_samples) * 0.65, 3)
    employment_years = np.clip(rng.poisson(lam=6, size=n_samples), 0, 35)

    home_ownership = rng.choice(["RENT", "MORTGAGE", "OWN"], size=n_samples, p=[0.48, 0.42, 0.10])
    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "small_business"],
        size=n_samples,
        p=[0.50, 0.25, 0.15, 0.10]
    )

    # Risk latent score
    z = (
        1.2
        - (credit_score - 600) / 70.0
        + (debt_to_income * 4.5)
        + (loan_amount / annual_income * 2.2)
        - (employment_years * 0.08)
        + (0.4 * (home_ownership == "RENT"))
        + rng.normal(0, 0.40, size=n_samples)
    )
    probs = 1.0 / (1.0 + np.exp(-z))
    default_risk = (probs > 0.42).astype(int)

    df = pd.DataFrame({
        "applicant_age": applicant_age,
        "annual_income": annual_income,
        "credit_score": credit_score,
        "loan_amount": loan_amount,
        "debt_to_income_ratio": debt_to_income,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "default_risk": default_risk
    })

    X = df.drop(columns=["default_risk"])
    y = df["default_risk"]
    return X, y


def load_employee_attrition_dataset(n_samples: int = 1200, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates a realistic benchmark Employee Attrition tabular dataset.
    """
    rng = np.random.RandomState(random_state)

    age = rng.randint(21, 60, size=n_samples)
    total_working_years = np.clip(age - rng.randint(20, 24, size=n_samples), 1, 40)
    years_at_company = np.array([rng.randint(0, max(1, twy + 1)) for twy in total_working_years])
    
    monthly_income = np.round(2800 + (total_working_years * 420) + rng.normal(0, 900, size=n_samples), -1)
    monthly_income = np.clip(monthly_income, 2500, 22000)

    job_satisfaction = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.15, 0.25, 0.35, 0.25])
    work_life_balance = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.10, 0.30, 0.45, 0.15])
    overtime = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])
    department = rng.choice(["Sales", "Research & Development", "Human Resources"], size=n_samples, p=[0.35, 0.58, 0.07])
    business_travel = rng.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], size=n_samples, p=[0.70, 0.20, 0.10])

    # Propensity score
    z = (
        -0.4
        + (1.1 * (overtime == "Yes"))
        - (job_satisfaction * 0.45)
        - (work_life_balance * 0.40)
        - (years_at_company * 0.06)
        + (0.5 * (business_travel == "Travel_Frequently"))
        - (monthly_income / 12000.0)
        + rng.normal(0, 0.35, size=n_samples)
    )
    probs = 1.0 / (1.0 + np.exp(-z))
    attrition = (probs > 0.45).astype(int)

    df = pd.DataFrame({
        "age": age,
        "monthly_income": monthly_income,
        "total_working_years": total_working_years,
        "years_at_company": years_at_company,
        "job_satisfaction": job_satisfaction,
        "work_life_balance": work_life_balance,
        "overtime": overtime,
        "department": department,
        "business_travel": business_travel,
        "attrition": attrition
    })

    X = df.drop(columns=["attrition"])
    y = df["attrition"]
    return X, y


def load_dataset_by_name(name: str = "customer_churn", n_samples: int = 1200, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Helper function to load one of the built-in benchmark datasets by key.
    """
    clean_name = name.lower().strip()
    if clean_name in ["churn", "customer_churn"]:
        X, y = load_customer_churn_dataset(n_samples=n_samples, random_state=random_state)
        meta = DATASET_METADATA["customer_churn"]
    elif clean_name in ["credit", "credit_risk"]:
        X, y = load_credit_risk_dataset(n_samples=n_samples, random_state=random_state)
        meta = DATASET_METADATA["credit_risk"]
    elif clean_name in ["attrition", "employee_attrition"]:
        X, y = load_employee_attrition_dataset(n_samples=n_samples, random_state=random_state)
        meta = DATASET_METADATA["employee_attrition"]
    else:
        # Default fallback
        X, y = load_customer_churn_dataset(n_samples=n_samples, random_state=random_state)
        meta = DATASET_METADATA["customer_churn"]

    return X, y, meta


def load_custom_csv(file_or_path: Any, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Parses a user-uploaded CSV file or path, validating features and target column.
    """
    if isinstance(file_or_path, str):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)

    if target_col is None or target_col not in df.columns:
        # Select last column by default
        target_col = df.columns[-1]

    # Clean target
    y_raw = df[target_col]
    X = df.drop(columns=[target_col])

    # Convert binary strings or categories to 0/1 if needed
    if y_raw.dtype == object or str(y_raw.dtype).startswith("category"):
        unique_vals = list(y_raw.dropna().unique())
        if len(unique_vals) == 2:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            y = y_raw.map(mapping)
        else:
            y = pd.factorize(y_raw)[0]
            y = pd.Series(y, index=df.index)
    else:
        y = y_raw

    meta = {
        "title": "Custom Uploaded Dataset",
        "description": f"Custom dataset with {df.shape[0]} rows and {df.shape[1]} columns. Target: '{target_col}'",
        "target_col": str(target_col),
        "positive_label": "Class 1",
        "negative_label": "Class 0",
        "domain": "User Data"
    }

    return X, y, meta


if __name__ == "__main__":
    print("Testing built-in benchmark datasets...")
    for key in DATASET_METADATA:
        X, y, meta = load_dataset_by_name(key, n_samples=300)
        print(f"[{meta['title']}] -> X shape: {X.shape}, y shape: {y.shape}, Balance: {y.value_counts().to_dict()}")
    print("Datasets loaded successfully!")
