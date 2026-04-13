"""
Python Agentic AI project for HR Analytics.

The agent interprets a user query and then uses tool-like functions to:
1) Load dataset
2) Clean dataset
3) Analyze attrition, salary, and department
4) Generate a report

Optional features:
- Save report to text file
- Send report via SMTP email
"""

import argparse
import smtplib
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict

import pandas as pd


def data_loading(file_path: str = "data.csv") -> pd.DataFrame:
    """Tool 1: Read HR dataset from CSV."""
    print(f"[Tool:data_loading] Loading dataset from '{file_path}'...")
    df = pd.read_csv(file_path)
    print(f"[Tool:data_loading] Loaded {len(df)} rows and {len(df.columns)} columns.")
    return df


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Tool 2: Clean missing values, duplicates, and Attrition encoding."""
    print("[Tool:data_cleaning] Cleaning dataset...")
    cleaned = df.copy()

    # Fill missing values by data type for robust automated processing.
    num_cols = cleaned.select_dtypes(include=["number"]).columns
    cat_cols = cleaned.select_dtypes(exclude=["number"]).columns

    for col in num_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    for col in cat_cols:
        if cleaned[col].isna().any():
            mode_series = cleaned[col].mode()
            if not mode_series.empty:
                cleaned[col] = cleaned[col].fillna(mode_series.iloc[0])

    # Remove duplicate records.
    duplicates_removed = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()

    # Encode Attrition into binary values so downstream analysis is consistent.
    if "Attrition" not in cleaned.columns:
        raise KeyError("Required column 'Attrition' not found in dataset.")
    cleaned["Attrition"] = cleaned["Attrition"].map({"Yes": 1, "No": 0})
    cleaned["Attrition"] = cleaned["Attrition"].fillna(0).astype(int)

    print(f"[Tool:data_cleaning] Duplicates removed: {duplicates_removed}")
    print(
        "[Tool:data_cleaning] Missing values remaining: "
        f"{int(cleaned.isna().sum().sum())}"
    )
    return cleaned


def analysis(df: pd.DataFrame, user_query: str) -> Dict[str, object]:
    """Tool 3: Analyze attrition, salary, and department based on user query intent."""
    print(f"[Tool:analysis] Running analysis for query: '{user_query}'")

    analysis_result: Dict[str, object] = {
        "query": user_query,
        "records": len(df),
        "columns": list(df.columns),
    }

    # The agent always performs core HR analysis requested in the specification.
    attrition_rate = float(df["Attrition"].mean() * 100)
    attrition_counts = df["Attrition"].value_counts().to_dict()
    analysis_result["attrition_rate"] = round(attrition_rate, 2)
    analysis_result["attrition_counts"] = attrition_counts

    dept_attrition = (df.groupby("Department")["Attrition"].mean() * 100).sort_values(
        ascending=False
    )
    analysis_result["department_attrition_rate"] = dept_attrition.round(2).to_dict()

    income_by_attrition = df.groupby("Attrition")["MonthlyIncome"].mean()
    analysis_result["avg_income_stayed"] = round(float(income_by_attrition.get(0, 0.0)), 2)
    analysis_result["avg_income_left"] = round(float(income_by_attrition.get(1, 0.0)), 2)

    salary_stats = df["MonthlyIncome"].describe().round(2).to_dict()
    analysis_result["salary_summary"] = salary_stats

    return analysis_result


def report_generation(result: Dict[str, object]) -> str:
    """Tool 4: Generate a readable summary report from analysis output."""
    dept_rates = result["department_attrition_rate"]
    top_dept = max(dept_rates, key=dept_rates.get)
    top_dept_rate = dept_rates[top_dept]

    report = f"""
HR ANALYTICS AGENT REPORT
=========================
User Query: {result['query']}
Records Analyzed: {result['records']}

1) ATTRITION OVERVIEW
- Attrition Rate: {result['attrition_rate']}%
- Attrition Counts: {result['attrition_counts']} (0=No, 1=Yes)

2) DEPARTMENT ANALYSIS
- Department-wise Attrition Rate (%): {result['department_attrition_rate']}
- Highest Attrition Department: {top_dept} ({top_dept_rate}%)

3) SALARY ANALYSIS
- Average MonthlyIncome (Stayed): {result['avg_income_stayed']}
- Average MonthlyIncome (Left): {result['avg_income_left']}
- Salary Summary: {result['salary_summary']}

4) KEY INSIGHTS
- Attrition is concentrated in {top_dept}.
- Employees who leave generally have lower average monthly income than those who stay.
- This report is generated automatically by the HR analytics agent workflow.
""".strip()

    print("[Tool:report_generation] Report generated.")
    return report


def save_report(report_text: str, output_path: str = "hr_agent_report.txt") -> None:
    """Optional helper: Save report as text file."""
    Path(output_path).write_text(report_text, encoding="utf-8")
    print(f"[Optional] Report saved to '{output_path}'.")


def send_report_email(
    report_text: str,
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    receiver_email: str,
) -> None:
    """Optional helper: Send report via SMTP."""
    msg = MIMEText(report_text)
    msg["Subject"] = "HR Analytics Agent Report"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [receiver_email], msg.as_string())
    print(f"[Optional] Report emailed to '{receiver_email}'.")


class HRAgent:
    """
    A simple agent orchestrator.

    Agent behavior:
    - Understands user query text.
    - Executes tools in a deterministic workflow.
    - Produces an automated summary report.
    """

    def __init__(self, dataset_path: str = "data.csv") -> None:
        self.dataset_path = dataset_path

    def run(self, user_query: str) -> str:
        """Execute agent workflow end-to-end."""
        print("\n[Agent] Starting HR analytics workflow...")

        # Step 1: Read dataset
        df = data_loading(self.dataset_path)

        # Step 2: Clean data
        clean_df = data_cleaning(df)

        # Step 3: Analyze attrition, salary, department
        result = analysis(clean_df, user_query)

        # Step 4: Generate summary report
        report = report_generation(result)

        print("[Agent] Workflow completed.\n")
        return report


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for running the agent and optional features."""
    parser = argparse.ArgumentParser(description="Agentic AI HR Analytics Project")
    parser.add_argument(
        "--query",
        type=str,
        default="analyze attrition",
        help="Natural language query for the HR analytics agent.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data.csv",
        help="Path to HR dataset CSV.",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save report as text file.",
    )
    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Send report by SMTP email (requires SMTP arguments).",
    )
    parser.add_argument("--smtp-server", type=str, default="")
    parser.add_argument("--smtp-port", type=int, default=587)
    parser.add_argument("--sender-email", type=str, default="")
    parser.add_argument("--sender-password", type=str, default="")
    parser.add_argument("--receiver-email", type=str, default="")
    return parser


def main() -> None:
    """Run the HR agent from CLI."""
    args = build_parser().parse_args()
    agent = HRAgent(dataset_path=args.dataset)
    final_report = agent.run(args.query)

    print(final_report)

    if args.save_report:
        save_report(final_report)

    if args.send_email:
        required = [
            args.smtp_server,
            args.sender_email,
            args.sender_password,
            args.receiver_email,
        ]
        if not all(required):
            raise ValueError(
                "For --send-email, provide --smtp-server, --sender-email, "
                "--sender-password, and --receiver-email."
            )
        send_report_email(
            report_text=final_report,
            smtp_server=args.smtp_server,
            smtp_port=args.smtp_port,
            sender_email=args.sender_email,
            sender_password=args.sender_password,
            receiver_email=args.receiver_email,
        )


if __name__ == "__main__":
    main()
