"""
현대카드 고객 데이터 생성기
실제 데이터 특성을 반영한 합성 데이터를 생성합니다.
타겟: 추가 카드 발급 여부 (additional_card: 0/1)
"""

import numpy as np
import pandas as pd
from config import (
    DATA_CONFIG, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    CATEGORICAL_VALUES, TARGET_COLUMN
)


def generate_hyundai_card_data(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    현대카드 고객 특성을 반영한 합성 데이터 생성

    추가 카드 발급 확률에 영향을 주는 요인:
    - 신용점수 높을수록 ↑
    - 연소득 높을수록 ↑
    - 월 카드 사용금액 높을수록 ↑
    - 연체 횟수 적을수록 ↑
    - 고용 형태 (재직자 > 자영업자 > 퇴직자 > 무직자) ↑
    - 교육 수준 높을수록 ↑
    - 신용 한도 사용률 낮을수록 ↑
    """
    rng = np.random.default_rng(random_seed)
    n = n_samples

    # ── 인구통계 피처 ────────────────────────────────────────────────────────────
    age = rng.normal(loc=42, scale=12, size=n).clip(22, 70).astype(int)

    marital_status = rng.choice(
        CATEGORICAL_VALUES["marital_status"],
        p=[0.30, 0.58, 0.12],
        size=n,
    )

    education_level = rng.choice(
        CATEGORICAL_VALUES["education_level"],
        p=[0.25, 0.50, 0.25],
        size=n,
    )

    region = rng.choice(
        CATEGORICAL_VALUES["region"],
        p=[0.35, 0.28, 0.12, 0.25],
        size=n,
    )

    # ── 고용 형태 ────────────────────────────────────────────────────────────────
    # 나이에 따라 고용 형태 확률 조정
    employment_type = np.empty(n, dtype=object)
    for i in range(n):
        a = age[i]
        if a < 30:
            p = [0.70, 0.10, 0.15, 0.05]
        elif a < 50:
            p = [0.60, 0.22, 0.08, 0.10]
        elif a < 60:
            p = [0.50, 0.20, 0.05, 0.25]
        else:
            p = [0.20, 0.10, 0.05, 0.65]
        employment_type[i] = rng.choice(CATEGORICAL_VALUES["employment_type"], p=p)

    # ── 소득 (교육·고용 형태 반영) ────────────────────────────────────────────────
    edu_income_multiplier = {
        "high_school": 0.75,
        "college": 1.0,
        "graduate": 1.35,
    }
    emp_income_multiplier = {
        "employed": 1.0,
        "self_employed": 1.15,
        "unemployed": 0.35,
        "retired": 0.55,
    }
    base_income = rng.normal(loc=5500, scale=2500, size=n).clip(800, 25000)
    annual_income = np.array([
        base_income[i]
        * edu_income_multiplier[education_level[i]]
        * emp_income_multiplier[employment_type[i]]
        for i in range(n)
    ]).clip(500, 30000).round(0)

    # ── 신용점수 (소득·연체·연령 반영) ─────────────────────────────────────────────
    credit_score_base = (annual_income / 30000) * 300 + rng.normal(500, 80, n)
    credit_score = credit_score_base.clip(300, 900).round(0)

    # ── 현대카드 거래 연수 ────────────────────────────────────────────────────────
    years_as_customer = rng.exponential(scale=4, size=n).clip(0, 20).round(1)

    # ── 기존 보유 카드 수 ────────────────────────────────────────────────────────
    num_existing_cards = rng.choice([0, 1, 2, 3, 4, 5], p=[0.05, 0.35, 0.30, 0.18, 0.08, 0.04], size=n)

    # ── 월 카드 사용금액 (소득 대비) ─────────────────────────────────────────────
    monthly_spending = (annual_income / 12 * rng.uniform(0.05, 0.6, n)).clip(0, 5000).round(0)

    # ── 월 거래 건수 (사용금액과 상관) ───────────────────────────────────────────
    monthly_transactions = (monthly_spending / 30 * rng.uniform(0.5, 2.5, n) + rng.poisson(3, n)).clip(0, 120).round(0)

    # ── 총 대출금액 ─────────────────────────────────────────────────────────────
    total_loan_amount = (annual_income * rng.uniform(0, 3, n) * (1 - (credit_score - 300) / 600 * 0.4)).clip(0, 80000).round(0)

    # ── 연체 횟수 (신용점수 역상관) ──────────────────────────────────────────────
    delinquency_prob = 1 - (credit_score - 300) / 600
    num_delinquencies = np.array([
        rng.binomial(n=10, p=delinquency_prob[i] * 0.15)
        for i in range(n)
    ]).clip(0, 5)

    # ── 신용 한도 사용률 (%) ────────────────────────────────────────────────────
    utilization_rate = (monthly_spending / (annual_income / 12 * 0.5) * 100 + rng.normal(0, 10, n)).clip(0, 150).round(1)

    # ── 타겟 변수: 추가 카드 발급 ────────────────────────────────────────────────
    # 로지스틱 함수 기반 확률 계산
    score = (
        0.003 * (credit_score - 600)          # 신용점수 기여
        + 0.00005 * (annual_income - 5000)    # 소득 기여
        + 0.001 * (monthly_spending - 150)    # 사용금액 기여
        + 0.005 * (monthly_transactions - 10) # 거래건수 기여
        - 0.3 * num_delinquencies             # 연체 패널티
        - 0.003 * (utilization_rate - 30)     # 한도 사용률 패널티
        + 0.05 * years_as_customer            # 장기고객 보너스
        + np.array([
            0.4 if e == "employed" else
            0.25 if e == "self_employed" else
            -0.5 if e == "unemployed" else
            0.0
            for e in employment_type
        ])
        + np.array([
            0.2 if edu == "graduate" else
            0.0 if edu == "college" else
            -0.15
            for edu in education_level
        ])
        + rng.normal(0, 0.3, n)  # 무작위 노이즈
    )
    prob = 1 / (1 + np.exp(-score))
    additional_card = (rng.uniform(size=n) < prob).astype(int)

    # ── DataFrame 조립 ─────────────────────────────────────────────────────────
    df = pd.DataFrame({
        "age":                  age,
        "annual_income":        annual_income,
        "credit_score":         credit_score,
        "num_existing_cards":   num_existing_cards,
        "monthly_spending":     monthly_spending,
        "years_as_customer":    years_as_customer,
        "total_loan_amount":    total_loan_amount,
        "monthly_transactions": monthly_transactions,
        "num_delinquencies":    num_delinquencies,
        "utilization_rate":     utilization_rate,
        "marital_status":       marital_status,
        "employment_type":      employment_type,
        "education_level":      education_level,
        "region":               region,
        "additional_card":      additional_card,
    })

    # 고객 ID 추가
    df.insert(0, "customer_id", [f"HCC_{i+1:05d}" for i in range(n)])

    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """데이터 요약 통계 출력"""
    print("=" * 60)
    print("현대카드 고객 데이터 요약")
    print("=" * 60)
    print(f"총 고객 수: {len(df):,}명")
    print(f"추가 발급 동의(1): {df[TARGET_COLUMN].sum():,}명 ({df[TARGET_COLUMN].mean()*100:.1f}%)")
    print(f"추가 발급 미동의(0): {(1-df[TARGET_COLUMN]).sum():,}명 ({(1-df[TARGET_COLUMN].mean())*100:.1f}%)")
    print()
    print("[수치형 피처 요약]")
    num_cols = NUMERICAL_FEATURES
    print(df[num_cols].describe().round(1).to_string())
    print()
    print("[범주형 피처 분포]")
    for col in CATEGORICAL_FEATURES:
        print(f"\n  {col}:")
        counts = df[col].value_counts()
        for val, cnt in counts.items():
            print(f"    {val:15s}: {cnt:5d}명 ({cnt/len(df)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    df = generate_hyundai_card_data(n_samples=5000, random_seed=42)
    print_data_summary(df)
    df.to_csv("data/hyundai_card_customers.csv", index=False)
    print("\n데이터 저장 완료: data/hyundai_card_customers.csv")
