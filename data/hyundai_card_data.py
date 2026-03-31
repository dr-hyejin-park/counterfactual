"""
현대카드 무실적 위험 고객 데이터 생성기
거래 행동 패턴 기반 합성 데이터를 생성합니다.

타겟: inactive_risk
  1 = 무실적 위험 (향후 3개월 내 무실적 전환 가능성 높음)
  0 = 활성 고객 (정상 카드 사용 유지)

무실적 위험을 높이는 요인:
  - 최근 거래 경과 기간 길수록 ↑
  - 최근 3개월 사용금액 감소 추세일수록 ↑
  - 무실적 월 수가 많을수록 ↑
  - 앱 로그인 경과 일수가 길수록 ↑
  - 활성 혜택 수가 적을수록 ↑
  - 포인트 잔액이 지나치게 많을수록 ↑ (소진 안 함 = 비참여)
  - 카드 등급이 낮을수록 ↑
  - 디지털 참여도가 낮을수록 ↑
"""

import numpy as np
import pandas as pd
from config import (
    DATA_CONFIG, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    CATEGORICAL_VALUES, TARGET_COLUMN,
)


def generate_hyundai_card_data(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    현대카드 고객 행동 패턴 기반 합성 데이터 생성
    무실적 위험(inactive_risk) 레이블 포함
    """
    rng = np.random.default_rng(random_seed)
    n = n_samples

    # ── 카드 등급 ────────────────────────────────────────────────────────────
    card_tier = rng.choice(
        CATEGORICAL_VALUES["card_tier"],
        p=[0.05, 0.15, 0.40, 0.40],   # The < Black < Red ≈ Blue
        size=n,
    )
    tier_rank = np.array([
        {"The": 4, "Black": 3, "Red": 2, "Blue": 1}[t] for t in card_tier
    ])  # 등급 숫자 (높을수록 프리미엄)

    # ── 거래 연수 ─────────────────────────────────────────────────────────────
    years_as_customer = rng.exponential(scale=4, size=n).clip(0, 20).round(1)

    # ── 이전 3-6개월 평균 사용금액 (기준점) ─────────────────────────────────────
    # 카드 등급이 높을수록 기본 사용금액 높음
    base_spending = (
        rng.lognormal(mean=4.8, sigma=0.8, size=n)
        * (0.6 + 0.15 * tier_rank)
    ).clip(5, 3000)
    avg_spending_prev3m = base_spending.round(0)

    # ── 최근 3개월 사용금액 트렌드 ──────────────────────────────────────────────
    # 대부분은 비슷하거나 약간 변동, 일부는 급감 (무실적 위험군)
    trend_ratio_base = rng.normal(loc=0.92, scale=0.3, size=n)  # 평균적으로 소폭 감소
    spending_trend_ratio = trend_ratio_base.clip(0.01, 2.5).round(3)
    avg_spending_3m = (avg_spending_prev3m * spending_trend_ratio).clip(0, 3000).round(0)

    # ── 최근 3개월 월 거래 건수 ──────────────────────────────────────────────────
    # 사용금액에 비례하되 카드 등급이 높을수록 거래 건수도 많음
    monthly_txn_base = avg_spending_3m / 20 + rng.poisson(3, n) + tier_rank * 0.5
    monthly_txn_count_3m = monthly_txn_base.clip(0, 80).round(0)

    # ── 최근 6개월 중 무실적 월 수 ──────────────────────────────────────────────
    # 트렌드가 나쁠수록, 사용금액이 낮을수록 무실적 월 증가
    inactivity_base_prob = np.clip(
        0.05
        + 0.4 * (avg_spending_3m < 20).astype(float)
        + 0.3 * (spending_trend_ratio < 0.5).astype(float)
        + 0.2 * (monthly_txn_count_3m < 2).astype(float),
        0, 0.9,
    )
    num_missed_months = np.array([
        rng.binomial(6, inactivity_base_prob[i]) for i in range(n)
    ]).clip(0, 6)

    # ── 마지막 거래 경과 월 수 ──────────────────────────────────────────────────
    months_since_last_txn = np.where(
        num_missed_months == 0,
        rng.uniform(0, 1, n),                # 활성 고객: 최근 1개월 이내
        rng.uniform(0.5, num_missed_months + 0.5, size=n),   # 미사용 기간 반영
    ).clip(0, 12).round(1)

    # ── 앱 마지막 로그인 경과 일수 ──────────────────────────────────────────────
    # 디지털 참여도가 낮은 고객 / 오래된 고객일수록 앱 미사용 기간 길어짐
    app_base_days = rng.exponential(scale=20, size=n) * (1 + months_since_last_txn * 3)
    days_since_app_login = app_base_days.clip(0, 365).round(0)

    # ── 미사용 포인트 잔액 ──────────────────────────────────────────────────────
    # 비활동 고객은 포인트를 소진하지 않아 잔액 과다
    loyalty_base = rng.exponential(scale=8000, size=n) * (1 + months_since_last_txn * 0.3)
    loyalty_points_balance = loyalty_base.clip(0, 200000).round(0)

    # ── 활성 혜택/서비스 수 ─────────────────────────────────────────────────────
    # 등급이 높고 활동적일수록 더 많은 혜택 이용
    benefits_base = (
        tier_rank * 1.5
        + rng.poisson(2, n)
        - months_since_last_txn * 0.3
        + (years_as_customer > 2).astype(float) * 0.5
    )
    num_active_benefits = benefits_base.clip(0, 12).round(0).astype(int)

    # ── 주요 사용 카테고리 ──────────────────────────────────────────────────────
    primary_spending_category = rng.choice(
        CATEGORICAL_VALUES["primary_spending_category"],
        p=[0.30, 0.28, 0.15, 0.17, 0.10],
        size=n,
    )

    # ── 주요 결제 수단 ──────────────────────────────────────────────────────────
    # 디지털 결제(app/online)일수록 활동성 높음
    payment_method = np.empty(n, dtype=object)
    for i in range(n):
        if days_since_app_login[i] < 7:
            p = [0.45, 0.30, 0.18, 0.07]   # app 위주
        elif days_since_app_login[i] < 30:
            p = [0.25, 0.30, 0.30, 0.15]
        else:
            p = [0.05, 0.15, 0.35, 0.45]   # offline 위주 (앱 미사용)
        payment_method[i] = rng.choice(CATEGORICAL_VALUES["payment_method"], p=p)

    # ── 디지털 참여도 ────────────────────────────────────────────────────────────
    engagement_level = np.empty(n, dtype=object)
    for i in range(n):
        d = days_since_app_login[i]
        m = months_since_last_txn[i]
        if d < 7 and m < 1:
            p = [0.70, 0.25, 0.05]   # high
        elif d < 30 and m < 2:
            p = [0.25, 0.55, 0.20]   # medium
        else:
            p = [0.05, 0.25, 0.70]   # low
        engagement_level[i] = rng.choice(CATEGORICAL_VALUES["engagement_level"], p=p)

    # ── 타겟: 무실적 위험 ────────────────────────────────────────────────────────
    score = (
        - 0.4  * months_since_last_txn               # 경과 기간 길수록 위험
        - 0.003 * avg_spending_3m                     # 사용금액 낮을수록 위험
        + 0.5  * spending_trend_ratio                 # 트렌드 좋을수록 안전
        + 0.08 * monthly_txn_count_3m                 # 거래 건수 많을수록 안전
        - 0.5  * num_missed_months                    # 무실적 월 많을수록 위험
        - 0.005 * days_since_app_login                # 앱 미사용 길수록 위험
        + 0.2  * num_active_benefits                  # 혜택 많을수록 안전
        + 0.3  * tier_rank                            # 등급 높을수록 안전
        + 0.1  * years_as_customer                    # 장기고객일수록 안전
        + np.array([
            0.5 if e == "high" else 0.0 if e == "medium" else -0.5
            for e in engagement_level
        ])
        + np.array([
            0.2 if p in ("app", "online") else -0.2
            for p in payment_method
        ])
        + rng.normal(0, 0.4, n)
    )
    # 무실적 위험: score가 낮을수록 위험 → 확률 반전
    prob_inactive = 1 / (1 + np.exp(score - 1))
    inactive_risk = (rng.uniform(size=n) < prob_inactive).astype(int)

    # ── DataFrame 조립 ─────────────────────────────────────────────────────
    df = pd.DataFrame({
        "months_since_last_txn":    months_since_last_txn,
        "avg_spending_3m":           avg_spending_3m,
        "avg_spending_prev3m":       avg_spending_prev3m,
        "spending_trend_ratio":      spending_trend_ratio,
        "monthly_txn_count_3m":      monthly_txn_count_3m,
        "num_missed_months":         num_missed_months,
        "days_since_app_login":      days_since_app_login,
        "loyalty_points_balance":    loyalty_points_balance,
        "num_active_benefits":       num_active_benefits,
        "years_as_customer":         years_as_customer,
        "card_tier":                 card_tier,
        "primary_spending_category": primary_spending_category,
        "payment_method":            payment_method,
        "engagement_level":          engagement_level,
        "inactive_risk":             inactive_risk,
    })

    df.insert(0, "customer_id", [f"HCC_{i+1:05d}" for i in range(n)])
    return df


def print_data_summary(df: pd.DataFrame) -> None:
    """데이터 요약 통계 출력"""
    print("=" * 60)
    print("현대카드 고객 행동 데이터 요약 (무실적 위험 분류)")
    print("=" * 60)
    print(f"총 고객 수: {len(df):,}명")
    print(f"무실적 위험(1): {df[TARGET_COLUMN].sum():,}명 ({df[TARGET_COLUMN].mean()*100:.1f}%)")
    print(f"활성 고객(0):   {(1-df[TARGET_COLUMN]).sum():,}명 ({(1-df[TARGET_COLUMN].mean())*100:.1f}%)")
    print()
    print("[수치형 피처 요약]")
    print(df[NUMERICAL_FEATURES].describe().round(2).to_string())
    print()
    print("[범주형 피처 분포]")
    for col in CATEGORICAL_FEATURES:
        print(f"\n  {col}:")
        counts = df[col].value_counts()
        for val, cnt in counts.items():
            print(f"    {val:20s}: {cnt:5d}명 ({cnt/len(df)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    df = generate_hyundai_card_data(n_samples=5000, random_seed=42)
    print_data_summary(df)
    df.to_csv("data/hyundai_card_customers.csv", index=False)
    print("\n데이터 저장 완료: data/hyundai_card_customers.csv")
