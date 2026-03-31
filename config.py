"""
현대카드 추가 발급 Counterfactual Explanation 시스템 설정
TabDiff 기반 반사실적 설명 생성 (논문: Tabular Diffusion Based Actionable
Counterfactual Explanations for Network Intrusion Detection 응용)
"""

# ─── 데이터 설정 ────────────────────────────────────────────────────────────────
DATA_CONFIG = {
    "n_samples": 5000,
    "random_seed": 42,
    "test_size": 0.2,
    "val_size": 0.1,
}

# ─── 피처 정의 ──────────────────────────────────────────────────────────────────
# 수치형 피처 (연속)
NUMERICAL_FEATURES = [
    "age",                    # 나이 (세)
    "annual_income",          # 연소득 (만원)
    "credit_score",           # 신용점수 (300~900)
    "num_existing_cards",     # 보유 카드 수
    "monthly_spending",       # 월 카드 사용금액 (만원)
    "years_as_customer",      # 현대카드 거래 연수
    "total_loan_amount",      # 총 대출금액 (만원)
    "monthly_transactions",   # 월 거래 건수
    "num_delinquencies",      # 연체 횟수
    "utilization_rate",       # 신용 한도 사용률 (%)
]

# 범주형 피처
CATEGORICAL_FEATURES = [
    "marital_status",         # 혼인 상태
    "employment_type",        # 고용 형태
    "education_level",        # 교육 수준
    "region",                 # 거주 지역
]

# 범주형 피처 값 목록
CATEGORICAL_VALUES = {
    "marital_status":   ["single", "married", "divorced"],
    "employment_type":  ["employed", "self_employed", "unemployed", "retired"],
    "education_level":  ["high_school", "college", "graduate"],
    "region":           ["Seoul", "Gyeonggi", "Busan", "Others"],
}

# 타겟 컬럼
TARGET_COLUMN = "additional_card"   # 1 = 추가 발급 동의/자격, 0 = 미동의/미자격

# 전체 피처 목록 (순서 고정)
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ─── 행동 가능성 제약 (Actionability Constraints) ───────────────────────────────
# 반사실적 생성 시 변경 불가 혹은 방향이 제한된 피처
IMMUTABLE_FEATURES = [
    "age",
    "marital_status",
    "region",
    "education_level",
    "years_as_customer",   # 시간 기반 — 자연 증가
    "total_loan_amount",   # 마케팅 액션으로 직접 변경 불가
]

# 값이 증가만 가능한 피처 (현실적으로 줄이기 어려운 항목)
INCREASING_ONLY_FEATURES = [
    "annual_income",
    "credit_score",
    "monthly_spending",
    "monthly_transactions",
]

# 값이 감소만 가능한 피처
DECREASING_ONLY_FEATURES = [
    "num_delinquencies",
    "utilization_rate",
]

# ─── TabDiff 모델 설정 ──────────────────────────────────────────────────────────
TABDIFF_CONFIG = {
    "num_timesteps": 1000,       # 전체 확산 스텝 수
    "beta_start": 1e-4,          # 노이즈 스케줄 시작값
    "beta_end": 2e-2,            # 노이즈 스케줄 종료값
    "hidden_dim": 256,           # Transformer 숨겨진 차원
    "num_heads": 8,              # Multi-head Attention 헤드 수
    "num_layers": 6,             # Transformer 레이어 수
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "num_epochs": 100,
    "device": "cpu",             # GPU 없는 환경 대비 cpu 기본
}

# ─── 분류기 설정 ────────────────────────────────────────────────────────────────
CLASSIFIER_CONFIG = {
    "hidden_dims": [128, 64, 32],
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "num_epochs": 80,
    "device": "cpu",
}

# ─── 반사실적 생성 설정 ─────────────────────────────────────────────────────────
CF_CONFIG = {
    "num_cf_timesteps": 300,     # 반사실적 생성에 사용할 확산 스텝 수 (T_cf)
    "guidance_scale": 3.0,       # 분류기 guidance 강도 λ
    "num_cf_samples": 5,         # 고객당 생성할 후보 CF 수
    "target_class": 1,           # 목표 클래스 (1 = 추가 발급 승인)
    "proximity_weight": 0.5,     # 원본 데이터와의 근접도 가중치
    "max_change_ratio": 0.5,     # 피처당 최대 변화 비율 (원본 대비)
}
