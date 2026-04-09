"""
현대카드 무실적 위험 고객 Wake-up 캠페인
TabDiff 기반 반사실적 설명 시스템 설정

[Use Case 변경]
추가 카드 발급 유도 → 무실적 위험 고객 활성화(Wake-up)
- 대상: 무실적 위험으로 예측된 고객 (inactive_risk = 1)
- 목표: 최소한의 행동 변화로 활성 고객(inactive_risk = 0) 상태로 전환
- 출력: 고객별 맞춤형 Wake-up Treatment 권고안
"""

# ─── 데이터 설정 ────────────────────────────────────────────────────────────────
DATA_CONFIG = {
    "n_samples": 5000,
    "random_seed": 42,
    "test_size": 0.2,
    "val_size": 0.1,
}

# ─── 피처 정의 ──────────────────────────────────────────────────────────────────
# 수치형 피처 (연속) — 행동/거래 패턴 중심
NUMERICAL_FEATURES = [
    "months_since_last_txn",      # 마지막 거래 경과 월 수 (높을수록 위험)
    "avg_spending_3m",             # 최근 3개월 평균 월 사용금액 (만원)
    "avg_spending_prev3m",         # 이전 3-6개월 평균 월 사용금액 (만원)
    "spending_trend_ratio",        # 사용금액 트렌드 (최근3m/이전3m, 1 미만 = 감소)
    "monthly_txn_count_3m",        # 최근 3개월 평균 월 거래 건수
    "num_missed_months",           # 최근 6개월 중 무실적(사용금액=0) 월 수
    "days_since_app_login",        # 앱 마지막 로그인 경과 일수
    "loyalty_points_balance",      # 미사용 포인트 잔액 (포인트)
    "num_active_benefits",         # 현재 활성 카드 혜택/서비스 수
    "years_as_customer",           # 현대카드 거래 연수
]

# 범주형 피처
CATEGORICAL_FEATURES = [
    "card_tier",                   # 카드 등급 (The, Black, Red, Blue)
    "primary_spending_category",   # 주요 사용 카테고리
    "payment_method",              # 주요 결제 수단
    "engagement_level",            # 디지털/앱 참여도
]

# 범주형 피처 값 목록
CATEGORICAL_VALUES = {
    "card_tier":                ["The", "Black", "Red", "Blue"],
    "primary_spending_category": ["dining", "shopping", "travel", "convenience", "online"],
    "payment_method":           ["app", "online", "offline_nfc", "offline_swipe"],
    "engagement_level":         ["high", "medium", "low"],
}

# 타겟 컬럼
TARGET_COLUMN = "inactive_risk"   # 1 = 무실적 위험, 0 = 활성 고객

# 전체 피처 목록 (순서 고정)
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ─── 행동 가능성 제약 (Actionability Constraints) ───────────────────────────────
# Wake-up 캠페인 맥락: 마케팅 액션으로 실제로 변화시킬 수 있는 피처만 허용

# 변경 불가 피처 (인구통계/기본 계약 조건)
IMMUTABLE_FEATURES = [
    "years_as_customer",           # 시간 기반 — 자연 변화
    "avg_spending_prev3m",         # 과거 실적 — 소급 변경 불가
    "card_tier",                   # 카드 등급 — 발급 조건에 의해 결정
]

# 값이 감소해야 하는 피처 (줄일수록 활성 상태 유지)
DECREASING_ONLY_FEATURES = [
    "months_since_last_txn",       # 최근 거래 늘어야 → 경과 월 감소
    "days_since_app_login",        # 앱 재방문 유도 → 경과 일수 감소
    "num_missed_months",           # 무실적 월 감소
]

# 값이 증가해야 하는 피처 (늘어날수록 활성 상태 유지)
INCREASING_ONLY_FEATURES = [
    "avg_spending_3m",             # 월 사용금액 증가
    "monthly_txn_count_3m",        # 월 거래 건수 증가
    "spending_trend_ratio",        # 사용 트렌드 개선
    "num_active_benefits",         # 혜택 가입 수 증가
]

# ─── TabDiff 모델 설정 ──────────────────────────────────────────────────────────
TABDIFF_CONFIG = {
    "num_timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 2e-2,
    "hidden_dim": 256,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "batch_size": 256,
    "num_epochs": 100,
    "device": "cpu",
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
    "num_cf_timesteps": 600,     # 반사실적 생성에 사용할 확산 스텝 수 (T_cf)
                                  # 400→600: noise level 높여 역확산 탐색 범위 확대
    "guidance_scale": 50.0,      # 분류기 guidance 강도 λ
                                  # 8.0→50.0: denoiser prior 압도를 위해 대폭 증가
                                  # DDIM에서 각 스텝마다 x₀_pred가 재예측되어
                                  # guidance가 누적되지 않으므로 충분히 강해야 함
    "num_cf_samples": 8,         # 고객당 생성할 후보 CF 수
    "target_class": 0,           # 목표 클래스 (0 = 활성 고객, 무실적 위험 해제)
    "proximity_weight": 0.005,   # 0.5→0.005: refine 시 경계 돌파 우선
    "max_change_ratio": 0.8,     # 0.5→0.8: 고위험 고객은 큰 변화 필요
    "refine_steps": 30,          # 10→30: 정제 스텝 증가
    "refine_lr": 0.05,           # 정제 학습률
    "ddim_steps": 50,            # DDIM 스텝 수 (None=DDPM 전체, 50=8× 빠름)
}
