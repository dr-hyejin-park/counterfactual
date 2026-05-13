# 무실적 위험 고객 Wake-up Campaign

**TabDiff 기반 반사실적 설명 시스템 — 고객별 맞춤형 Wake-up Treatment 권고**

> 논문 응용: *"Tabular Diffusion Based Actionable Counterfactual Explanations  
> for Network Intrusion Detection"* → 현대카드 무실적 위험 고객 활성화 도메인 재적용

---

## 개요

카드 사용이 감소하거나 장기 미사용으로 **무실적 전환이 예상되는 고객**을 대상으로,  
TabDiff(표형 데이터 확산 모델)를 활용하여:

- "어떤 행동 변화가 이 고객을 활성 상태로 되돌릴 수 있는가?"
- 각 고객의 **맞춤형 Wake-up Treatment**(캠페인 처방)를 생성
- 마케팅 팀이 개인화된 리텐션 캠페인을 기획하는 데 직접 활용

### 출력 예시

```
고객 ID: HCC_01076  |  카드 등급: Red  |  거래 연수: 0.6년
현재 무실적 위험도: 80.7%  (최근 무실적 4개월, 앱 미사용 101일)

[Wake-up 시나리오 — ✓ 달성]
목표 달성 시 무실적 위험도: 24.5%  (↓56.2%p 감소)

피처                     현재값     목표값        변화
──────────────────────────────────────────────────────
최근 3M 월 사용금액        41만원    139만원   ↑98 (239%)
앱 미사용 기간            101일        0일  ↓101.0 (100%)
최근 3M 월 거래 건수        5건        9건    ↑4.0 (80%)
마지막 거래 경과           4.3개월    0.9개월  ↓3.4 (79%)
활성 혜택 수               5개        8개    ↑3.0 (60%)

[Wake-up 캠페인 액션 권고]
→ [사용 실적] 월 사용금액 목표 달성 시 캐시백/포인트 추가 적립 캠페인
→ [앱 재참여] 푸시 알림 발송 / 101일 미접속 특별 혜택 안내
→ [결제 빈도] 소액 결제 N건당 포인트 지급 미션 이벤트
```

---

## 시스템 아키텍처

```
현대카드 고객 행동 데이터 (거래 패턴, 앱 사용, 혜택 이용 등)
                 │
                 ▼
    ┌─────────────────────┐
    │   전처리 파이프라인  │  StandardScaler + One-hot (26차원)
    └────────┬────────────┘
             │
        ┌────┴────┐
        ▼         ▼
  ┌──────────┐  ┌──────────────────┐
  │ TabDiff  │  │  무실적 위험      │  P(inactive_risk=1 | x)
  │  DDPM    │  │  분류기           │  신경망 기반
  └────┬─────┘  └──────┬───────────┘
       │                │
       └───────┬────────┘
               ▼
  ┌─────────────────────────┐
  │   Counterfactual 생성   │
  │  • 부분 노이즈 추가      │
  │  • Classifier Guidance  │  ∇ log P(inactive_risk=0 | x_t)
  │  • 행동 가능성 제약      │  과거 실적·등급 불변, 사용금액↑ 등
  └─────────────────────────┘
               │
               ▼
       고객별 Wake-up Treatment 권고
```

### 핵심 알고리즘 (논문 Algorithm 1 응용)

```
Input : x_factual (무실적 위험 고객), 분류기 f, TabDiff ε_θ
Output: x_cf (활성 고객 상태로 전환하는 최소 변화)

1. x_{T_cf} = q(x_{T_cf} | x_factual)        ← 부분 노이즈 추가
2. for t = T_cf, ..., 1:
     ε̃ = ε_θ(x_t, t) - λ√(1-ᾱ_t) · ∇ log P(inactive_risk=0 | x_t)
     x_{t-1} = DDPM_reverse(x_t, ε̃)
     x_{t-1} = apply_constraints(x_{t-1}, x_factual)
3. return decode(x_0)
```

---

## 파일 구조

```
counterfactual/
├── config.py                     # 피처 정의, 행동 가능성 제약, 모델 파라미터
├── train.py                      # 모델 학습 스크립트
├── run_demo.py                   # Wake-up CF 생성 및 보고서 출력
├── requirements.txt
│
├── data/
│   └── hyundai_card_data.py      # 합성 고객 행동 데이터 생성기
│
├── models/
│   ├── tabdiff.py                # TabDiff DDPM (Transformer 기반)
│   └── classifier.py             # 무실적 위험 예측 분류기
│
├── cf_engine/
│   ├── constraints.py            # 행동 가능성 제약 적용
│   └── generator.py              # CF 생성 + Wake-up Treatment 추출
│
└── utils/
    └── preprocessing.py          # 전처리 파이프라인 (스케일링 + 인코딩)
```

---

## 사용법

### 1. 환경 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 학습

```bash
# 전체 학습 (권장: 5000 샘플, 100에폭)
python train.py

# 빠른 테스트 (2000 샘플, 10에폭)
python train.py --quick
```

출력 파일:
- `checkpoints/tabdiff.pt` — TabDiff DDPM 모델
- `checkpoints/classifier.pt` — 무실적 위험 분류기
- `checkpoints/preprocessor.pkl` — 전처리기
- `checkpoints/test_customers.csv` — 테스트 고객 데이터

### 3. Wake-up CF 생성 (데모)

```bash
# 무실적 위험 고객 10명 대상 CF 생성
python run_demo.py --n_customers 10

# 특정 고객 ID
python run_demo.py --customer_id HCC_01076
```

출력:
- 콘솔: 고객별 Wake-up 보고서 (Treatment 권고 포함)
- `results/wakeup_cf_results.csv` — 상세 결과 CSV
- `results/cf_report.png` — 분석 시각화

---

## 피처 정의 및 행동 가능성 제약

| 피처 | 설명 | 제약 방향 |
|------|------|-----------|
| months_since_last_txn | 마지막 거래 경과 월 수 | 감소만 허용 ↓ |
| avg_spending_3m | 최근 3개월 평균 월 사용금액 | 증가만 허용 ↑ |
| avg_spending_prev3m | 이전 3-6개월 평균 월 사용금액 | **불변** |
| spending_trend_ratio | 사용금액 트렌드 (최근/이전) | 증가만 허용 ↑ |
| monthly_txn_count_3m | 최근 3개월 평균 월 거래 건수 | 증가만 허용 ↑ |
| num_missed_months | 최근 6개월 무실적 월 수 | 감소만 허용 ↓ |
| days_since_app_login | 앱 마지막 로그인 경과 일수 | 감소만 허용 ↓ |
| loyalty_points_balance | 미사용 포인트 잔액 | 자유 변경 |
| num_active_benefits | 활성 혜택/서비스 수 | 증가만 허용 ↑ |
| years_as_customer | 거래 연수 | **불변** |
| card_tier | 카드 등급 | **불변** |
| primary_spending_category | 주요 사용 카테고리 | 자유 변경 |
| payment_method | 주요 결제 수단 | 자유 변경 |
| engagement_level | 디지털 참여도 | 자유 변경 |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| **Validity** | CF 적용 후 활성 고객(0)으로 예측되는 비율 |
| **Proximity** | CF와 원본 간 L2 거리 (낮을수록 최소 변화) |
| **Sparsity** | 변경된 피처 비율 (낮을수록 단순한 처방) |
| **Risk Drop** | 무실적 위험도 감소폭 (%p) |

---

## 마케팅 캠페인 활용 방안

Treatment 유형별 권고 캠페인:

| Treatment | 캠페인 액션 |
|-----------|------------|
| 앱 미사용 기간 감소 ↓ | 미접속 일수 기반 푸시 알림 / 재방문 이벤트 |
| 월 사용금액 증가 ↑ | 월 실적 구간별 캐시백/포인트 지급 |
| 월 거래 건수 증가 ↑ | 소액 N건당 보너스 포인트 미션 |
| 무실적 월 감소 ↓ | 연속 이용 리워드 (streak 보너스) |
| 활성 혜택 수 증가 ↑ | 미가입 혜택 안내 / 포인트 활용 가이드 |
| 포인트 소진 ↓ | 포인트 만료 알림 / 사용처 추천 |
| 결제 수단 전환 | 앱결제·간편결제 전환 시 추가 적립 |
