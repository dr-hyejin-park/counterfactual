# 현대카드 추가 발급 Counterfactual Explanation System

**TabDiff 기반 반사실적 설명 생성 — 고객별 추가 카드 발급 유도 Treatment 권고**

> 논문 응용: *"Tabular Diffusion Based Actionable Counterfactual Explanations  
> for Network Intrusion Detection"* → 현대카드 추가 발급 도메인에 재적용

---

## 개요

추가 카드 발급이 **거절된 고객**에 대해 TabDiff(표형 데이터 확산 모델)를 활용하여:

- "어떤 조건이 바뀌면 추가 발급이 승인될 수 있는가?"
- 각 고객의 **맞춤형 Treatment**(처방 권고안)를 생성
- 마케팅 팀의 개인화된 캠페인 기획에 활용

### 출력 예시

```
고객 ID: HCC_00466
현재 예측: 추가 발급 거절 (확률: 33.3%)

[반사실적 시나리오 — 유효]
목표 예측: 추가 발급 승인 (확률: 62.2%)

피처               현재값         목표값          변화
──────────────────────────────────────────────────
연소득          3,653만원     5,230만원   ↑1,577 (43%)
월 거래 건수        10건           13건       ↑3 (30%)
신용점수           463점          544점      ↑81 (17%)
신용 한도 사용률     66%            56%      ↓11 (16%)

[마케팅 인사이트]
→ 소득 증가 시 자동 한도 상향 안내
→ 소액 결제 혜택(포인트/캐시백) 강화 마케팅
→ 신용점수 개선 프로그램 안내 (크레딧 빌딩 상품)
```

---

## 시스템 아키텍처

```
현대카드 고객 데이터
        │
        ▼
┌─────────────────┐
│  전처리 파이프라인  │  StandardScaler + One-hot 인코딩
│  (24차원 벡터)   │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────┐
│TabDiff │ │ 분류기        │  신경망 기반
│ DDPM   │ │(CardIssuance)│  binary classifier
└────┬───┘ └──────┬───────┘
     │             │
     └──────┬──────┘
            ▼
┌────────────────────────┐
│  Counterfactual 생성기  │
│  • 부분 노이즈 추가     │
│  • Classifier Guidance │  ∇ log P(y=1|x_t)
│  • 행동 가능성 제약     │  불변/단조 피처 처리
└────────────────────────┘
            │
            ▼
    고객별 Treatment 권고안
```

### 핵심 알고리즘 (논문 Algorithm 1 응용)

```
Input : x_factual (거절 고객), 분류기 f, TabDiff 모델 ε_θ
Output: x_cf (최소 변화로 승인 조건 충족)

1. x_{T_cf} = q(x_{T_cf} | x_factual)  ← 부분 노이즈 추가
2. for t = T_cf, ..., 1:
     ε̃ = ε_θ(x_t, t) - λ√(1-ᾱ_t) · ∇ log f(y=1|x_t)
     x_{t-1} = DDPM_reverse(x_t, ε̃)
     x_{t-1} = apply_constraints(x_{t-1}, x_factual)
3. return decode(x_0)
```

---

## 파일 구조

```
counterfactual/
├── config.py                    # 전역 설정 (피처, 모델, CF 파라미터)
├── train.py                     # 모델 학습 스크립트
├── run_demo.py                  # CF 생성 및 보고서 출력
├── requirements.txt
│
├── data/
│   └── hyundai_card_data.py     # 합성 현대카드 데이터 생성기
│
├── models/
│   ├── tabdiff.py               # TabDiff DDPM 모델 (Transformer 기반)
│   └── classifier.py            # 추가 발급 예측 분류기
│
├── cf_engine/
│   ├── constraints.py           # 행동 가능성 제약 (불변/단조 피처)
│   └── generator.py             # Counterfactual 생성 및 Treatment 추출
│
└── utils/
    └── preprocessing.py         # 데이터 전처리 파이프라인
```

---

## 사용법

### 1. 환경 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 학습

```bash
# 전체 학습 (권장)
python train.py

# 빠른 테스트
python train.py --quick
```

학습 결과:
- `checkpoints/tabdiff.pt` — TabDiff DDPM 모델
- `checkpoints/classifier.pt` — 추가 발급 분류기
- `checkpoints/preprocessor.pkl` — 전처리기
- `checkpoints/test_customers.csv` — 테스트 고객 데이터
- `data/hyundai_card_customers.csv` — 전체 합성 데이터

### 3. Counterfactual 생성 (데모)

```bash
# 거절 고객 10명 대상 CF 생성
python run_demo.py --n_customers 10

# 특정 고객 ID 조회
python run_demo.py --customer_id HCC_00042
```

결과:
- 콘솔: 고객별 반사실적 보고서 (Treatment 권고)
- `results/cf_results.csv` — 상세 결과
- `results/cf_report.png` — 시각화

---

## 피처 정의

| 피처 | 설명 | 제약 |
|------|------|------|
| age | 나이 | 불변 |
| annual_income | 연소득 (만원) | 증가만 허용 |
| credit_score | 신용점수 (300~900) | 증가만 허용 |
| num_existing_cards | 보유 카드 수 | 자유 |
| monthly_spending | 월 카드 사용금액 (만원) | 증가만 허용 |
| years_as_customer | 현대카드 거래 연수 | 불변 |
| total_loan_amount | 총 대출금액 (만원) | 불변 |
| monthly_transactions | 월 거래 건수 | 증가만 허용 |
| num_delinquencies | 연체 횟수 | 감소만 허용 |
| utilization_rate | 신용 한도 사용률 (%) | 감소만 허용 |
| marital_status | 혼인 상태 | 불변 |
| employment_type | 고용 형태 | 자유 |
| education_level | 교육 수준 | 불변 |
| region | 거주 지역 | 불변 |

---

## 평가 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| **Validity** | CF가 목표 클래스(1)로 예측되는 비율 | 높을수록 좋음 ↑ |
| **Proximity** | CF와 원본 간 L2 거리 (정규화 공간) | 낮을수록 좋음 ↓ |
| **Sparsity** | 변경된 피처 비율 | 낮을수록 좋음 ↓ |

---

## 마케팅 활용 방안

생성된 Treatment를 기반으로 다음 캠페인 기획 가능:

- **신용점수 개선 타겟**: 크레딧 빌딩 상품 교차 판매
- **사용 실적 증가 타겟**: 포인트/캐시백 강화 캠페인
- **연체 관리 타겟**: 납기일 알림, 자동이체 유도
- **한도 사용률 타겟**: 한도 상향 가이드 및 분산 사용 유도
