"""
TabDiff 기반 Counterfactual Explanation 생성기

논문 알고리즘 구현:
  Input : 원본 고객 데이터 x_factual (클래스 0 — 추가 발급 거절)
  Output: 반사실적 x_cf (클래스 1 — 추가 발급 승인)으로 변경하는 최소한의 변화

알고리즘 (논문 Algorithm 1 응용):
  1. x_factual을 인코딩
  2. 부분 노이즈 추가: x_{T_cf} ~ q(x_{T_cf} | x_factual)
  3. t = T_cf, ..., 1 역확산:
       x_{t-1} = DDPM_reverse(x_t, t)
                 + λ * ∇_{x_t} log p(y=1 | x_t)  [classifier guidance]
       x_{t-1} = apply_constraints(x_{t-1}, x_factual)  [행동 가능성]
  4. x_0 디코딩 → 원래 스케일
  5. 후보 중 proximity 최소 + validity 최대 선택
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict
from tqdm import tqdm

from config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, CATEGORICAL_VALUES,
    IMMUTABLE_FEATURES, INCREASING_ONLY_FEATURES, DECREASING_ONLY_FEATURES,
    CF_CONFIG,
)
from cf_engine.constraints import (
    ActionabilityConstraints,
    compute_proximity,
    compute_sparsity,
    compute_validity,
)


class TabDiffCFGenerator:
    """
    TabDiff 기반 반사실적 설명 생성기

    마케팅 활용:
    - 추가 카드 발급 거절 고객(class 0)에 대해
    - 최소한의 변화로 승인(class 1) 받을 수 있는 시나리오 생성
    - 각 고객별 맞춤형 Treatment 권고안 제시
    """

    def __init__(
        self,
        tabdiff_model,
        classifier,
        preprocessor,
        device: str = "cpu",
        cf_config: Optional[Dict] = None,
    ):
        self.tabdiff = tabdiff_model.to(device)
        self.tabdiff.eval()
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.preprocessor = preprocessor
        self.device = device
        self.cfg = cf_config or CF_CONFIG

        # 행동 가능성 제약
        self.constraints = ActionabilityConstraints(preprocessor)

    def _guidance_fn(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Classifier guidance: log P(y=1 | x_t)
        TabDiff 역확산 방향을 클래스 1(추가 발급 승인)로 유도
        """
        return self.classifier.log_prob_target(x_t)

    def generate_for_customer(
        self,
        customer_row: pd.Series,
        num_candidates: int = 5,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        단일 고객에 대한 반사실적 후보 생성

        Args:
            customer_row: 고객 데이터 (pd.Series, 피처만 포함)
            num_candidates: 생성할 후보 수
            verbose: 진행 상황 출력

        Returns:
            cf_candidates: 반사실적 후보 DataFrame (num_candidates 행)
        """
        # 인코딩
        x_factual_np = self.preprocessor.transform(
            pd.DataFrame([customer_row[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]])
        )  # (1, total_dim)

        # 후보 생성을 위해 배치 복제
        x_factual = torch.tensor(
            np.repeat(x_factual_np, num_candidates, axis=0),
            dtype=torch.float32,
            device=self.device,
        )  # (num_candidates, total_dim)

        t_start = self.cfg["num_cf_timesteps"]
        guidance_scale = self.cfg["guidance_scale"]

        # 반사실적 역확산 생성
        x_cf = self.tabdiff.generate_counterfactual_trajectory(
            x_factual=x_factual,
            t_start=t_start,
            guidance_fn=self._guidance_fn,
            guidance_scale=guidance_scale,
            constraint_fn=lambda xc, xf: self.constraints.apply(xc, xf),
        )  # (num_candidates, total_dim)

        # 디코딩 → 원래 스케일
        x_cf_np = x_cf.cpu().numpy()
        cf_df = self.preprocessor.inverse_transform(x_cf_np)

        return cf_df

    def select_best_cf(
        self,
        cf_candidates: pd.DataFrame,
        factual_row: pd.Series,
    ) -> pd.Series:
        """
        후보 반사실적 중 가장 좋은 것 선택
        기준: 분류기 신뢰도 최대 + 원본과의 거리 최소
        """
        x_factual_np = self.preprocessor.transform(
            pd.DataFrame([factual_row[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]])
        )
        x_cf_np = self.preprocessor.transform(
            cf_candidates[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        )

        # 분류기 예측 확률
        x_cf_tensor = torch.tensor(x_cf_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            cf_probs = self.classifier.predict_proba(x_cf_tensor).cpu().numpy()

        # 유효한 CF만 (p(y=1) > 0.5)
        valid_mask = cf_probs > 0.5
        if valid_mask.any():
            valid_idx = np.where(valid_mask)[0]
            # 유효한 것 중 원본과 거리 최소
            dists = np.linalg.norm(
                x_cf_np[valid_idx, :self.preprocessor.num_dim]
                - x_factual_np[:, :self.preprocessor.num_dim],
                axis=1,
            )
            best_idx = valid_idx[np.argmin(dists)]
        else:
            # 유효한 CF가 없으면 확률 최대인 것 선택
            best_idx = int(np.argmax(cf_probs))

        return cf_candidates.iloc[best_idx]

    def explain(
        self,
        customer_df: pd.DataFrame,
        max_customers: int = 20,
        verbose: bool = True,
    ) -> List[Dict]:
        """
        여러 고객에 대한 반사실적 설명 일괄 생성

        추가 발급 거절(0) 고객들에 대해 승인(1)을 위한
        맞춤형 Treatment 권고안 생성

        Args:
            customer_df: 고객 데이터 DataFrame (타겟 컬럼 포함)
            max_customers: 최대 처리 고객 수
            verbose: 진행 상황 출력

        Returns:
            results: 고객별 설명 결과 리스트
        """
        from config import TARGET_COLUMN

        # 추가 발급 거절 고객 필터링
        if TARGET_COLUMN in customer_df.columns:
            rejected = customer_df[customer_df[TARGET_COLUMN] == 0].copy()
        else:
            rejected = customer_df.copy()

        rejected = rejected.head(max_customers)

        # 분류기로 거절 확률 확인 (이중 확인)
        X_enc = self.preprocessor.transform(rejected[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
        X_tensor = torch.tensor(X_enc, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            reject_probs = self.classifier.predict_proba(X_tensor).cpu().numpy()

        results = []
        iterator = tqdm(range(len(rejected)), desc="CF 생성 중") if verbose else range(len(rejected))

        for i in iterator:
            row = rejected.iloc[i]
            customer_id = row.get("customer_id", f"고객_{i+1}")

            # 반사실적 후보 생성
            cf_candidates = self.generate_for_customer(
                row,
                num_candidates=self.cfg["num_cf_samples"],
            )

            # 최적 CF 선택
            best_cf = self.select_best_cf(cf_candidates, row)

            # CF 예측 확률
            x_cf_enc = self.preprocessor.transform(
                pd.DataFrame([best_cf[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]])
            )
            x_cf_tensor = torch.tensor(x_cf_enc, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                cf_prob = float(self.classifier.predict_proba(x_cf_tensor).cpu().numpy()[0])

            # 변화된 피처 분석 (Treatment 도출)
            treatments = self._extract_treatments(row, best_cf)

            results.append({
                "customer_id": customer_id,
                "factual": row,
                "counterfactual": best_cf,
                "factual_prob": float(reject_probs[i]),
                "cf_prob": cf_prob,
                "cf_valid": cf_prob > 0.5,
                "treatments": treatments,
                "num_changes": len(treatments),
            })

        return results

    def _extract_treatments(
        self,
        factual: pd.Series,
        counterfactual: pd.Series,
    ) -> List[Dict]:
        """
        원본과 반사실적의 차이에서 Treatment 권고안 추출

        불변 피처와 미미한 변화는 제외하고
        실질적인 변화만 Treatment로 제시
        """
        treatments = []
        threshold_num = 0.01  # 정규화 공간에서의 최소 변화량

        # 수치형 피처 비교
        for feat in NUMERICAL_FEATURES:
            if feat in IMMUTABLE_FEATURES:
                continue
            orig_val = float(factual[feat])
            cf_val = float(counterfactual[feat])
            if abs(cf_val - orig_val) < 1e-6:
                continue

            # 정규화 공간에서의 변화량 계산
            scaler = self.preprocessor.num_scaler
            feat_idx = NUMERICAL_FEATURES.index(feat)
            orig_scaled = (orig_val - scaler.mean_[feat_idx]) / scaler.scale_[feat_idx]
            cf_scaled = (cf_val - scaler.mean_[feat_idx]) / scaler.scale_[feat_idx]
            if abs(cf_scaled - orig_scaled) < threshold_num:
                continue

            # 변화 방향 검증 (단조 제약 위반 방지)
            if feat in INCREASING_ONLY_FEATURES and cf_val < orig_val:
                continue
            if feat in DECREASING_ONLY_FEATURES and cf_val > orig_val:
                continue

            delta = cf_val - orig_val
            pct_change = (delta / orig_val * 100) if orig_val != 0 else float("inf")

            treatments.append({
                "feature": feat,
                "feature_kr": _feat_name_kr(feat),
                "original": orig_val,
                "counterfactual": cf_val,
                "delta": delta,
                "pct_change": pct_change,
                "direction": "↑" if delta > 0 else "↓",
                "type": "numerical",
                "unit": _feat_unit(feat),
            })

        # 범주형 피처 비교
        for feat in CATEGORICAL_FEATURES:
            if feat in IMMUTABLE_FEATURES:
                continue
            orig_val = str(factual[feat])
            cf_val = str(counterfactual[feat])
            if orig_val == cf_val:
                continue

            treatments.append({
                "feature": feat,
                "feature_kr": _feat_name_kr(feat),
                "original": orig_val,
                "counterfactual": cf_val,
                "delta": None,
                "pct_change": None,
                "direction": "→",
                "type": "categorical",
                "unit": "",
            })

        # 변화량 크기 순 정렬
        num_treatments = [t for t in treatments if t["type"] == "numerical"]
        cat_treatments = [t for t in treatments if t["type"] == "categorical"]
        num_treatments.sort(key=lambda x: abs(x["pct_change"]) if x["pct_change"] else 0, reverse=True)

        return num_treatments + cat_treatments


# ─── 한국어 피처명 매핑 ─────────────────────────────────────────────────────────

def _feat_name_kr(feat: str) -> str:
    mapping = {
        "age": "나이",
        "annual_income": "연소득",
        "credit_score": "신용점수",
        "num_existing_cards": "보유 카드 수",
        "monthly_spending": "월 카드 사용금액",
        "years_as_customer": "거래 연수",
        "total_loan_amount": "총 대출금액",
        "monthly_transactions": "월 거래 건수",
        "num_delinquencies": "연체 횟수",
        "utilization_rate": "신용 한도 사용률",
        "marital_status": "혼인 상태",
        "employment_type": "고용 형태",
        "education_level": "교육 수준",
        "region": "거주 지역",
    }
    return mapping.get(feat, feat)


def _feat_unit(feat: str) -> str:
    units = {
        "annual_income": "만원",
        "monthly_spending": "만원",
        "total_loan_amount": "만원",
        "credit_score": "점",
        "utilization_rate": "%",
        "monthly_transactions": "건",
        "num_delinquencies": "회",
        "num_existing_cards": "장",
        "years_as_customer": "년",
        "age": "세",
    }
    return units.get(feat, "")


def print_cf_report(result: Dict, show_all_features: bool = False) -> None:
    """고객별 반사실적 설명 보고서 출력"""
    cid = result["customer_id"]
    factual_prob = result["factual_prob"]
    cf_prob = result["cf_prob"]
    cf_valid = result["cf_valid"]
    treatments = result["treatments"]

    print("\n" + "=" * 65)
    print(f"  고객 ID: {cid}")
    print("=" * 65)
    print(f"  현재 예측: {'추가 발급 거절' if factual_prob < 0.5 else '추가 발급 승인'} "
          f"(확률: {factual_prob*100:.1f}%)")
    print()

    if treatments:
        status = "✓ 유효" if cf_valid else "✗ 미달성"
        print(f"  [반사실적 시나리오 — {status}]")
        print(f"  목표 예측: 추가 발급 승인 (확률: {cf_prob*100:.1f}%)")
        print()
        print(f"  {'피처':<18} {'현재값':>12} {'목표값':>12} {'변화':>15}")
        print("  " + "─" * 60)

        for t in treatments:
            feat_kr = t["feature_kr"]
            unit = t["unit"]
            orig = t["original"]
            cf = t["counterfactual"]
            direction = t["direction"]

            if t["type"] == "numerical":
                orig_str = f"{orig:,.0f}{unit}" if unit else f"{orig:.2f}"
                cf_str = f"{cf:,.0f}{unit}" if unit else f"{cf:.2f}"
                delta = t["delta"]
                pct = t["pct_change"]
                change_str = f"{direction}{abs(delta):,.0f} ({abs(pct):.0f}%)"
            else:
                orig_str = _translate_cat(t["feature"], str(orig))
                cf_str = _translate_cat(t["feature"], str(cf))
                change_str = f"{direction}"

            print(f"  {feat_kr:<18} {orig_str:>12} {cf_str:>12} {change_str:>15}")

        print()
        print(f"  [마케팅 인사이트]")
        _print_marketing_insight(treatments)
    else:
        print("  [변화 없음] 현재 상태로도 조건 충족 가능")

    print("=" * 65)


def _translate_cat(feat: str, val: str) -> str:
    """범주형 값 한국어 번역"""
    trans = {
        "marital_status": {"single": "미혼", "married": "기혼", "divorced": "이혼/별거"},
        "employment_type": {"employed": "재직자", "self_employed": "자영업", "unemployed": "무직", "retired": "퇴직자"},
        "education_level": {"high_school": "고졸", "college": "대졸", "graduate": "대학원졸"},
        "region": {"Seoul": "서울", "Gyeonggi": "경기", "Busan": "부산", "Others": "기타"},
    }
    return trans.get(feat, {}).get(val, val)


def _print_marketing_insight(treatments: List[Dict]) -> None:
    """Treatment 기반 마케팅 액션 권고"""
    insights = []
    for t in treatments:
        feat = t["feature"]
        direction = t["direction"]

        if feat == "credit_score" and direction == "↑":
            insights.append("  → 신용점수 개선 프로그램 안내 (크레딧 빌딩 상품)")
        elif feat == "monthly_spending" and direction == "↑":
            insights.append("  → 카드 사용 실적 증가 혜택 캠페인 타겟팅")
        elif feat == "monthly_transactions" and direction == "↑":
            insights.append("  → 소액 결제 혜택(포인트/캐시백) 강화 마케팅")
        elif feat == "annual_income" and direction == "↑":
            insights.append("  → 소득 증가 시 자동 한도 상향 안내")
        elif feat == "num_delinquencies" and direction == "↓":
            insights.append("  → 연체 관리 지원 서비스 안내 (납기일 알림 등)")
        elif feat == "utilization_rate" and direction == "↓":
            insights.append("  → 신용 한도 사용률 낮추기 가이드 제공")
        elif feat == "employment_type":
            insights.append("  → 고용 상태 변화 시 재심사 안내")

    if insights:
        for ins in insights[:3]:  # 상위 3개만
            print(ins)
    else:
        print("  → 현재 프로필 기반 맞춤형 카드 혜택 안내 권고")
