"""
TabDiff 기반 무실적 위험 고객 Wake-up Counterfactual 생성기

[Use Case]
- 대상: inactive_risk = 1로 예측된 고객 (무실적 위험군)
- 목표: inactive_risk = 0 (활성 고객) 상태로 전환하기 위한 최소 변화 도출
- 출력: 고객별 맞춤형 Wake-up Treatment 권고안

[알고리즘]
1. x_factual(무실적 위험 고객) 인코딩
2. x_{T_cf} = q(x_{T_cf} | x_factual)  ← 부분 노이즈 추가
3. for t = T_cf, ..., 1:
     ε̃ = ε_θ(x_t, t) - λ√(1-ᾱ_t) · ∇_{x_t} log P(y=0|x_t)   ← 클래스 0(활성) 방향 유도
     x_{t-1} = DDPM_reverse(x_t, ε̃)
     x_{t-1} = apply_constraints(x_{t-1}, x_factual)
4. 디코딩 → 원래 스케일
5. 후보 중 validity 최대 + proximity 최소 선택
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Optional, Dict
from tqdm import tqdm

from config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, CATEGORICAL_VALUES,
    IMMUTABLE_FEATURES, INCREASING_ONLY_FEATURES, DECREASING_ONLY_FEATURES,
    CF_CONFIG, TARGET_COLUMN,
)
from cf_engine.constraints import (
    ActionabilityConstraints,
    compute_proximity,
    compute_sparsity,
)


class TabDiffCFGenerator:
    """
    TabDiff 기반 무실적 위험 고객 Wake-up Counterfactual 생성기

    목표 클래스: inactive_risk = 0 (활성 고객)
    → classifier guidance: ∇ log P(y=0|x_t) = -∇ log P(y=1|x_t) 방향으로 유도
    """

    def __init__(self, tabdiff_model, classifier, preprocessor,
                 device: str = "cpu", cf_config: Optional[Dict] = None):
        self.tabdiff = tabdiff_model.to(device)
        self.tabdiff.eval()
        self.classifier = classifier.to(device)
        self.classifier.eval()
        self.preprocessor = preprocessor
        self.device = device
        self.cfg = cf_config or CF_CONFIG
        self.constraints = ActionabilityConstraints(preprocessor)

    def _guidance_fn(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Classifier guidance: log P(y=0 | x_t) = log(1 - σ(f(x_t)))
        목표 클래스 0(활성 고객) 방향으로 역확산 유도
        """
        import torch.nn.functional as F
        logit = self.classifier(x_t)          # (batch,)
        # log P(y=0|x) = log(1 - sigmoid(logit)) = log sigmoid(-logit)
        return F.logsigmoid(-logit)

    def generate_for_customer(self, customer_row: pd.Series,
                               num_candidates: int = 5) -> pd.DataFrame:
        """단일 고객에 대한 반사실적 후보 생성"""
        x_factual_np = self.preprocessor.transform(
            pd.DataFrame([customer_row[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]])
        )
        x_factual = torch.tensor(
            np.repeat(x_factual_np, num_candidates, axis=0),
            dtype=torch.float32,
            device=self.device,
        )

        x_cf = self.tabdiff.generate_counterfactual_trajectory(
            x_factual=x_factual,
            t_start=self.cfg["num_cf_timesteps"],
            guidance_fn=self._guidance_fn,
            guidance_scale=self.cfg["guidance_scale"],
            constraint_fn=lambda xc, xf: self.constraints.apply(xc, xf),
        )

        return self.preprocessor.inverse_transform(x_cf.cpu().numpy())

    def generate_for_batch(
        self,
        customer_df: pd.DataFrame,
        num_candidates: int = 2,
    ) -> List[pd.DataFrame]:
        """
        여러 고객을 한 번의 diffusion pass로 처리 (대규모 처리 최적화)

        전략: 각 고객을 num_candidates개 복제 → 단일 대형 배치로 역확산
        처리량: 고객당 개별 처리 대비 N배 빠름 (GPU 환경에서 효과 극대화)

        Args:
            customer_df:    고객 DataFrame (n행)
            num_candidates: 고객당 후보 CF 수 (대규모 시 2개 권장)
        Returns:
            cf_list: 고객별 CF 후보 DataFrame 리스트 (n개)
        """
        n = len(customer_df)
        x_factual_np = self.preprocessor.transform(
            customer_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        )  # (n, dim)

        # 각 고객을 num_candidates번 복제 → [c1,c1, c2,c2, ...] 순서
        x_factual_repeated = np.repeat(x_factual_np, num_candidates, axis=0)  # (n*k, dim)
        x_factual_t = torch.tensor(x_factual_repeated, dtype=torch.float32, device=self.device)

        # 단일 대형 배치로 역확산 수행
        x_cf_all = self.tabdiff.generate_counterfactual_trajectory(
            x_factual=x_factual_t,
            t_start=self.cfg["num_cf_timesteps"],
            guidance_fn=self._guidance_fn,
            guidance_scale=self.cfg["guidance_scale"],
            constraint_fn=lambda xc, xf: self.constraints.apply(xc, xf),
        )  # (n*k, dim)

        x_cf_np = x_cf_all.cpu().numpy()

        # 고객별 슬라이스로 분리 → 개별 DataFrame
        cf_list = []
        for i in range(n):
            start = i * num_candidates
            end   = start + num_candidates
            cf_list.append(
                self.preprocessor.inverse_transform(x_cf_np[start:end])
            )
        return cf_list

    def select_best_cf(self, cf_candidates: pd.DataFrame,
                        factual_row: pd.Series) -> pd.Series:
        """
        후보 중 최적 CF 선택
        기준: 분류기가 활성(0)으로 예측 + 원본과 거리 최소
        """
        x_factual_np = self.preprocessor.transform(
            pd.DataFrame([factual_row[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]])
        )
        x_cf_np = self.preprocessor.transform(
            cf_candidates[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        )
        x_cf_tensor = torch.tensor(x_cf_np, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            cf_probs = self.classifier.predict_proba(x_cf_tensor).cpu().numpy()

        # 활성 고객 예측: prob < 0.5 (inactive_risk=0)
        valid_mask = cf_probs < 0.5
        if valid_mask.any():
            valid_idx = np.where(valid_mask)[0]
            dists = np.linalg.norm(
                x_cf_np[valid_idx, :self.preprocessor.num_dim]
                - x_factual_np[:, :self.preprocessor.num_dim],
                axis=1,
            )
            best_idx = valid_idx[np.argmin(dists)]
        else:
            # 유효한 CF 없으면 확률 최소(가장 활성에 가까운) 선택
            best_idx = int(np.argmin(cf_probs))

        return cf_candidates.iloc[best_idx]

    def explain(self, customer_df: pd.DataFrame,
                max_customers: int = 20, verbose: bool = True) -> List[Dict]:
        """
        무실적 위험 고객 일괄 CF 설명 생성

        Args:
            customer_df: 고객 데이터 (타겟 컬럼 포함)
            max_customers: 최대 처리 고객 수
        Returns:
            results: 고객별 Wake-up 설명 리스트
        """
        if TARGET_COLUMN in customer_df.columns:
            target_df = customer_df[customer_df[TARGET_COLUMN] == 1].copy()
        else:
            target_df = customer_df.copy()
        target_df = target_df.head(max_customers)

        X_enc = self.preprocessor.transform(
            target_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        )
        X_tensor = torch.tensor(X_enc, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            risk_probs = self.classifier.predict_proba(X_tensor).cpu().numpy()

        results = []
        iterator = tqdm(range(len(target_df)), desc="CF 생성 중") if verbose else range(len(target_df))

        for i in iterator:
            row = target_df.iloc[i]
            customer_id = row.get("customer_id", f"고객_{i+1}")

            cf_candidates = self.generate_for_customer(
                row, num_candidates=self.cfg["num_cf_samples"]
            )
            best_cf = self.select_best_cf(cf_candidates, row)

            x_cf_enc = self.preprocessor.transform(
                pd.DataFrame([best_cf[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]])
            )
            x_cf_tensor = torch.tensor(x_cf_enc, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                cf_prob = float(self.classifier.predict_proba(x_cf_tensor).cpu().numpy()[0])

            # CF 유효: inactive_risk=0 (활성) 예측
            cf_valid = cf_prob < 0.5

            treatments = self._extract_treatments(row, best_cf)

            results.append({
                "customer_id":   customer_id,
                "factual":       row,
                "counterfactual": best_cf,
                "factual_prob":  float(risk_probs[i]),   # P(inactive_risk=1)
                "cf_prob":       cf_prob,                 # P(inactive_risk=1) after CF
                "cf_valid":      cf_valid,
                "treatments":    treatments,
                "num_changes":   len(treatments),
            })

        return results

    def _extract_treatments(self, factual: pd.Series,
                             counterfactual: pd.Series) -> List[Dict]:
        """원본 vs CF 비교 → Wake-up Treatment 도출"""
        treatments = []
        threshold_num = 0.03

        for feat in NUMERICAL_FEATURES:
            if feat in IMMUTABLE_FEATURES:
                continue
            orig_val = float(factual[feat])
            cf_val   = float(counterfactual[feat])
            if abs(cf_val - orig_val) < 1e-6:
                continue

            scaler = self.preprocessor.num_scaler
            idx = NUMERICAL_FEATURES.index(feat)
            orig_s = (orig_val - scaler.mean_[idx]) / scaler.scale_[idx]
            cf_s   = (cf_val   - scaler.mean_[idx]) / scaler.scale_[idx]
            if abs(cf_s - orig_s) < threshold_num:
                continue

            # 단조 제약 위반 제외
            if feat in INCREASING_ONLY_FEATURES and cf_val < orig_val:
                continue
            if feat in DECREASING_ONLY_FEATURES and cf_val > orig_val:
                continue

            delta = cf_val - orig_val
            pct   = (delta / orig_val * 100) if orig_val != 0 else float("inf")

            treatments.append({
                "feature":         feat,
                "feature_kr":      _feat_name_kr(feat),
                "original":        orig_val,
                "counterfactual":  cf_val,
                "delta":           delta,
                "pct_change":      pct,
                "direction":       "↑" if delta > 0 else "↓",
                "type":            "numerical",
                "unit":            _feat_unit(feat),
            })

        for feat in CATEGORICAL_FEATURES:
            if feat in IMMUTABLE_FEATURES:
                continue
            orig_val = str(factual[feat])
            cf_val   = str(counterfactual[feat])
            if orig_val == cf_val:
                continue
            treatments.append({
                "feature":         feat,
                "feature_kr":      _feat_name_kr(feat),
                "original":        orig_val,
                "counterfactual":  cf_val,
                "delta":           None,
                "pct_change":      None,
                "direction":       "→",
                "type":            "categorical",
                "unit":            "",
            })

        num_t = [t for t in treatments if t["type"] == "numerical"]
        cat_t = [t for t in treatments if t["type"] == "categorical"]
        num_t.sort(key=lambda x: abs(x["pct_change"]) if x["pct_change"] else 0, reverse=True)
        return num_t + cat_t


# ─── 한국어 피처명·단위 매핑 ────────────────────────────────────────────────────

def _feat_name_kr(feat: str) -> str:
    mapping = {
        "months_since_last_txn":      "마지막 거래 경과",
        "avg_spending_3m":             "최근 3M 월 사용금액",
        "avg_spending_prev3m":         "이전 3M 월 사용금액",
        "spending_trend_ratio":        "사용금액 트렌드",
        "monthly_txn_count_3m":        "최근 3M 월 거래 건수",
        "num_missed_months":           "무실적 월 수",
        "days_since_app_login":        "앱 미사용 기간",
        "loyalty_points_balance":      "미사용 포인트 잔액",
        "num_active_benefits":         "활성 혜택 수",
        "years_as_customer":           "거래 연수",
        "card_tier":                   "카드 등급",
        "primary_spending_category":   "주요 사용 카테고리",
        "payment_method":              "결제 수단",
        "engagement_level":            "디지털 참여도",
    }
    return mapping.get(feat, feat)


def _feat_unit(feat: str) -> str:
    units = {
        "months_since_last_txn":   "개월",
        "avg_spending_3m":         "만원",
        "avg_spending_prev3m":     "만원",
        "spending_trend_ratio":    "배",
        "monthly_txn_count_3m":    "건",
        "num_missed_months":       "개월",
        "days_since_app_login":    "일",
        "loyalty_points_balance":  "P",
        "num_active_benefits":     "개",
        "years_as_customer":       "년",
    }
    return units.get(feat, "")


def _translate_cat(feat: str, val: str) -> str:
    trans = {
        "card_tier":                 {"The": "The Card", "Black": "Black", "Red": "Red", "Blue": "Blue"},
        "primary_spending_category": {"dining": "외식", "shopping": "쇼핑", "travel": "여행",
                                      "convenience": "편의점", "online": "온라인"},
        "payment_method":            {"app": "앱결제", "online": "온라인결제",
                                      "offline_nfc": "오프라인NFC", "offline_swipe": "오프라인마그네틱"},
        "engagement_level":          {"high": "높음", "medium": "보통", "low": "낮음"},
    }
    return trans.get(feat, {}).get(val, val)


def print_cf_report(result: Dict) -> None:
    """고객별 Wake-up Counterfactual 보고서 출력"""
    cid          = result["customer_id"]
    risk_prob    = result["factual_prob"]      # P(inactive_risk=1) — 원본
    cf_risk_prob = result["cf_prob"]           # P(inactive_risk=1) — CF 적용 후
    cf_valid     = result["cf_valid"]
    treatments   = result["treatments"]

    # 카드 등급·거래 연수 등 참고 정보 추출
    factual = result["factual"]
    tier     = factual.get("card_tier", "-")
    yrs      = factual.get("years_as_customer", 0)
    missed   = factual.get("num_missed_months", 0)
    days_app = factual.get("days_since_app_login", 0)

    print("\n" + "=" * 68)
    print(f"  고객 ID: {cid}  |  카드 등급: {tier}  |  거래 연수: {yrs:.1f}년")
    print("=" * 68)
    print(f"  현재 무실적 위험도: {risk_prob*100:.1f}%  "
          f"(최근 무실적 {int(missed)}개월, 앱 미사용 {int(days_app)}일)")
    print()

    if not treatments:
        print("  [변화 없음] 현재 상태로도 활성 전환 가능")
    else:
        status = "✓ Wake-up 달성" if cf_valid else "✗ 미달성 (추가 개입 필요)"
        print(f"  [Wake-up 시나리오 — {status}]")
        print(f"  목표 달성 시 무실적 위험도: {cf_risk_prob*100:.1f}%  "
              f"(↓{(risk_prob - cf_risk_prob)*100:.1f}%p 감소)")
        print()
        print(f"  {'피처':<22} {'현재값':>13} {'목표값':>13} {'변화':>16}")
        print("  " + "─" * 66)

        for t in treatments:
            feat_kr   = t["feature_kr"]
            unit      = t["unit"]
            orig      = t["original"]
            cf        = t["counterfactual"]
            direction = t["direction"]

            if t["type"] == "numerical":
                if unit in ("만원", "P"):
                    orig_str = f"{orig:,.0f}{unit}"
                    cf_str   = f"{cf:,.0f}{unit}"
                elif unit in ("배",):
                    orig_str = f"{orig:.2f}{unit}"
                    cf_str   = f"{cf:.2f}{unit}"
                else:
                    orig_str = f"{orig:.1f}{unit}"
                    cf_str   = f"{cf:.1f}{unit}"
                pct = t["pct_change"]
                delta_abs = abs(t["delta"])
                if unit in ("만원", "P"):
                    change_str = f"{direction}{delta_abs:,.0f} ({abs(pct):.0f}%)"
                elif unit in ("배",):
                    change_str = f"{direction}{delta_abs:.2f} ({abs(pct):.0f}%)"
                else:
                    change_str = f"{direction}{delta_abs:.1f} ({abs(pct):.0f}%)"
            else:
                orig_str   = _translate_cat(t["feature"], str(orig))
                cf_str     = _translate_cat(t["feature"], str(cf))
                change_str = f"{direction}"

            print(f"  {feat_kr:<22} {orig_str:>13} {cf_str:>13} {change_str:>16}")

        print()
        print("  [Wake-up 캠페인 액션 권고]")
        _print_wakeup_actions(treatments, factual)

    print("=" * 68)


def _print_wakeup_actions(treatments: List[Dict], factual: pd.Series) -> None:
    """Treatment 기반 구체적인 Wake-up 캠페인 액션 출력"""
    actions = []
    for t in treatments:
        feat      = t["feature"]
        direction = t["direction"]

        if feat == "days_since_app_login" and direction == "↓":
            days = factual.get("days_since_app_login", 0)
            actions.append(f"  → [앱 재참여] 푸시 알림 발송 / {int(days)}일 미접속 특별 혜택 안내")
        elif feat == "avg_spending_3m" and direction == "↑":
            actions.append("  → [사용 실적] 월 사용금액 목표 달성 시 캐시백/포인트 추가 적립 캠페인")
        elif feat == "monthly_txn_count_3m" and direction == "↑":
            actions.append("  → [결제 빈도] 소액 결제 N건당 포인트 지급 미션 이벤트")
        elif feat == "num_missed_months" and direction == "↓":
            actions.append("  → [연속 사용] 연속 이용 개월 수에 따른 보너스 포인트 리워드")
        elif feat == "num_active_benefits" and direction == "↑":
            pnt = factual.get("loyalty_points_balance", 0)
            actions.append(f"  → [혜택 등록] 미가입 서비스 안내 / 포인트 {int(pnt):,}P 활용 혜택 추천")
        elif feat == "loyalty_points_balance" and direction == "↓":
            actions.append("  → [포인트 소진] 포인트 만료 예정 알림 / 포인트 사용처 추천")
        elif feat == "spending_trend_ratio" and direction == "↑":
            actions.append("  → [트렌드 반전] 전월 대비 사용 증가 시 보너스 혜택 제공")
        elif feat == "engagement_level":
            cf_val = t["counterfactual"]
            actions.append(f"  → [디지털 참여] 앱 전용 이벤트 / 간편결제 전환 유도 ({cf_val} 수준 목표)")
        elif feat == "payment_method":
            cf_val = _translate_cat("payment_method", str(t["counterfactual"]))
            actions.append(f"  → [결제 수단] {cf_val} 전환 시 추가 적립 혜택 안내")
        elif feat == "primary_spending_category":
            cf_val = _translate_cat("primary_spending_category", str(t["counterfactual"]))
            actions.append(f"  → [카테고리 확장] {cf_val} 카테고리 특별 할인/포인트 캠페인")

    seen = set()
    for a in actions:
        if a not in seen:
            print(a)
            seen.add(a)
        if len(seen) >= 3:
            break


# ─── 대규모 처리용 결과 직렬화 헬퍼 ────────────────────────────────────────────

ACTION_TAG = {
    "days_since_app_login":      "앱재참여",
    "avg_spending_3m":           "사용실적증대",
    "monthly_txn_count_3m":      "결제빈도증가",
    "num_missed_months":         "연속사용유지",
    "num_active_benefits":       "혜택등록유도",
    "loyalty_points_balance":    "포인트소진",
    "spending_trend_ratio":      "트렌드반전",
    "months_since_last_txn":     "거래재개",
    "engagement_level":          "디지털참여강화",
    "payment_method":            "결제수단전환",
    "primary_spending_category": "카테고리확장",
}


def build_treatment_rows(result: dict) -> dict:
    """
    CF 결과 dict → 와이드 포맷 행(row)
    고객 × 피처별 변화(orig/cf/delta/pct) + 캠페인 액션 태그
    """
    row = {
        "customer_id":  result["customer_id"],
        "risk_prob":    round(result["factual_prob"], 4),
        "cf_risk_prob": round(result["cf_prob"], 4),
        "risk_drop":    round(result["factual_prob"] - result["cf_prob"], 4),
        "cf_valid":     result["cf_valid"],
        "num_changes":  result["num_changes"],
        "card_tier":         result["factual"].get("card_tier", "-"),
        "years_as_customer": result["factual"].get("years_as_customer", 0),
        "engagement_level":  result["factual"].get("engagement_level", "-"),
    }
    for feat in NUMERICAL_FEATURES:
        orig = float(result["factual"].get(feat, 0))
        cf   = float(result["counterfactual"].get(feat, 0))
        row[f"{feat}_orig"]  = orig
        row[f"{feat}_cf"]    = cf
        row[f"{feat}_delta"] = round(cf - orig, 2)
        row[f"{feat}_pct"]   = round((cf - orig) / orig * 100, 1) if orig != 0 else 0.0
    for feat in CATEGORICAL_FEATURES:
        row[f"{feat}_orig"]    = str(result["factual"].get(feat, ""))
        row[f"{feat}_cf"]      = str(result["counterfactual"].get(feat, ""))
        row[f"{feat}_changed"] = int(row[f"{feat}_orig"] != row[f"{feat}_cf"])
    action_tags = [ACTION_TAG.get(t["feature"], t["feature"]) for t in result["treatments"]]
    row["action_tags"] = " | ".join(action_tags)
    row["num_actions"] = len(action_tags)
    return row


def build_action_rows(result: dict) -> list:
    """
    CF 결과 dict → 롱 포맷 행 리스트 (고객 × treatment 1행)
    """
    rows = []
    for t in result["treatments"]:
        feat = t["feature"]
        unit = _feat_unit(feat)
        if t["type"] == "numerical":
            orig_str = f"{t['original']:.1f}{unit}"
            cf_str   = f"{t['counterfactual']:.1f}{unit}"
            change   = f"{t['direction']}{abs(t['delta']):.1f} ({abs(t['pct_change']):.0f}%)"
        else:
            orig_str = _translate_cat(feat, str(t["original"]))
            cf_str   = _translate_cat(feat, str(t["counterfactual"]))
            change   = f"→ {cf_str}"
        rows.append({
            "customer_id":  result["customer_id"],
            "risk_prob":    round(result["factual_prob"], 4),
            "cf_risk_prob": round(result["cf_prob"], 4),
            "cf_valid":     result["cf_valid"],
            "card_tier":    result["factual"].get("card_tier", "-"),
            "feature":      feat,
            "feature_kr":   _feat_name_kr(feat),
            "action_tag":   ACTION_TAG.get(feat, feat),
            "original":     orig_str,
            "target":       cf_str,
            "change":       change,
            "direction":    t["direction"],
            "pct_change":   t.get("pct_change"),
            "type":         t["type"],
        })
    return rows
