"""
CF 품질 평가 지표 — 4대 핵심 지표

  1. Validity Rate   : CF가 분류기를 실제로 flip하는 비율
                       높을수록 좋음 (기준: ≥70% 양호)
  2. Proximity       : 정규화 공간에서 원본 대비 변화량 (L1/L2)
                       작을수록 좋음 (최소 변화로 목표 달성 = 현실적 CF)
  3. Actionability   : 행동가능성 제약(불변/단조) 준수 비율
                       높을수록 좋음 (기준: ≥95% 양호)
  4. Plausibility    : CF가 실제 학습 데이터 분포 내에 있는지
                       높을수록 좋음 — OOD(out-of-distribution) CF 탐지

[사용법]
    from cf_engine.evaluator import evaluate_all
    metrics = evaluate_all(results, preprocessor, save_path="results/cf_eval.json")
"""

import json
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    IMMUTABLE_FEATURES, INCREASING_ONLY_FEATURES, DECREASING_ONLY_FEATURES,
)

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ─── 내부 헬퍼 ───────────────────────────────────────────────────────────────

def _sample_results(results: List[dict], max_samples: int) -> List[dict]:
    """균등 간격 샘플링 (대규모 데이터 평가 속도 제어)"""
    if max_samples <= 0 or len(results) <= max_samples:
        return results
    indices = np.linspace(0, len(results) - 1, max_samples, dtype=int)
    return [results[i] for i in indices]


def _batch_transform(results: List[dict], preprocessor, key: str) -> np.ndarray:
    """factual 또는 counterfactual DataFrame을 배치 변환"""
    rows = pd.DataFrame([r[key][ALL_FEATURES] for r in results])
    return preprocessor.transform(rows)


# ─── 1. Validity Rate ────────────────────────────────────────────────────────

def compute_validity_rate(results: List[dict]) -> float:
    """
    Validity Rate = 분류기를 실제로 flip한 CF 비율

    CF가 유효하다 = cf_prob < 0.5 (inactive_risk=0 예측)
    """
    if not results:
        return 0.0
    return float(np.mean([float(r["cf_valid"]) for r in results]))


# ─── 2. Proximity ────────────────────────────────────────────────────────────

def compute_proximity(
    results: List[dict],
    preprocessor,
    max_samples: int = 5000,
) -> Dict:
    """
    Proximity: 정규화 공간에서 원본 ↔ CF 거리

    수치형 피처의 StandardScaler 정규화 공간에서 거리를 측정합니다.
    단위가 다른 피처들을 동일 스케일로 비교할 수 있어
    "어느 피처가 얼마나 변했는지" 직관적으로 파악 가능합니다.

    Returns:
        l1_mean     : 평균 L1 거리 (피처별 절대 변화량 합)
        l2_mean     : 평균 L2 거리 (유클리드 거리)
        l1_per_feat : 수치형 피처별 평균 절대 변화량 (σ 단위)
        n_sampled   : 실제 평가에 사용된 샘플 수
    """
    sample = _sample_results(results, max_samples)
    if not sample:
        return {"l1_mean": 0.0, "l2_mean": 0.0, "l1_per_feat": {}, "n_sampled": 0}

    num_dim = preprocessor.num_dim

    try:
        x_f = _batch_transform(sample, preprocessor, "factual")
        x_c = _batch_transform(sample, preprocessor, "counterfactual")
    except Exception as e:
        print(f"  [Proximity] 변환 오류: {e}")
        return {"l1_mean": float("nan"), "l2_mean": float("nan"),
                "l1_per_feat": {}, "n_sampled": 0}

    diffs = np.abs(x_c[:, :num_dim] - x_f[:, :num_dim])  # (n, num_dim)
    l1_per_feat = {
        feat: float(np.mean(diffs[:, i]))
        for i, feat in enumerate(NUMERICAL_FEATURES)
    }
    return {
        "l1_mean":     float(np.mean(diffs.sum(axis=1))),
        "l2_mean":     float(np.mean(np.linalg.norm(diffs, axis=1))),
        "l1_per_feat": l1_per_feat,
        "n_sampled":   len(sample),
    }


# ─── 3. Actionability Rate ───────────────────────────────────────────────────

def compute_actionability_rate(
    results: List[dict],
    max_samples: int = 10000,
) -> Dict:
    """
    Actionability Rate: 행동가능성 제약 완전 준수 비율

    점검 항목:
      ① 불변 피처 (IMMUTABLE_FEATURES):
            CF값 = factual값  (허용 오차 1e-4)
      ② 단조 증가 (INCREASING_ONLY_FEATURES):
            CF값 ≥ factual값  (사용 실적·거래건수 등 — 줄이면 안 됨)
      ③ 단조 감소 (DECREASING_ONLY_FEATURES):
            CF값 ≤ factual값  (무실적 경과 월·앱 미사용 일수 등 — 늘면 안 됨)

    Returns:
        overall       : 모든 제약을 완전히 준수하는 CF 비율
        immutable_ok  : 불변 피처 준수율
        monotone_ok   : 단조 제약 준수율
        per_feature   : 피처별 위반 건수 dict
        n_sampled     : 평가 샘플 수
    """
    sample = _sample_results(results, max_samples)
    if not sample:
        return {"overall": 0.0, "immutable_ok": 0.0, "monotone_ok": 0.0,
                "per_feature": {}, "n_sampled": 0}

    tol = 1e-4
    imm_ok_list, mono_ok_list, full_ok_list = [], [], []
    per_feat_viol: Dict[str, int] = {}

    for r in sample:
        f, c = r["factual"], r["counterfactual"]
        imm_ok = mono_ok = True

        for feat in IMMUTABLE_FEATURES:
            if feat in NUMERICAL_FEATURES:
                if abs(float(c[feat]) - float(f[feat])) > tol:
                    imm_ok = False
                    per_feat_viol[feat] = per_feat_viol.get(feat, 0) + 1

        for feat in INCREASING_ONLY_FEATURES:
            if feat in NUMERICAL_FEATURES:
                if float(c[feat]) < float(f[feat]) - tol:
                    mono_ok = False
                    per_feat_viol[feat] = per_feat_viol.get(feat, 0) + 1

        for feat in DECREASING_ONLY_FEATURES:
            if feat in NUMERICAL_FEATURES:
                if float(c[feat]) > float(f[feat]) + tol:
                    mono_ok = False
                    per_feat_viol[feat] = per_feat_viol.get(feat, 0) + 1

        imm_ok_list.append(float(imm_ok))
        mono_ok_list.append(float(mono_ok))
        full_ok_list.append(float(imm_ok and mono_ok))

    return {
        "overall":      float(np.mean(full_ok_list)),
        "immutable_ok": float(np.mean(imm_ok_list)),
        "monotone_ok":  float(np.mean(mono_ok_list)),
        "per_feature":  per_feat_viol,
        "n_sampled":    len(sample),
    }


# ─── 4. Plausibility ─────────────────────────────────────────────────────────

def compute_plausibility(
    results: List[dict],
    preprocessor,
    sigma: float = 2.5,
    max_samples: int = 5000,
) -> Dict:
    """
    Plausibility: CF가 실제 학습 데이터 분포 내에 있는지

    [평가 방법]
      방법 A (기본): 정규화 공간 ±sigma σ 기준
                    sigma=2.5 → 정규분포 가정 시 학습 데이터 ~98.8% 커버
      방법 B (정확): preprocessor.num_pct05 / num_pct95 저장된 경우
                    실제 5th–95th percentile 범위 기준

    [해석]
      - overall 낮음 → CF가 학습 분포 밖 → 비현실적 or adversarial CF
      - mean_sigma 높음 → CF가 정규화 공간에서 극단값 → OOD 가능성

    Returns:
        overall      : 모든 수치형 피처가 분포 범위 내인 CF 비율
        per_feature  : 피처별 분포 범위 내 비율
        mean_sigma   : CF 정규화 값 절대값 평균 (σ 단위, 작을수록 분포 중심)
        method       : 사용된 평가 방법 문자열
        n_sampled    : 평가 샘플 수
    """
    sample = _sample_results(results, max_samples)
    if not sample:
        return {"overall": 0.0, "per_feature": {}, "mean_sigma": 0.0,
                "method": "n/a", "n_sampled": 0}

    num_dim = preprocessor.num_dim
    has_pct = (
        hasattr(preprocessor, "num_pct05")
        and hasattr(preprocessor, "num_pct95")
    )

    try:
        x_c_norm = _batch_transform(sample, preprocessor, "counterfactual")[:, :num_dim]
    except Exception as e:
        print(f"  [Plausibility] 변환 오류: {e}")
        return {"overall": float("nan"), "per_feature": {}, "mean_sigma": float("nan"),
                "method": "error", "n_sampled": 0}

    mean_sigma = float(np.abs(x_c_norm).mean())

    if has_pct:
        # 원래 스케일로 역변환 후 percentile 비교
        x_orig = preprocessor.num_scaler.inverse_transform(x_c_norm)  # (n, num_dim)
        lo = preprocessor.num_pct05[np.newaxis, :]
        hi = preprocessor.num_pct95[np.newaxis, :]
        within = (x_orig >= lo) & (x_orig <= hi)
        method = "5th–95th percentile"
    else:
        within = np.abs(x_c_norm) <= sigma
        method = f"±{sigma}σ"

    per_feat = {
        feat: float(within[:, i].mean())
        for i, feat in enumerate(NUMERICAL_FEATURES)
    }
    return {
        "overall":     float(within.all(axis=1).mean()),
        "per_feature": per_feat,
        "mean_sigma":  mean_sigma,
        "method":      method,
        "n_sampled":   len(sample),
    }


# ─── 통합 평가 ───────────────────────────────────────────────────────────────

def evaluate_all(
    results: List[dict],
    preprocessor,
    max_eval_samples: int = 10000,
    verbose: bool = True,
    save_path: Optional[str] = None,
) -> Dict:
    """
    4대 CF 품질 지표 일괄 계산 및 출력

    Args:
        results:           CF 결과 리스트 (generate_cf_two_stage 반환값)
        preprocessor:      HyundaiCardPreprocessor
        max_eval_samples:  대규모 데이터 샘플링 상한 (0=전체)
        verbose:           상세 보고서 콘솔 출력
        save_path:         JSON 저장 경로 (None=저장 안 함)

    Returns:
        metrics: 모든 지표가 담긴 dict (JSON 직렬화 가능)
    """
    if not results:
        print("  [평가] CF 결과가 없습니다.")
        return {}

    n = len(results)
    validity  = compute_validity_rate(results)
    proximity = compute_proximity(results, preprocessor, max_eval_samples)
    action    = compute_actionability_rate(results, max_eval_samples)
    plaus     = compute_plausibility(results, preprocessor, max_samples=max_eval_samples)

    metrics = {
        "n_total":                     n,
        "n_valid":                     int(sum(r["cf_valid"] for r in results)),
        # 1. Validity
        "validity_rate":               round(validity, 4),
        # 2. Proximity
        "proximity_l1":                round(proximity["l1_mean"], 4),
        "proximity_l2":                round(proximity["l2_mean"], 4),
        "proximity_per_feat":          {f: round(v, 4) for f, v in proximity["l1_per_feat"].items()},
        # 3. Actionability
        "actionability_rate":          round(action["overall"], 4),
        "actionability_immutable":     round(action["immutable_ok"], 4),
        "actionability_monotone":      round(action["monotone_ok"], 4),
        "actionability_violations":    action["per_feature"],
        # 4. Plausibility
        "plausibility_rate":           round(plaus["overall"], 4),
        "plausibility_per_feat":       {f: round(v, 4) for f, v in plaus["per_feature"].items()},
        "plausibility_mean_sigma":     round(plaus["mean_sigma"], 4),
        "plausibility_method":         plaus["method"],
        # 메타
        "eval_samples":                proximity.get("n_sampled", n),
    }

    if verbose:
        _print_report(metrics, proximity, action, plaus)

    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2, ensure_ascii=False)
        print(f"  평가 결과 저장: {save_path}")

    return metrics


# ─── 보고서 출력 ─────────────────────────────────────────────────────────────

def _print_report(metrics: Dict, proximity: Dict, action: Dict, plaus: Dict) -> None:
    n     = metrics["n_total"]
    v     = metrics["validity_rate"]
    a     = metrics["actionability_rate"]
    p     = metrics["plausibility_rate"]
    l1    = metrics["proximity_l1"]
    l2    = metrics["proximity_l2"]
    nsamp = metrics["eval_samples"]

    def _grade(val, thr_good, thr_ok):
        return "◎ 양호" if val >= thr_good else ("△ 보통" if val >= thr_ok else "✗ 불량")

    print("\n" + "=" * 72)
    print("  ■ CF 품질 평가 보고서  (Counterfactual Evaluation Report)")
    print(f"  평가 대상: 전체 {n:,}명 중 {nsamp:,}명 샘플  |  "
          f"유효 CF: {metrics['n_valid']:,}명")
    print("=" * 72)

    # 1. Validity
    grade_v = _grade(v, 0.70, 0.40)
    print(f"\n  [{grade_v}] 1. Validity Rate       {v*100:6.1f}%")
    print(f"               분류기를 flip한 CF 비율  |  기준: ≥70% 양호")

    # 2. Proximity
    print(f"\n  [○ 참고]  2. Proximity (정규화 공간)")
    print(f"               L1 = {l1:.3f}σ   L2 = {l2:.3f}σ   "
          f"(작을수록 최소 변화로 목표 달성)")
    top3 = sorted(proximity["l1_per_feat"].items(), key=lambda x: -x[1])[:3]
    if top3:
        print("               변화량 상위 3 피처:")
        for feat, val in top3:
            bar = "█" * min(int(val * 8), 20)
            print(f"                 {feat:<28} {val:.3f}σ  {bar}")

    # 3. Actionability
    grade_a = _grade(a, 0.95, 0.80)
    print(f"\n  [{grade_a}] 3. Actionability Rate  {a*100:6.1f}%")
    print(f"               불변 피처: {action['immutable_ok']*100:.1f}%  |  "
          f"단조 제약: {action['monotone_ok']*100:.1f}%")
    viols = {f: c for f, c in action["per_feature"].items() if c > 0}
    if viols:
        top_v = sorted(viols.items(), key=lambda x: -x[1])[:3]
        print("               위반: " + "  /  ".join(f"{f} {c}건" for f, c in top_v))

    # 4. Plausibility
    grade_p = _grade(p, 0.85, 0.60)
    print(f"\n  [{grade_p}] 4. Plausibility Rate   {p*100:6.1f}%  ({plaus['method']})")
    print(f"               평균 |정규화값|: {plaus['mean_sigma']:.2f}σ  "
          f"(클수록 분포 이탈 위험)")
    low = [(f, val) for f, val in plaus["per_feature"].items() if val < 0.85]
    if low:
        low.sort(key=lambda x: x[1])
        print("               분포 이탈 위험 피처:")
        for feat, val in low[:3]:
            print(f"                 {feat:<28} {val*100:.1f}%")

    # 종합
    print("\n" + "─" * 72)
    print(f"  종합  Validity {v*100:.1f}%  Proximity L1={l1:.3f}  "
          f"Actionability {a*100:.1f}%  Plausibility {p*100:.1f}%")

    # 개선 제안
    issues = []
    if v < 0.40:
        issues.append("유효율 낮음 → max_cf_changes 증가 or Stage3 opt_iter 증가")
    if l1 > 3.0:
        issues.append("변화 과다 → max_cf_changes 축소 (현재 4), sparsity_weight 증가")
    if a < 0.80:
        issues.append("제약 위반 → constraints.apply() 적용 경로 재확인")
    if p < 0.60:
        issues.append("분포 이탈 → guidance_scale 축소, ±3σ 클리핑 강화")
    if issues:
        print("\n  ⚠ 개선 권고:")
        for iss in issues:
            print(f"    • {iss}")
    print("=" * 72)
