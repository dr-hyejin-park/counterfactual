"""
현대카드 무실적 위험 고객 Wake-up Counterfactual 데모

실행:
  python run_demo.py [--n_customers N] [--customer_id HCC_00042]

기능:
  1. 학습된 TabDiff + 분류기 로드
  2. 무실적 위험 고객 대상 CF 생성
  3. 고객별 Wake-up Treatment 권고 보고서 출력
  4. 품질 평가 지표 (유효성/근접도/희소성)
  5. 시각화 저장 (results/cf_report.png)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TABDIFF_CONFIG, CF_CONFIG,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
)
from utils.preprocessing import HyundaiCardPreprocessor
from models.tabdiff import TabDiff
from models.classifier import CardIssuanceClassifier
from cf_engine.generator import TabDiffCFGenerator, print_cf_report, _translate_cat, _feat_name_kr
from cf_engine.constraints import compute_proximity, compute_sparsity


def parse_args():
    parser = argparse.ArgumentParser(description="현대카드 무실적 Wake-up CF 데모")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--n_customers",    type=int, default=10)
    parser.add_argument("--customer_id",    type=str, default=None)
    parser.add_argument("--device",         type=str, default="cpu")
    parser.add_argument("--output_dir",     type=str, default="results")
    parser.add_argument("--no_viz",         action="store_true")
    return parser.parse_args()


def load_models(checkpoint_dir: str, device: str):
    print(f"\n모델 로드 중: {checkpoint_dir}/")
    preprocessor = HyundaiCardPreprocessor.load(f"{checkpoint_dir}/preprocessor.pkl")
    print(f"  전처리기 로드 완료 (입력 차원: {preprocessor.total_dim})")

    clf_ckpt   = torch.load(f"{checkpoint_dir}/classifier.pt",  map_location=device)
    classifier = CardIssuanceClassifier(input_dim=clf_ckpt["input_dim"])
    classifier.load_state_dict(clf_ckpt["model_state"])
    classifier.to(device).eval()
    print("  분류기 로드 완료")

    td_ckpt = torch.load(f"{checkpoint_dir}/tabdiff.pt", map_location=device)
    cfg     = td_ckpt["config"]
    tabdiff = TabDiff(
        input_dim=td_ckpt["input_dim"],
        num_timesteps=cfg["num_timesteps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        device=device,
    )
    tabdiff.load_state_dict(td_ckpt["model_state"])
    tabdiff.to(device).eval()
    print("  TabDiff 로드 완료")
    return preprocessor, classifier, tabdiff


def evaluate_cf_quality(results: List[Dict], preprocessor, classifier, device: str) -> Dict:
    """반사실적 품질 지표 계산"""
    valid_results = [r for r in results if r["cf_valid"]]
    if not valid_results:
        return {"validity": 0.0, "proximity": float("inf"),
                "sparsity": float("inf"), "num_valid": 0, "num_total": len(results)}

    fact_df = pd.DataFrame([r["factual"]       for r in valid_results])[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    cf_df   = pd.DataFrame([r["counterfactual"] for r in valid_results])[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    x_fact_enc = preprocessor.transform(fact_df)
    x_cf_enc   = preprocessor.transform(cf_df)

    return {
        "validity":         len(valid_results) / len(results),
        "proximity":        compute_proximity(x_cf_enc, x_fact_enc, preprocessor),
        "sparsity":         compute_sparsity(x_cf_enc, x_fact_enc, preprocessor),
        "num_valid":        len(valid_results),
        "num_total":        len(results),
        "avg_cf_risk_prob": np.mean([r["cf_prob"]      for r in valid_results]),
        "avg_risk_drop":    np.mean([r["factual_prob"] - r["cf_prob"] for r in valid_results]),
        "avg_num_changes":  np.mean([r["num_changes"]  for r in results]),
    }


def visualize_results(results: List[Dict], output_dir: str) -> None:
    """Wake-up CF 분석 시각화 (영문 레이블)"""
    os.makedirs(output_dir, exist_ok=True)

    feat_en = {
        "마지막 거래 경과":       "Last Txn (months)",
        "최근 3M 월 사용금액":    "Spending 3M",
        "사용금액 트렌드":         "Spending Trend",
        "최근 3M 월 거래 건수":   "Txn Count 3M",
        "무실적 월 수":            "Missed Months",
        "앱 미사용 기간":          "App Inactive (days)",
        "미사용 포인트 잔액":      "Points Balance",
        "활성 혜택 수":            "Active Benefits",
        "디지털 참여도":           "Engagement",
        "결제 수단":               "Payment Method",
        "주요 사용 카테고리":      "Spending Category",
    }
    def en(name: str) -> str:
        return feat_en.get(name, name)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Hyundai Card — Inactive Risk Customer Wake-up Campaign\n"
        "TabDiff Counterfactual Explanation Analysis",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # ── 1. 무실적 위험도: Factual vs CF ──────────────────────────────────
    ax = axes[0, 0]
    risk_probs = [r["factual_prob"] for r in results]
    cf_probs   = [r["cf_prob"]      for r in results]
    x = np.arange(len(results))
    w = 0.35
    ax.bar(x - w/2, risk_probs, w, label="Factual (Current Risk)", color="#E74C3C", alpha=0.8)
    ax.bar(x + w/2, cf_probs,   w, label="After CF (Target Risk)",  color="#2ECC71", alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Threshold (0.5)")
    ax.set_xlabel("Customer Index")
    ax.set_ylabel("P(Inactive Risk = 1)")
    ax.set_title("Inactive Risk: Factual vs After Wake-up CF")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([r["customer_id"].replace("HCC_", "") for r in results], rotation=45, fontsize=7)

    # ── 2. CF 유효성 파이 차트 ────────────────────────────────────────────
    ax = axes[0, 1]
    n_valid   = sum(1 for r in results if r["cf_valid"])
    n_invalid = len(results) - n_valid
    ax.pie(
        [n_valid, n_invalid],
        labels=[f"Valid Wake-up CF\n({n_valid} customers)",
                f"Needs More Effort\n({n_invalid} customers)"],
        colors=["#2ECC71", "#E74C3C"],
        autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title(f"CF Validity (n={len(results)})")

    # ── 3. Treatment 빈도 분석 ────────────────────────────────────────────
    ax = axes[0, 2]
    feat_counts: Dict[str, int] = {}
    for r in results:
        for t in r["treatments"]:
            lbl = en(t["feature_kr"])
            feat_counts[lbl] = feat_counts.get(lbl, 0) + 1
    if feat_counts:
        feats  = sorted(feat_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        names  = [f[0] for f in feats]
        counts = [f[1] for f in feats]
        ax.barh(names, counts, color=sns.color_palette("Blues_r", len(feats)))
        ax.set_xlabel("Customers Requiring Change")
        ax.set_title("Most Required Feature Changes\n(Wake-up Treatment Frequency)")
        ax.set_xlim(0, len(results) + 1)
        for i, v in enumerate(counts):
            ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
    else:
        ax.set_title("Treatment Frequency")

    # ── 4. 변화 방향별 분석 (↑/↓) ─────────────────────────────────────────
    ax = axes[1, 0]
    up_counts: Dict[str, int]   = {}
    down_counts: Dict[str, int] = {}
    for r in results:
        for t in r["treatments"]:
            if t["type"] == "numerical":
                lbl = en(t["feature_kr"])
                if t["direction"] == "↑":
                    up_counts[lbl]   = up_counts.get(lbl, 0) + 1
                else:
                    down_counts[lbl] = down_counts.get(lbl, 0) + 1
    all_feats = sorted(set(list(up_counts.keys()) + list(down_counts.keys())))
    if all_feats:
        y_pos = np.arange(len(all_feats))
        ups   = [ up_counts.get(f, 0) for f in all_feats]
        downs = [-down_counts.get(f, 0) for f in all_feats]
        ax.barh(y_pos, ups,   color="#2ECC71", alpha=0.8, label="Needs to Increase")
        ax.barh(y_pos, downs, color="#E74C3C", alpha=0.8, label="Needs to Decrease")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_feats, fontsize=8)
        ax.set_xlabel("Number of Customers")
        ax.set_title("Required Change Direction\n(Wake-up Action Direction)")
        ax.legend(fontsize=8)

    # ── 5. 위험도 감소폭 ─────────────────────────────────────────────────
    ax = axes[1, 1]
    risk_drops = [r["factual_prob"] - r["cf_prob"] for r in results]
    cids       = [r["customer_id"].replace("HCC_", "C") for r in results]
    colors     = ["#2ECC71" if d > 0 else "#E74C3C" for d in risk_drops]
    bars = ax.bar(range(len(results)), risk_drops, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Customer")
    ax.set_ylabel("Inactive Risk Drop (Factual - CF)")
    ax.set_title("Inactive Risk Reduction by Wake-up CF")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(cids, rotation=45, fontsize=7)
    for bar, val in zip(bars, risk_drops):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{val:+.2f}", ha="center", va="bottom", fontsize=7)

    # ── 6. 평균 변화율 요약 ───────────────────────────────────────────────
    ax = axes[1, 2]
    feat_pcts: Dict[str, List[float]] = {}
    for r in results:
        for t in r["treatments"]:
            if t["type"] == "numerical" and t["pct_change"] is not None:
                lbl = en(t["feature_kr"])
                feat_pcts.setdefault(lbl, []).append(t["pct_change"])
    if feat_pcts:
        avg_pcts     = {k: np.mean(v) for k, v in feat_pcts.items()}
        sorted_feats = sorted(avg_pcts.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        names = [f[0] for f in sorted_feats]
        vals  = [f[1] for f in sorted_feats]
        colors = ["#2ECC71" if v > 0 else "#E74C3C" for v in vals]
        ax.barh(names, vals, color=colors, alpha=0.8)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Average % Change Required")
        ax.set_title("Average Required Change\n(Top 6 Features)")
        for i, v in enumerate(vals):
            ax.text(v + (1 if v >= 0 else -1), i, f"{v:+.0f}%", va="center", fontsize=8)

    plt.tight_layout()
    path = f"{output_dir}/cf_report.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  시각화 저장: {path}")


def print_evaluation_summary(metrics: Dict) -> None:
    print("\n" + "=" * 60)
    print("반사실적 품질 평가 지표 (Wake-up Campaign)")
    print("=" * 60)
    print(f"  유효성 (Validity)        : {metrics['validity']*100:.1f}% "
          f"({metrics.get('num_valid',0)}/{metrics.get('num_total',0)}명)")
    print(f"  근접도 (Proximity)       : {metrics['proximity']:.4f} (낮을수록 좋음)")
    print(f"  희소성 (Sparsity)        : {metrics['sparsity']:.4f} (낮을수록 좋음)")
    if "avg_cf_risk_prob" in metrics:
        print(f"  CF 적용 후 평균 위험도   : {metrics['avg_cf_risk_prob']*100:.1f}%")
    if "avg_risk_drop" in metrics:
        print(f"  평균 위험도 감소폭       : {metrics['avg_risk_drop']*100:.1f}%p")
    if "avg_num_changes" in metrics:
        print(f"  평균 변경 피처 수        : {metrics['avg_num_changes']:.1f}개")
    print("=" * 60)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 68)
    print("  현대카드 무실적 위험 고객 Wake-up Campaign")
    print("  TabDiff Counterfactual Explanation System")
    print("=" * 68)

    preprocessor, classifier, tabdiff = load_models(args.checkpoint_dir, args.device)

    cf_generator = TabDiffCFGenerator(
        tabdiff_model=tabdiff,
        classifier=classifier,
        preprocessor=preprocessor,
        device=args.device,
        cf_config=CF_CONFIG,
    )

    test_csv = f"{args.checkpoint_dir}/test_customers.csv"
    if not os.path.exists(test_csv):
        print(f"\n테스트 데이터 없음: {test_csv}")
        print("먼저 train.py를 실행하세요.")
        return

    test_df = pd.read_csv(test_csv)

    if args.customer_id:
        target_df = test_df[test_df["customer_id"] == args.customer_id]
        if len(target_df) == 0:
            print(f"고객 ID {args.customer_id}를 찾을 수 없습니다.")
            return
    else:
        at_risk = test_df[test_df[TARGET_COLUMN] == 1]
        target_df = at_risk.head(args.n_customers)
        print(f"\n대상 고객: {len(target_df)}명 (무실적 위험 고객)")

    print(f"\nCounterfactual 생성 중 "
          f"(T_cf={CF_CONFIG['num_cf_timesteps']}, λ={CF_CONFIG['guidance_scale']})...")
    results = cf_generator.explain(
        customer_df=target_df,
        max_customers=len(target_df),
        verbose=True,
    )

    print("\n\n" + "=" * 68)
    print("  고객별 Wake-up Counterfactual 보고서")
    print("=" * 68)
    for result in results:
        print_cf_report(result)

    metrics = evaluate_cf_quality(results, preprocessor, classifier, args.device)
    print_evaluation_summary(metrics)

    # ── CSV 저장 ────────────────────────────────────────────────────────────
    rows = []
    for r in results:
        row = {
            "customer_id":   r["customer_id"],
            "risk_prob":     r["factual_prob"],
            "cf_risk_prob":  r["cf_prob"],
            "risk_drop_pct": (r["factual_prob"] - r["cf_prob"]) * 100,
            "cf_valid":      r["cf_valid"],
            "num_changes":   r["num_changes"],
        }
        for feat in NUMERICAL_FEATURES:
            row[f"fact_{feat}"] = float(r["factual"].get(feat, None))
            row[f"cf_{feat}"]   = float(r["counterfactual"].get(feat, None))
        row["treatments_summary"] = " | ".join([
            f"{t['feature_kr']}: {t['original']:.1f}→{t['counterfactual']:.1f}"
            if t["type"] == "numerical"
            else f"{t['feature_kr']}: {t['original']}→{t['counterfactual']}"
            for t in r["treatments"]
        ])
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_csv = f"{args.output_dir}/wakeup_cf_results.csv"
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    print(f"\n  결과 저장: {result_csv}")

    if not args.no_viz:
        visualize_results(results, args.output_dir)

    print("\n완료! 결과 파일:")
    print(f"  - {result_csv}")
    if not args.no_viz:
        print(f"  - {args.output_dir}/cf_report.png")


if __name__ == "__main__":
    main()
