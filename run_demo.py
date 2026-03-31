"""
현대카드 추가 발급 Counterfactual Explanation 데모

실행:
  python run_demo.py [--n_customers N] [--customer_id HCC_00042]

기능:
  1. 학습된 TabDiff + 분류기 로드
  2. 추가 발급 거절 고객 대상 CF 생성
  3. 고객별 치료(Treatment) 권고안 보고서 출력
  4. 평가 지표 (유효성/근접도/희소성) 출력
  5. 시각화 저장 (results/cf_report.png)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # 헤드리스 서버 대응
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_CONFIG, TABDIFF_CONFIG, CLASSIFIER_CONFIG, CF_CONFIG,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
)
from utils.preprocessing import HyundaiCardPreprocessor
from models.tabdiff import TabDiff
from models.classifier import CardIssuanceClassifier
from cf_engine.generator import TabDiffCFGenerator, print_cf_report
from cf_engine.constraints import compute_proximity, compute_sparsity, compute_validity


def parse_args():
    parser = argparse.ArgumentParser(description="현대카드 CF 데모")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--n_customers", type=int, default=10,
                        help="CF를 생성할 거절 고객 수")
    parser.add_argument("--customer_id", type=str, default=None,
                        help="특정 고객 ID (예: HCC_00042)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--no_viz", action="store_true", help="시각화 생략")
    return parser.parse_args()


def load_models(checkpoint_dir: str, device: str):
    """저장된 모델 로드"""
    print(f"\n모델 로드 중: {checkpoint_dir}/")

    preprocessor = HyundaiCardPreprocessor.load(f"{checkpoint_dir}/preprocessor.pkl")
    print(f"  전처리기 로드 완료 (입력 차원: {preprocessor.total_dim})")

    clf_ckpt = torch.load(f"{checkpoint_dir}/classifier.pt", map_location=device)
    classifier = CardIssuanceClassifier(input_dim=clf_ckpt["input_dim"])
    classifier.load_state_dict(clf_ckpt["model_state"])
    classifier.to(device).eval()
    print("  분류기 로드 완료")

    tabdiff_ckpt = torch.load(f"{checkpoint_dir}/tabdiff.pt", map_location=device)
    cfg = tabdiff_ckpt["config"]
    tabdiff = TabDiff(
        input_dim=tabdiff_ckpt["input_dim"],
        num_timesteps=cfg["num_timesteps"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        hidden_dim=cfg["hidden_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        device=device,
    )
    tabdiff.load_state_dict(tabdiff_ckpt["model_state"])
    tabdiff.to(device).eval()
    print("  TabDiff 로드 완료")

    return preprocessor, classifier, tabdiff


def evaluate_cf_quality(
    results: List[Dict],
    preprocessor,
    classifier,
    device: str,
) -> Dict:
    """반사실적 품질 평가 지표 계산"""
    valid_results = [r for r in results if r["cf_valid"]]

    if not valid_results:
        return {"validity": 0.0, "proximity": float("inf"), "sparsity": float("inf")}

    factuals = [r["factual"] for r in valid_results]
    cfs = [r["counterfactual"] for r in valid_results]

    fact_df = pd.DataFrame(factuals)[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    cf_df = pd.DataFrame(cfs)[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]

    x_fact_enc = preprocessor.transform(fact_df)
    x_cf_enc = preprocessor.transform(cf_df)

    proximity = compute_proximity(x_cf_enc, x_fact_enc, preprocessor)
    sparsity = compute_sparsity(x_cf_enc, x_fact_enc, preprocessor)
    validity = len(valid_results) / len(results)

    return {
        "validity": validity,
        "proximity": proximity,
        "sparsity": sparsity,
        "num_valid": len(valid_results),
        "num_total": len(results),
        "avg_cf_prob": np.mean([r["cf_prob"] for r in valid_results]),
        "avg_num_changes": np.mean([r["num_changes"] for r in results]),
    }


def visualize_results(results: List[Dict], output_dir: str, preprocessor) -> None:
    """결과 시각화 (영문 레이블 — 한글 폰트 의존성 없음)"""
    os.makedirs(output_dir, exist_ok=True)

    # feature_kr → English 매핑
    feat_en = {
        "나이": "Age", "연소득": "Annual Income", "신용점수": "Credit Score",
        "보유 카드 수": "Num Cards", "월 카드 사용금액": "Monthly Spending",
        "거래 연수": "Years as Customer", "총 대출금액": "Total Loan",
        "월 거래 건수": "Monthly Txn", "연체 횟수": "Delinquencies",
        "신용 한도 사용률": "Utilization Rate",
        "혼인 상태": "Marital Status", "고용 형태": "Employment Type",
        "교육 수준": "Education", "거주 지역": "Region",
    }

    def en(name: str) -> str:
        return feat_en.get(name, name)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Hyundai Card - Additional Card Issuance\n"
        "Counterfactual Explanation Analysis (TabDiff-based)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # ── 1. CF 예측 확률 분포 ───────────────────────────────────────────────
    ax = axes[0, 0]
    factual_probs = [r["factual_prob"] for r in results]
    cf_probs = [r["cf_prob"] for r in results]
    x = np.arange(len(results))
    width = 0.35
    ax.bar(x - width/2, factual_probs, width, label="Factual (Current)", color="#E74C3C", alpha=0.8)
    ax.bar(x + width/2, cf_probs, width, label="Counterfactual (CF)", color="#2ECC71", alpha=0.8)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Decision Boundary (0.5)")
    ax.set_xlabel("Customer Index")
    ax.set_ylabel("P(Additional Card = 1)")
    ax.set_title("Approval Probability: Factual vs Counterfactual")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([r["customer_id"].replace("HCC_", "") for r in results], rotation=45, fontsize=7)

    # ── 2. CF 유효성 파이 차트 ────────────────────────────────────────────
    ax = axes[0, 1]
    n_valid = sum(1 for r in results if r["cf_valid"])
    n_invalid = len(results) - n_valid
    ax.pie(
        [n_valid, n_invalid],
        labels=[f"Valid CF\n({n_valid} customers)", f"Invalid CF\n({n_invalid} customers)"],
        colors=["#2ECC71", "#E74C3C"],
        autopct="%1.0f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )
    ax.set_title(f"CF Validity\n(Total: {len(results)} customers)")

    # ── 3. Treatment 빈도 분석 ────────────────────────────────────────────
    ax = axes[0, 2]
    feat_counts: Dict[str, int] = {}
    for r in results:
        for t in r["treatments"]:
            feat_label = en(t["feature_kr"])
            feat_counts[feat_label] = feat_counts.get(feat_label, 0) + 1
    if feat_counts:
        feats = sorted(feat_counts.items(), key=lambda x: x[1], reverse=True)[:8]
        names = [f[0] for f in feats]
        counts = [f[1] for f in feats]
        colors = sns.color_palette("Blues_r", len(feats))
        ax.barh(names, counts, color=colors)
        ax.set_xlabel("Number of Customers Requiring Change")
        ax.set_title("Most Frequently Required Feature Changes\n(Treatment Analysis)")
        ax.set_xlim(0, len(results) + 1)
        for i, v in enumerate(counts):
            ax.text(v + 0.1, i, str(v), va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No treatments", ha="center", va="center")
        ax.set_title("Treatment Analysis")

    # ── 4. 변화 방향별 피처 분석 ─────────────────────────────────────────
    ax = axes[1, 0]
    up_counts: Dict[str, int] = {}
    down_counts: Dict[str, int] = {}
    for r in results:
        for t in r["treatments"]:
            if t["type"] == "numerical":
                feat_label = en(t["feature_kr"])
                if t["direction"] == "↑":
                    up_counts[feat_label] = up_counts.get(feat_label, 0) + 1
                else:
                    down_counts[feat_label] = down_counts.get(feat_label, 0) + 1
    all_feats = sorted(set(list(up_counts.keys()) + list(down_counts.keys())))
    if all_feats:
        y_pos = np.arange(len(all_feats))
        ups = [up_counts.get(f, 0) for f in all_feats]
        downs = [-down_counts.get(f, 0) for f in all_feats]
        ax.barh(y_pos, ups, color="#2ECC71", alpha=0.8, label="Increase")
        ax.barh(y_pos, downs, color="#E74C3C", alpha=0.8, label="Decrease")
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_feats, fontsize=8)
        ax.set_xlabel("Number of Customers")
        ax.set_title("Required Change Direction per Feature")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("Change Direction Analysis")

    # ── 5. CF 확률 향상 ───────────────────────────────────────────────────
    ax = axes[1, 1]
    prob_improvements = [r["cf_prob"] - r["factual_prob"] for r in results]
    customer_ids = [r["customer_id"].replace("HCC_", "C") for r in results]
    colors = ["#2ECC71" if p > 0 else "#E74C3C" for p in prob_improvements]
    bars = ax.bar(range(len(results)), prob_improvements, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Customer")
    ax.set_ylabel("Probability Improvement (CF - Factual)")
    ax.set_title("Approval Probability Improvement by CF")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(customer_ids, rotation=45, fontsize=7)
    for bar, val in zip(bars, prob_improvements):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:+.2f}", ha="center", va="bottom", fontsize=7)

    # ── 6. 평균 변화량 요약 ──────────────────────────────────────────────
    ax = axes[1, 2]
    feat_avg_pct: Dict[str, List[float]] = {}
    for r in results:
        for t in r["treatments"]:
            if t["type"] == "numerical" and t["pct_change"] is not None:
                feat_label = en(t["feature_kr"])
                if feat_label not in feat_avg_pct:
                    feat_avg_pct[feat_label] = []
                feat_avg_pct[feat_label].append(t["pct_change"])
    if feat_avg_pct:
        avg_pcts = {k: np.mean(v) for k, v in feat_avg_pct.items()}
        sorted_feats = sorted(avg_pcts.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        names = [f[0] for f in sorted_feats]
        vals = [f[1] for f in sorted_feats]
        colors = ["#2ECC71" if v > 0 else "#E74C3C" for v in vals]
        ax.barh(names, vals, color=colors, alpha=0.8)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_xlabel("Average % Change Required")
        ax.set_title("Average Required Change\n(Top 6 Features)")
        for i, v in enumerate(vals):
            ax.text(v + (1 if v >= 0 else -1), i, f"{v:+.0f}%", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title("Average Change Analysis")

    plt.tight_layout()
    save_path = f"{output_dir}/cf_report.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  시각화 저장: {save_path}")


def print_evaluation_summary(metrics: Dict) -> None:
    """평가 지표 요약 출력"""
    print("\n" + "=" * 60)
    print("반사실적 품질 평가 지표")
    print("=" * 60)
    print(f"  유효성 (Validity)    : {metrics['validity']*100:.1f}% "
          f"({metrics.get('num_valid', 0)}/{metrics.get('num_total', 0)}명)")
    print(f"  근접도 (Proximity)   : {metrics['proximity']:.4f} (낮을수록 ↓ 좋음)")
    print(f"  희소성 (Sparsity)    : {metrics['sparsity']:.4f} (낮을수록 ↓ 좋음)")
    if "avg_cf_prob" in metrics:
        print(f"  평균 CF 확률         : {metrics['avg_cf_prob']*100:.1f}%")
    if "avg_num_changes" in metrics:
        print(f"  평균 변경 피처 수    : {metrics['avg_num_changes']:.1f}개")
    print("=" * 60)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  현대카드 추가 발급 Counterfactual Explanation 시스템")
    print("  (TabDiff 기반 반사실적 설명 — 논문 응용 구현)")
    print("=" * 60)

    # ── 모델 로드 ────────────────────────────────────────────────────────────
    preprocessor, classifier, tabdiff = load_models(args.checkpoint_dir, args.device)

    # CF 생성기 초기화
    cf_generator = TabDiffCFGenerator(
        tabdiff_model=tabdiff,
        classifier=classifier,
        preprocessor=preprocessor,
        device=args.device,
        cf_config=CF_CONFIG,
    )

    # ── 테스트 고객 데이터 로드 ───────────────────────────────────────────────
    test_csv = f"{args.checkpoint_dir}/test_customers.csv"
    if os.path.exists(test_csv):
        test_df = pd.read_csv(test_csv)
    else:
        print(f"\n테스트 데이터 없음: {test_csv}")
        print("먼저 train.py를 실행하세요.")
        return

    # 특정 고객 ID 선택
    if args.customer_id:
        customer_df = test_df[test_df["customer_id"] == args.customer_id]
        if len(customer_df) == 0:
            print(f"고객 ID {args.customer_id}를 찾을 수 없습니다.")
            return
        target_df = customer_df
    else:
        # 추가 발급 거절 고객 n명 선택
        rejected_df = test_df[test_df[TARGET_COLUMN] == 0]
        target_df = rejected_df.head(args.n_customers)
        print(f"\n대상 고객: {len(target_df)}명 (추가 발급 거절 고객 중)")

    # ── CF 생성 ──────────────────────────────────────────────────────────────
    print(f"\nCounterfactual 생성 중 (T_cf={CF_CONFIG['num_cf_timesteps']}, "
          f"λ={CF_CONFIG['guidance_scale']})...")
    results = cf_generator.explain(
        customer_df=target_df,
        max_customers=len(target_df),
        verbose=True,
    )

    # ── 결과 출력 ────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  고객별 반사실적 설명 보고서")
    print("=" * 65)
    for result in results:
        print_cf_report(result)

    # ── 품질 평가 ────────────────────────────────────────────────────────────
    metrics = evaluate_cf_quality(results, preprocessor, classifier, args.device)
    print_evaluation_summary(metrics)

    # ── 결과 CSV 저장 ─────────────────────────────────────────────────────────
    rows = []
    for r in results:
        row = {
            "customer_id": r["customer_id"],
            "factual_prob": r["factual_prob"],
            "cf_prob": r["cf_prob"],
            "cf_valid": r["cf_valid"],
            "num_changes": r["num_changes"],
        }
        # 원본 피처값
        for feat in NUMERICAL_FEATURES:
            row[f"fact_{feat}"] = float(r["factual"].get(feat, None))
            row[f"cf_{feat}"] = float(r["counterfactual"].get(feat, None))
        # Treatment 요약
        treatments_str = " | ".join([
            f"{t['feature_kr']}: {t['original']:.0f}→{t['counterfactual']:.0f}"
            if t["type"] == "numerical"
            else f"{t['feature_kr']}: {t['original']}→{t['counterfactual']}"
            for t in r["treatments"]
        ])
        row["treatments_summary"] = treatments_str
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_csv = f"{args.output_dir}/cf_results.csv"
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    print(f"\n  결과 저장: {result_csv}")

    # ── 시각화 ──────────────────────────────────────────────────────────────
    if not args.no_viz:
        visualize_results(results, args.output_dir, preprocessor)

    print("\n완료! 결과 파일:")
    print(f"  - {result_csv}")
    if not args.no_viz:
        print(f"  - {args.output_dir}/cf_report.png")


if __name__ == "__main__":
    main()
