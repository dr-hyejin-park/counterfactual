"""
무실적 위험 상위 10% 고객 전체 Treatment 매핑 (대규모 지원)

[기본 실행]
  python top10_wakeup.py --data_path data/hyundai_card_customers.csv

[대규모 (800~900만 명) 실행]
  python top10_wakeup.py \\
      --data_path /data/9M_customers.parquet \\
      --chunk_size 200000 \\           # 청크당 20만 명 (메모리 ~400MB)
      --batch_size 32 \\               # 배치 CF 생성 크기 (GPU 시 512+)
      --top_pct 10 \\
      --max_cf_customers 50000 \\      # CF 생성 상한 (상위 10%가 너무 많을 때)
      --device cuda \\                 # GPU 자동 감지: --device auto
      --output_format parquet \\       # 대용량 출력은 Parquet 권장
      --resume                         # 중단 후 재개

[옵션]
  --top_pct N          : 상위 N% 선별 (기본 10)
  --max_cf_customers N : CF 생성 최대 인원 (기본 무제한)
  --device auto|cpu|cuda|mps
  --chunk_size N       : 스코어링 청크 크기 (기본 100000)
  --batch_size N       : CF 배치 크기 (기본 자동 추천)
  --num_candidates N   : 고객당 CF 후보 수 (기본 2, 대규모 시 1~2 권장)
  --fast               : timestep 100 (빠른 생성)
  --output_format csv|parquet
  --resume             : 체크포인트 재시작
  --no_viz             : 시각화 생략

[출력]
  results/top10_risk_scored.{csv|parquet}      — 전체 고객 위험도
  results/top10_treatment_mapping.csv          — 와이드 포맷 (고객 × 피처 변화)
  results/top10_treatment_actions.csv          — 롱 포맷 (고객 × 캠페인 액션)
  results/top10_heatmap.png                    — 히트맵
  results/top10_summary.png                    — 종합 분석
"""

import os, sys, argparse, time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, CF_CONFIG, TABDIFF_CONFIG
from utils.preprocessing import HyundaiCardPreprocessor
from models.tabdiff import TabDiff
from models.classifier import InactiveRiskClassifier
from cf_engine.generator import (
    TabDiffCFGenerator, print_cf_report, build_treatment_rows, build_action_rows,
    _feat_name_kr, _feat_unit, _translate_cat,
)
from scale.data_loader import iter_chunks, count_rows
from scale.batch_scorer import score_in_chunks, select_top_pct, detect_device
from scale.parallel_cf import generate_cf_batched, recommend_batch_size

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

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


def parse_args():
    p = argparse.ArgumentParser(
        description="무실적 위험 상위 10% — 전체 Treatment 매핑 (대규모 지원)"
    )
    p.add_argument("--checkpoint_dir",   default="checkpoints")
    p.add_argument("--data_path",        default="data/hyundai_card_customers.csv")
    p.add_argument("--output_dir",       default="results")
    p.add_argument("--top_pct",          type=float, default=10.0)
    p.add_argument("--max_cf_customers", type=int,   default=None,
                   help="CF 생성 상한 인원 (None=무제한, 대규모 시 50000 권장)")
    p.add_argument("--min_risk",         type=float, default=0.0,
                   help="CF 대상 최소 위험도 (예: 0.6 → 60%% 이상만)")
    # 디바이스
    p.add_argument("--device",           default="auto",
                   help="cpu | cuda | mps | auto (자동 감지)")
    # 대규모 처리
    p.add_argument("--chunk_size",       type=int,   default=100_000,
                   help="스코어링 청크 크기 (기본 100,000)")
    p.add_argument("--batch_size",       type=int,   default=None,
                   help="CF 배치 크기 (None=자동 추천)")
    p.add_argument("--num_candidates",   type=int,   default=2,
                   help="고객당 CF 후보 수 (대규모 시 2 권장)")
    # CF timestep
    p.add_argument("--fast",             action="store_true",
                   help="빠른 CF: timestep=100")
    p.add_argument("--cf_timesteps",     type=int,   default=None,
                   help="CF 역확산 스텝 수 (None → fast:100, 일반:200)")
    # 출력
    p.add_argument("--output_format",    default="csv", choices=["csv", "parquet"])
    p.add_argument("--resume",           action="store_true",
                   help="중단 후 체크포인트에서 재개")
    p.add_argument("--no_viz",           action="store_true")
    p.add_argument("--no_detail",        action="store_true",
                   help="고객별 상세 보고서 콘솔 출력 생략")
    return p.parse_args()


def load_models(ckpt_dir: str, device: str):
    preprocessor = HyundaiCardPreprocessor.load(f"{ckpt_dir}/preprocessor.pkl")
    clf_ckpt   = torch.load(f"{ckpt_dir}/classifier.pt",  map_location=device)
    hidden_dims = tuple(clf_ckpt.get("hidden_dims", (128, 64, 32)))
    classifier = InactiveRiskClassifier(
        input_dim=clf_ckpt["input_dim"],
        hidden_dims=hidden_dims,
    )
    classifier.load_state_dict(clf_ckpt["model_state"])
    classifier.to(device).eval()
    td_ckpt = torch.load(f"{ckpt_dir}/tabdiff.pt", map_location=device)
    cfg     = td_ckpt["config"]
    tabdiff = TabDiff(
        input_dim=td_ckpt["input_dim"],
        num_timesteps=cfg["num_timesteps"],
        beta_start=cfg["beta_start"], beta_end=cfg["beta_end"],
        hidden_dim=cfg["hidden_dim"], num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"], dropout=cfg["dropout"],
        device=device,
    )
    tabdiff.load_state_dict(td_ckpt["model_state"])
    tabdiff.to(device).eval()
    return preprocessor, classifier, tabdiff


# ─── 콘솔 출력 ──────────────────────────────────────────────────────────────

def print_treatment_table(results: List[Dict], top_pct: float) -> None:
    n = len(results)
    print("\n" + "=" * 82)
    print(f"  무실적 위험 상위 {top_pct:.0f}% 고객 전체 Treatment 매핑 ({n:,}명)")
    print("=" * 82)
    print(f"  {'#':>4}  {'고객 ID':<12} {'위험도':>6} {'CF 후':>6} {'감소폭':>7}  "
          f"{'CF':^5} {'등급':<6} {'주요 캠페인 액션 (상위 3개)'}")
    print("  " + "─" * 78)
    for i, r in enumerate(results, 1):
        mark  = "✓" if r["cf_valid"] else "✗"
        tier  = r["factual"].get("card_tier", "-")
        risk  = r["factual_prob"] * 100
        cf_p  = r["cf_prob"]     * 100
        drop  = (r["factual_prob"] - r["cf_prob"]) * 100
        tags  = [ACTION_TAG.get(t["feature"], t["feature"]) for t in r["treatments"][:3]]
        print(f"  {i:>4}  {r['customer_id']:<12} {risk:>5.1f}% "
              f"{cf_p:>5.1f}% {drop:>+6.1f}%p  {mark:^5} {tier:<6} "
              f"{' / '.join(tags) or '-'}")
    n_valid  = sum(1 for r in results if r["cf_valid"])
    avg_drop = np.mean([r["factual_prob"] - r["cf_prob"] for r in results]) * 100
    print("  " + "─" * 78)
    print(f"  CF 유효: {n_valid:,}/{n:,}명 ({n_valid/n*100:.0f}%)  "
          f"평균 위험 감소: {avg_drop:+.1f}%p")
    print("=" * 82)


def print_aggregated_summary(results: List[Dict], top_pct: float) -> None:
    print("\n" + "=" * 70)
    print(f"  [무실적 위험 상위 {top_pct:.0f}%] 전체 Treatment 집계 요약")
    print("=" * 70)

    # 전체 액션 빈도
    action_counts: Dict[str, int] = {}
    for r in results:
        for t in r["treatments"]:
            tag = ACTION_TAG.get(t["feature"], t["feature"])
            action_counts[tag] = action_counts.get(tag, 0) + 1
    n = len(results)
    print(f"\n  [필요 캠페인 액션 전체 빈도 — {n}명 기준]")
    print(f"  {'액션':16} {'고객 수':>8} {'비율':>8}   바 차트")
    print("  " + "─" * 55)
    for tag, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        pct  = cnt / n * 100
        bar  = "█" * int(pct / 3)
        print(f"  {tag:16} {cnt:>8,}명  {pct:>6.1f}%  {bar}")

    # 카드 등급별 평균 위험도 감소폭
    tier_stats: Dict[str, list] = {}
    for r in results:
        tier = r["factual"].get("card_tier", "Unknown")
        tier_stats.setdefault(tier, []).append(r["factual_prob"] - r["cf_prob"])
    print(f"\n  [카드 등급별 평균 위험도 감소폭]")
    for tier in ["The", "Black", "Red", "Blue"]:
        if tier in tier_stats:
            avg = np.mean(tier_stats[tier]) * 100
            cnt = len(tier_stats[tier])
            print(f"    {tier:6} Card: {avg:+.1f}%p  (n={cnt:,})")

    # 위험도 구간별 통계
    print(f"\n  [위험도 구간별 처방 복잡도]")
    bins = [(0.5, 0.65, "Medium (50-65%)"),
            (0.65, 0.80, "High   (65-80%)"),
            (0.80, 1.01, "Critical (80%+)")]
    for lo, hi, label in bins:
        grp = [r for r in results if lo <= r["factual_prob"] < hi]
        if grp:
            avg_chg = np.mean([r["num_changes"] for r in grp])
            avg_drp = np.mean([r["factual_prob"] - r["cf_prob"] for r in grp]) * 100
            valid_r = np.mean([r["cf_valid"] for r in grp]) * 100
            print(f"    {label}: n={len(grp):>5,}명  "
                  f"평균변화={avg_chg:.1f}개  위험↓={avg_drp:+.1f}%p  "
                  f"유효={valid_r:.0f}%")
    print("=" * 70)


# ─── 시각화 ─────────────────────────────────────────────────────────────────

def visualize_heatmap(results: List[Dict], wide_df: pd.DataFrame,
                       output_dir: str, max_display: int = 100) -> None:
    """고객 × 수치형 피처 변화율 히트맵 (최대 max_display명 표시)"""
    display_df = wide_df.head(max_display)
    pct_cols   = [f"{f}_pct" for f in NUMERICAL_FEATURES]
    matrix     = np.clip(display_df[pct_cols].values.astype(float), -200, 400)

    n_show = len(display_df)
    fig_h  = max(8, n_show * 0.17)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#E74C3C", "#ffffff", "#2ECC71"], N=256
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap,
                   vmin=-200, vmax=300, interpolation="nearest")
    ax.set_xticks(range(len(NUMERICAL_FEATURES)))
    feat_en = {
        "months_since_last_txn": "Last Txn(mo)",
        "avg_spending_3m":       "Spending 3M",
        "avg_spending_prev3m":   "Spending Prev",
        "spending_trend_ratio":  "Trend Ratio",
        "monthly_txn_count_3m":  "Txn Count",
        "num_missed_months":     "Missed Mo.",
        "days_since_app_login":  "App Gap(d)",
        "loyalty_points_balance":"Points Bal.",
        "num_active_benefits":   "Benefits",
        "years_as_customer":     "Customer Yr",
    }
    ax.set_xticklabels(
        [feat_en.get(f, f) for f in NUMERICAL_FEATURES],
        rotation=35, ha="right", fontsize=8,
    )
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(
        [f"{row.customer_id} ({row.risk_prob*100:.0f}%)"
         for _, row in display_df[["customer_id","risk_prob"]].iterrows()],
        fontsize=6,
    )
    title_sfx = f" (showing top {n_show})" if len(wide_df) > max_display else ""
    ax.set_title(
        f"Inactive Risk Top 10% — Feature Change Heatmap{title_sfx}\n"
        "(Green=Increase Needed, Red=Decrease Needed, White=No Change)",
        fontsize=11,
    )
    for i in range(n_show):
        for j in range(len(NUMERICAL_FEATURES)):
            val = matrix[i, j]
            if abs(val) > 10:
                ax.text(j, i, f"{val:+.0f}%", ha="center", va="center",
                        fontsize=4.5, color="black")
    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.01)
    cbar.set_label("% Change Required", fontsize=9)
    plt.tight_layout()
    path = f"{output_dir}/top10_heatmap.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  히트맵: {path}")


def visualize_summary(results: List[Dict], scored_df: pd.DataFrame,
                       top_pct: float, output_dir: str) -> None:
    """종합 분석 시각화 (4개 패널)"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        f"Inactive Risk Top {top_pct:.0f}% — Wake-up Campaign Treatment Analysis\n"
        f"(n={len(results):,} customers)",
        fontsize=13, fontweight="bold",
    )

    # 1. 전체 위험도 분포
    ax = axes[0, 0]
    all_probs = scored_df["risk_prob"].values
    threshold = (min(r["factual_prob"] for r in results)) if results else 0
    ax.hist(all_probs, bins=50, color="#95A5A6", alpha=0.7, edgecolor="white")
    ax.axvline(x=threshold, color="#E74C3C", linewidth=2, linestyle="--",
               label=f"Top {top_pct:.0f}% threshold ({threshold:.2f})")
    ax.fill_betweenx([0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100],
                     threshold, 1.0, alpha=0.12, color="#E74C3C")
    ax.set_xlabel("P(Inactive Risk=1)")
    ax.set_ylabel("Customers")
    ax.set_title(f"Risk Score Distribution\n"
                 f"Total: {len(scored_df):,} | Top{top_pct:.0f}%: {len(results):,}")
    ax.legend(fontsize=8)

    # 2. 액션 빈도
    ax = axes[0, 1]
    action_counts: Dict[str, int] = {}
    for r in results:
        for t in r["treatments"]:
            tag = ACTION_TAG.get(t["feature"], t["feature"])
            action_counts[tag] = action_counts.get(tag, 0) + 1
    if action_counts:
        sorted_a = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        names = [a[0] for a in sorted_a]
        pcts  = [a[1] / len(results) * 100 for a in sorted_a]
        bars  = ax.barh(names, pcts, color=sns.color_palette("RdYlGn_r", len(names)), alpha=0.85)
        ax.set_xlabel("% of Top-10% Customers")
        ax.set_title("Wake-up Action Frequency (Top 10)")
        ax.set_xlim(0, 115)
        for bar, pct in zip(bars, pcts):
            ax.text(pct + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{pct:.0f}%", va="center", fontsize=8)

    # 3. 카드 등급 × 위험 감소폭
    ax = axes[1, 0]
    tier_drops: Dict[str, list] = {}
    tier_valid: Dict[str, list] = {}
    for r in results:
        t = r["factual"].get("card_tier", "Unknown")
        tier_drops.setdefault(t, []).append((r["factual_prob"] - r["cf_prob"]) * 100)
        tier_valid.setdefault(t, []).append(int(r["cf_valid"]))
    tiers = [t for t in ["The", "Black", "Red", "Blue"] if t in tier_drops]
    if tiers:
        x = np.arange(len(tiers))
        w = 0.38
        avg_d  = [np.mean(tier_drops[t]) for t in tiers]
        valid_r = [np.mean(tier_valid[t]) * 100 for t in tiers]
        b1 = ax.bar(x - w/2, avg_d,   w, label="Avg Risk Drop (%p)", color="#2ECC71", alpha=0.8)
        b2 = ax.bar(x + w/2, valid_r, w, label="CF Validity (%)",    color="#3498DB", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n(n={len(tier_drops[t]):,})" for t in tiers])
        ax.set_ylabel("Value (%)")
        ax.set_title("Risk Drop & CF Validity by Card Tier")
        ax.legend(fontsize=9)
        for bar, v in zip(b1, avg_d):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", fontsize=8)

    # 4. 위험도 구간 × 처방 복잡도
    ax  = axes[1, 1]
    ax2 = ax.twinx()
    bins = [(0.5,0.65,"Medium\n(50-65%)"), (0.65,0.80,"High\n(65-80%)"), (0.80,1.01,"Critical\n(80%+)")]
    labels, avg_chgs, valid_rs, cnts = [], [], [], []
    for lo, hi, lbl in bins:
        grp = [r for r in results if lo <= r["factual_prob"] < hi]
        labels.append(lbl)
        cnts.append(len(grp))
        avg_chgs.append(np.mean([r["num_changes"] for r in grp]) if grp else 0)
        valid_rs.append(np.mean([r["cf_valid"]    for r in grp]) * 100 if grp else 0)
    x = np.arange(len(labels))
    ax.bar(x, avg_chgs, color=["#F39C12","#E67E22","#E74C3C"], alpha=0.8, label="Avg Changes")
    ax2.plot(x, valid_rs, "D--", color="#2ECC71", linewidth=2, markersize=8, label="CF Validity (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={c:,})" for l, c in zip(labels, cnts)])
    ax.set_ylabel("Avg Feature Changes")
    ax2.set_ylabel("CF Validity (%)")
    ax2.set_ylim(0, 110)
    ax.set_title("Risk Level vs Treatment Complexity")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper left")

    plt.tight_layout()
    path = f"{output_dir}/top10_summary.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  종합 분석: {path}")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    t_start = time.time()

    print("=" * 72)
    print("  현대카드 무실적 위험 상위 10% 고객 — 전체 Treatment 매핑 시스템")
    print("=" * 72)

    # ── 디바이스 결정 ─────────────────────────────────────────────────────────
    print("\n[1단계] 디바이스 & 모델 설정")
    if args.device == "auto":
        device = detect_device(prefer_gpu=True)
    else:
        device = args.device
    print(f"  사용 디바이스: {device.upper()}")

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    preprocessor, classifier, tabdiff = load_models(args.checkpoint_dir, device)
    print(f"  입력 차원: {preprocessor.total_dim}  |  "
          f"TabDiff 파라미터: {sum(p.numel() for p in tabdiff.parameters()):,}")

    # ── CF 설정 ───────────────────────────────────────────────────────────────
    cf_cfg = dict(CF_CONFIG)
    if args.cf_timesteps:
        cf_cfg["num_cf_timesteps"] = args.cf_timesteps
    elif args.fast:
        cf_cfg["num_cf_timesteps"] = 100
    else:
        cf_cfg["num_cf_timesteps"] = 200
    cf_cfg["num_cf_samples"] = args.num_candidates

    batch_size = args.batch_size or recommend_batch_size(
        device, preprocessor.total_dim, cf_cfg["num_cf_timesteps"]
    )
    print(f"  CF timestep: {cf_cfg['num_cf_timesteps']}  |  "
          f"배치 크기: {batch_size}  |  후보 수: {args.num_candidates}")

    # ── 전체 고객 스코어링 ────────────────────────────────────────────────────
    print(f"\n[2단계] 전체 고객 무실적 위험도 추론 (청크: {args.chunk_size:,}명)")
    score_ext = "parquet" if args.output_format == "parquet" else "csv"
    score_path = f"{args.output_dir}/top10_risk_scored.{score_ext}"

    scored_df = score_in_chunks(
        data_path=args.data_path,
        classifier=classifier,
        preprocessor=preprocessor,
        device=device,
        chunk_size=args.chunk_size,
        output_path=score_path,
        resume=args.resume,
    )
    total_customers = len(scored_df)
    print(f"  전체 고객: {total_customers:,}명  |  "
          f"평균 위험도: {scored_df['risk_prob'].mean()*100:.1f}%")

    # ── 상위 N% 선별 ──────────────────────────────────────────────────────────
    print(f"\n[3단계] 상위 {args.top_pct:.0f}% 고객 선별")
    top_df = select_top_pct(scored_df, top_pct=args.top_pct,
                             min_risk_threshold=args.min_risk)

    # ── 원본 피처 데이터 로드 (CF 생성에 전체 피처 컬럼 필요) ──────────────────
    top_ids = set(top_df["customer_id"].values)
    print(f"  원본 피처 로드 중 ({len(top_ids):,}명)...")
    full_rows = []
    for chunk in iter_chunks(args.data_path, chunk_size=args.chunk_size):
        if "customer_id" in chunk.columns:
            sub = chunk[chunk["customer_id"].isin(top_ids)]
        else:
            sub = chunk
        if len(sub) > 0:
            full_rows.append(sub)
    if full_rows:
        top_features_df = pd.concat(full_rows, ignore_index=True)
        top_df = top_features_df.merge(
            top_df[["customer_id", "risk_prob", "risk_label"]],
            on="customer_id", how="inner",
        )
        top_df = top_df.sort_values("risk_prob", ascending=False).reset_index(drop=True)

    # CF 생성 대상 인원 상한 적용
    if args.max_cf_customers and len(top_df) > args.max_cf_customers:
        print(f"  CF 생성 상한 적용: {len(top_df):,}명 → {args.max_cf_customers:,}명 "
              f"(--max_cf_customers {args.max_cf_customers})")
        top_df = top_df.head(args.max_cf_customers)

    tier_dist = top_df.get("card_tier", pd.Series()).value_counts().to_dict() \
                if "card_tier" in top_df.columns else {}
    print(f"  CF 생성 대상: {len(top_df):,}명  |  카드 등급 분포: {tier_dist}")
    print(f"  위험도 범위: {top_df['risk_prob'].min()*100:.1f}% ~ "
          f"{top_df['risk_prob'].max()*100:.1f}%")

    # ── CF 생성 (배치 병렬) ────────────────────────────────────────────────────
    print(f"\n[4단계] 배치 CF 생성 ({len(top_df):,}명, 배치={batch_size})")
    cf_generator = TabDiffCFGenerator(
        tabdiff_model=tabdiff,
        classifier=classifier,
        preprocessor=preprocessor,
        device=device,
        cf_config=cf_cfg,
    )

    wide_path = f"{args.output_dir}/top10_treatment_mapping.csv"
    long_path = f"{args.output_dir}/top10_treatment_actions.csv"

    # 재시작 시 기존 파일 있으면 이어쓰기, 없으면 새로 쓰기
    if not args.resume:
        for p in [wide_path, long_path]:
            if os.path.exists(p):
                os.remove(p)

    results = generate_cf_batched(
        top_df=top_df,
        cf_generator=cf_generator,
        batch_size=batch_size,
        num_candidates=args.num_candidates,
        checkpoint_dir=args.output_dir,
        output_wide_path=wide_path,
        output_long_path=long_path,
        resume=args.resume,
        verbose=True,
    )

    # 재시작으로 일부만 새로 생성된 경우 기존 파일 전체 재로드
    if os.path.exists(wide_path):
        wide_df = pd.read_csv(wide_path)
    else:
        wide_df = pd.DataFrame([build_treatment_rows(r) for r in results])

    # ── 결과가 없는 경우 처리 ──────────────────────────────────────────────────
    if len(results) == 0 and len(wide_df) == 0:
        print("  CF 생성 결과 없음. 종료.")
        return

    # ── 콘솔 출력 ─────────────────────────────────────────────────────────────
    print(f"\n[5단계] 결과 출력")
    if results:
        print_treatment_table(results, args.top_pct)
        if not args.no_detail:
            for r in results:
                print_cf_report(r)
        print_aggregated_summary(results, args.top_pct)

    # ── 저장 현황 ─────────────────────────────────────────────────────────────
    print(f"\n[6단계] 저장 현황")
    print(f"  위험도 스코어    : {score_path}  ({total_customers:,}명)")
    print(f"  와이드 Treatment : {wide_path}  ({len(wide_df):,}행 × {len(wide_df.columns)}열)")
    if os.path.exists(long_path):
        long_df = pd.read_csv(long_path)
        print(f"  롱 포맷(액션별) : {long_path}  ({len(long_df):,}행)")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    if not args.no_viz and results:
        print(f"\n[7단계] 시각화")
        visualize_heatmap(results, wide_df, args.output_dir, max_display=80)
        visualize_summary(results, scored_df, args.top_pct, args.output_dir)

    # ── 최종 요약 ─────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    n = len(results) or len(wide_df)
    n_valid = sum(1 for r in results if r["cf_valid"]) if results else \
              int(wide_df["cf_valid"].sum()) if "cf_valid" in wide_df.columns else 0
    avg_drop = np.mean([r["factual_prob"] - r["cf_prob"] for r in results]) * 100 if results else \
               float(wide_df["risk_drop"].mean() * 100) if "risk_drop" in wide_df.columns else 0

    print("\n" + "=" * 72)
    print(f"  전체 처리 완료   총 소요 시간: {elapsed:.1f}초")
    print(f"  전체 고객 수     : {total_customers:,}명")
    print(f"  상위 {args.top_pct:.0f}% CF 대상  : {n:,}명")
    print(f"  CF 유효 달성율   : {n_valid:,}/{n:,}명 ({n_valid/max(n,1)*100:.1f}%)")
    print(f"  평균 위험 감소폭 : {avg_drop:+.1f}%p")
    print("=" * 72)


if __name__ == "__main__":
    main()
