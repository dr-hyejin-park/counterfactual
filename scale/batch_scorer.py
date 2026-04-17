"""
대규모 배치 스코어링

800~900만 명을 청크 단위로 나눠 무실적 위험도를 추론합니다.
- 청크별 메모리 해제로 OOM 방지
- tqdm 진행률 + 소요 시간 추정
- 중간 결과를 분할 파일로 저장 (재시작 지원)
- GPU 자동 활용
"""

import os
import math
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Optional

from scale.data_loader import iter_chunks, count_rows
from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def score_in_chunks(
    data_path: str,
    classifier,
    preprocessor,
    device: str,
    chunk_size: int = 100_000,
    output_path: Optional[str] = None,
    resume: bool = True,
) -> pd.DataFrame:
    """
    전체 고객 데이터를 청크 단위로 읽어 무실적 위험도 추론

    Args:
        data_path:   고객 데이터 파일 경로 (CSV or Parquet)
        classifier:  학습된 분류기
        preprocessor: 전처리기
        device:      'cpu' or 'cuda'
        chunk_size:  한 번에 처리할 고객 수
        output_path: 결과 저장 경로 (None이면 메모리에만 유지)
        resume:      중간 저장 파일 존재 시 이어서 처리

    Returns:
        scored_df: customer_id + risk_prob 컬럼 포함 DataFrame
                   위험도 내림차순 정렬
    """
    # ── 재시작 처리: 이미 처리된 청크 파악 ──────────────────────────────────
    checkpoint_path = output_path.replace(".csv", "_ckpt.csv") if output_path else None
    processed_ids: set = set()
    all_results = []   # ← all_results를 먼저 선언해야 checkpoint 로드 시 append 가능

    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        done_df = pd.read_csv(checkpoint_path)          # 전체 컬럼 로드
        processed_ids = set(done_df["customer_id"].values)
        all_results.append(done_df)                     # ← 기존 결과를 최종 병합에 포함
        print(f"  체크포인트 발견: {len(processed_ids):,}명 이미 처리됨 — 이어서 시작")

    ckpt_file = open(checkpoint_path, "a") if checkpoint_path else None
    write_header = (len(processed_ids) == 0) if checkpoint_path else False

    total_rows  = count_rows(data_path)
    n_chunks    = math.ceil(total_rows / chunk_size)
    t0          = time.time()

    pbar = tqdm(total=total_rows, desc="무실적 위험도 추론", unit="명")

    for chunk in iter_chunks(data_path, chunk_size=chunk_size):
        # 이미 처리된 고객 건너뛰기 (재시작)
        if processed_ids and "customer_id" in chunk.columns:
            chunk = chunk[~chunk["customer_id"].isin(processed_ids)]
        if len(chunk) == 0:
            pbar.update(chunk_size)
            continue

        # 전처리 + 추론
        X_enc = preprocessor.transform(chunk[ALL_FEATURES])
        X_t   = torch.tensor(X_enc, dtype=torch.float32, device=device)

        with torch.no_grad():
            probs = classifier.predict_proba(X_t).cpu().numpy()

        result = chunk[["customer_id"]].copy() if "customer_id" in chunk.columns else chunk[[chunk.columns[0]]].copy()
        result.columns = ["customer_id"]

        # 보조 컬럼 포함 (분석용)
        for col in ["card_tier", "engagement_level", TARGET_COLUMN]:
            if col in chunk.columns:
                result[col] = chunk[col].values

        result["risk_prob"]  = probs.round(6)
        result["risk_label"] = (probs > 0.5).astype(np.int8)

        all_results.append(result)

        # 체크포인트 저장 (청크 단위)
        if ckpt_file is not None:
            result.to_csv(ckpt_file, index=False, header=write_header)
            write_header = False
            ckpt_file.flush()

        pbar.update(len(chunk))

        # 메모리 해제
        del X_enc, X_t, probs

    pbar.close()
    if ckpt_file:
        ckpt_file.close()

    elapsed = time.time() - t0
    scored  = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    scored  = scored.sort_values("risk_prob", ascending=False).reset_index(drop=True)

    print(f"  추론 완료: {len(scored):,}명 | 소요 시간: {elapsed:.1f}초 "
          f"({elapsed/max(len(scored),1)*1000:.2f}ms/고객)")

    if output_path:
        scored.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"  스코어 저장: {output_path}")
        # 체크포인트 파일 정리
        if checkpoint_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

    return scored


def select_top_pct(
    scored_df: pd.DataFrame,
    top_pct: float = 10.0,
    min_risk_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    스코어링 결과에서 상위 N% 고객 선별

    Args:
        scored_df:            score_in_chunks() 결과 (위험도 내림차순 정렬됨)
        top_pct:              상위 몇 % (기본 10%)
        min_risk_threshold:   최소 위험도 하한선 (예: 0.6 → 60% 이상만)
    Returns:
        top_df: 상위 N% 고객
    """
    n_top = max(1, int(len(scored_df) * top_pct / 100))
    top_df = scored_df.head(n_top).copy()

    if min_risk_threshold > 0:
        top_df = top_df[top_df["risk_prob"] >= min_risk_threshold]

    threshold = float(top_df["risk_prob"].min())
    print(f"  상위 {top_pct:.1f}% 선별: {len(top_df):,}명 (위험도 >= {threshold:.3f})")
    return top_df.reset_index(drop=True)


def select_boundary_pct(
    scored_df: pd.DataFrame,
    pct: float = 10.0,
) -> pd.DataFrame:
    """
    Decision boundary 근처 고객 선별 — |risk_prob - 0.5| 최소 N%

    [선택 근거]
    경계 근처 고객은:
      - 분류기가 가장 불확실하게 판단하는 구간 (0.5 ± δ)
      - 소수의 행동 변화만으로도 active ↔ inactive_risk 전환 가능
      - CF가 현실적으로 실행 가능한 범위 내에서 생성될 가능성 높음
      - 위험도 최상위 고객(risk ≈ 0.95+)은 대규모 변화 없이는 flip 어려움

    Args:
        scored_df: score_in_chunks() 결과 DataFrame
        pct:       선별 비율 (기본 10%)
    Returns:
        boundary_df: 경계 근처 N% 고객, |risk_prob - 0.5| 오름차순 정렬
    """
    n = max(1, int(len(scored_df) * pct / 100))
    dist = (scored_df["risk_prob"] - 0.5).abs()
    idx = dist.nsmallest(n).index
    boundary_df = scored_df.loc[idx].copy()
    boundary_df = boundary_df.sort_values(
        by="risk_prob", ascending=False
    ).reset_index(drop=True)

    lo = float(boundary_df["risk_prob"].min())
    hi = float(boundary_df["risk_prob"].max())
    mid_dist = float(dist.loc[idx].mean())
    print(f"  경계 근처 {pct:.1f}% 선별: {len(boundary_df):,}명  "
          f"(risk_prob {lo:.3f}~{hi:.3f}, |prob-0.5| 평균 {mid_dist:.3f})")
    return boundary_df


def detect_device(prefer_gpu: bool = True) -> str:
    """
    GPU/CPU 자동 감지

    Returns:
        'cuda'  — NVIDIA GPU 사용 가능
        'mps'   — Apple Silicon GPU
        'cpu'   — CPU fallback
    """
    if prefer_gpu:
        if torch.cuda.is_available():
            n   = torch.cuda.device_count()
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  GPU 감지: {gpu} × {n}개 ({vram:.1f} GB VRAM) → cuda 사용")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  GPU 감지: Apple Silicon MPS → mps 사용")
            return "mps"
    print("  GPU 없음 → CPU 사용 (대규모 처리 시 --device cuda 권장)")
    return "cpu"
