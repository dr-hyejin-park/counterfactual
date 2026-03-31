"""
대규모 병렬 CF 생성

800~900만 명 상위 10%는 80~90만 명.
현실적 처리 전략:
  1. 배치 diffusion — 여러 고객을 단일 텐서로 처리 (GPU 효율 극대화)
  2. 체크포인트 — 중단 후 재개 가능
  3. 스트리밍 출력 — 결과를 즉시 디스크에 기록 (메모리 절약)
  4. 멀티프로세스 — CPU 코어 병렬 활용 (GPU 없는 환경)
"""

import os
import time
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Optional, List

from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, CF_CONFIG
from cf_engine.generator import TabDiffCFGenerator, build_treatment_rows, build_action_rows

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def generate_cf_batched(
    top_df: pd.DataFrame,
    cf_generator: TabDiffCFGenerator,
    batch_size: int = 16,
    num_candidates: int = 2,
    checkpoint_dir: Optional[str] = None,
    output_wide_path: Optional[str] = None,
    output_long_path: Optional[str] = None,
    resume: bool = True,
    verbose: bool = True,
) -> List[dict]:
    """
    상위 10% 고객 전체에 대해 배치 CF 생성

    배치 처리 방식:
      batch_size=16, num_candidates=2 → 32개 샘플을 한 diffusion pass로 처리
      개별 처리 대비 ~batch_size배 빠른 처리 (특히 GPU 환경)

    체크포인트 전략:
      - 배치 완료마다 결과를 CSV에 append
      - 재시작 시 이미 완료된 고객 ID skip

    Args:
        top_df:           상위 N% 고객 DataFrame
        cf_generator:     TabDiffCFGenerator 인스턴스
        batch_size:       한 번에 처리할 고객 수 (GPU RAM에 맞게 조정)
        num_candidates:   고객당 CF 후보 수 (대규모 시 2 권장)
        checkpoint_dir:   중간 결과 저장 디렉토리
        output_wide_path: 와이드 포맷 CSV 경로
        output_long_path: 롱 포맷(액션별) CSV 경로
        resume:           중단 후 재개 여부
    Returns:
        all_results: 고객별 CF 결과 dict 리스트
    """
    os.makedirs(checkpoint_dir, exist_ok=True) if checkpoint_dir else None

    # ── 재시작: 이미 처리된 고객 확인 ─────────────────────────────────────────
    done_ids: set = set()
    if resume and output_wide_path and os.path.exists(output_wide_path):
        done_df = pd.read_csv(output_wide_path, usecols=["customer_id"])
        done_ids = set(done_df["customer_id"].values)
        print(f"  재시작 감지: {len(done_ids):,}명 이미 완료 — 나머지만 처리")

    pending = top_df[~top_df["customer_id"].isin(done_ids)].reset_index(drop=True) \
              if done_ids else top_df

    if len(pending) == 0:
        print("  모든 고객 처리 완료 (resume)")
        return []

    n_batches = math.ceil(len(pending) / batch_size)
    all_results: List[dict] = []

    # 스트리밍 CSV writer
    wide_writer_open = (output_wide_path is not None)
    long_writer_open = (output_long_path is not None)
    write_wide_header = (len(done_ids) == 0) if output_wide_path else False
    write_long_header = (len(done_ids) == 0) if output_long_path else False

    t0   = time.time()
    pbar = tqdm(total=len(pending), desc="배치 CF 생성", unit="명") if verbose else None

    for b_idx in range(n_batches):
        start = b_idx * batch_size
        end   = min(start + batch_size, len(pending))
        batch_df = pending.iloc[start:end]

        # ── 배치 CF 생성 ──────────────────────────────────────────────────────
        cf_list = cf_generator.generate_for_batch(
            customer_df=batch_df,
            num_candidates=num_candidates,
        )

        # ── 각 고객의 최적 CF 선택 + 결과 집계 ──────────────────────────────
        batch_results = []
        for i, (_, row) in enumerate(batch_df.iterrows()):
            cf_candidates = cf_list[i]
            best_cf = cf_generator.select_best_cf(cf_candidates, row)

            from config import CF_CONFIG
            x_cf_enc = cf_generator.preprocessor.transform(
                pd.DataFrame([best_cf[ALL_FEATURES]])
            )
            x_cf_t = torch.tensor(x_cf_enc, dtype=torch.float32, device=cf_generator.device)
            with torch.no_grad():
                cf_prob = float(cf_generator.classifier.predict_proba(x_cf_t).cpu().numpy()[0])

            treatments = cf_generator._extract_treatments(row, best_cf)

            result = {
                "customer_id":    row.get("customer_id", f"C_{start+i}"),
                "factual":        row,
                "counterfactual": best_cf,
                "factual_prob":   float(row.get("risk_prob", 0.0)),
                "cf_prob":        cf_prob,
                "cf_valid":       cf_prob < 0.5,
                "treatments":     treatments,
                "num_changes":    len(treatments),
            }
            batch_results.append(result)
            all_results.append(result)

        # ── 스트리밍 저장 (배치 완료 즉시 write) ───────────────────────────────
        if wide_writer_open:
            wide_rows = [build_treatment_rows(r) for r in batch_results]
            wide_batch = pd.DataFrame(wide_rows)
            wide_batch.to_csv(
                output_wide_path, mode="a", index=False,
                header=write_wide_header, encoding="utf-8-sig",
            )
            write_wide_header = False

        if long_writer_open:
            long_rows = []
            for r in batch_results:
                long_rows.extend(build_action_rows(r))
            if long_rows:
                long_batch = pd.DataFrame(long_rows)
                long_batch.to_csv(
                    output_long_path, mode="a", index=False,
                    header=write_long_header, encoding="utf-8-sig",
                )
                write_long_header = False

        if pbar:
            pbar.update(len(batch_df))

    if pbar:
        pbar.close()

    elapsed = time.time() - t0
    n_done  = len(pending)
    print(f"  CF 생성 완료: {n_done:,}명 | 소요: {elapsed:.1f}초 "
          f"({elapsed/max(n_done,1):.2f}초/고객)")
    return all_results


def recommend_batch_size(device: str, input_dim: int, num_timesteps: int) -> int:
    """
    디바이스/모델 크기에 따른 최적 배치 크기 추천

    GPU VRAM 추정:
      배치 메모리 ≈ batch_size × num_candidates × input_dim × num_timesteps × 4bytes × 2
    """
    if device == "cuda":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # 보수적 추정: VRAM의 50%만 배치에 할당
        bytes_per_sample = input_dim * num_timesteps * 4 * 2
        max_batch = int((vram_gb * 0.5 * 1e9) / bytes_per_sample)
        recommended = min(max(4, max_batch), 512)
        print(f"  추천 배치 크기: {recommended} (VRAM {vram_gb:.1f}GB 기준)")
        return recommended
    elif device == "mps":
        return 32
    else:
        # CPU: 메모리 여유롭지만 병렬 효율을 위해 16-32
        return 16
