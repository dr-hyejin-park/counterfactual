"""
대규모 병렬 CF 생성 — 속도 최적화 버전

[병목 분석]
  800K 고객 × 8후보 × 400스텝 × forward pass ≒ 2560만 회
  batch=32 → 25,000 배치 → GPU 가동률 매우 낮음

[핵심 최적화 3가지]
  1. DDIM 샘플링: 400스텝 → 50스텝 (8× 빠름, 품질 유사)
  2. 대형 배치:   GPU 512+, MPS 128, CPU 32
  3. 2-Stage 처리:
       1단계 — 모든 고객에 fast CF (DDIM 50스텝, 후보 3개, refine 없음)
       2단계 — 미달성 고객에만 deep CF (후보 5개, refine 10스텝)
               → refine 비용을 invalid 비율만큼만 부담

[예상 처리 시간 (A100 80GB 기준)]
  batch=512, DDIM 50스텝, 2-stage:
    1단계: 800K × 3 / 512 = 4,688 배치 × 50스텝 ≈ 10분
    2단계: ~200K × 5 / 512 = 1,953 배치 × 50스텝 + refine ≈ 7분
    합계: ≈ 17~20분
"""

import os
import time
import math
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Optional, List, Dict

from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, CF_CONFIG
from cf_engine.generator import TabDiffCFGenerator, build_treatment_rows, build_action_rows

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


# ─── 배치 크기 추천 ─────────────────────────────────────────────────────────────

def recommend_batch_size(device: str, input_dim: int, num_timesteps: int) -> int:
    """
    디바이스별 최적 배치 크기 추천.
    DDIM 사용 시 스텝 수가 줄어 더 큰 배치가 가능합니다.
    """
    if device == "cuda":
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            vram_gb = 16.0
        # gradient 포함 메모리: batch × dim × 4bytes × 10(모델·중간값) 추정
        bytes_per_sample = input_dim * 4 * 10
        max_batch = int((vram_gb * 0.6 * 1e9) / max(bytes_per_sample, 1))
        recommended = min(max(32, max_batch), 1024)
        print(f"  추천 배치 크기: {recommended} (VRAM {vram_gb:.1f}GB 기준)")
        return recommended
    elif device == "mps":
        print("  추천 배치 크기: 128 (Apple MPS 기준)")
        return 128
    else:
        print("  추천 배치 크기: 32 (CPU 기준, GPU 사용 시 --device cuda 권장)")
        return 32


# ─── 단일 배치 CF 생성 (내부 헬퍼) ──────────────────────────────────────────────

def _run_batch(
    batch_df: pd.DataFrame,
    cf_generator: TabDiffCFGenerator,
    num_candidates: int,
    start_offset: int,
) -> List[dict]:
    """하나의 mini-batch에 대해 CF를 생성하고 결과 dict 리스트 반환"""
    cf_list = cf_generator.generate_for_batch(
        customer_df=batch_df,
        num_candidates=num_candidates,
    )
    results = []
    for i, (_, row) in enumerate(batch_df.iterrows()):
        best_cf = cf_generator.select_best_cf(cf_list[i], row)
        x_cf_enc = cf_generator.preprocessor.transform(
            pd.DataFrame([best_cf[ALL_FEATURES]])
        )
        x_cf_t = torch.tensor(x_cf_enc, dtype=torch.float32, device=cf_generator.device)
        with torch.no_grad():
            cf_prob = float(cf_generator.classifier.predict_proba(x_cf_t).cpu().numpy()[0])

        results.append({
            "customer_id":    row.get("customer_id", f"C_{start_offset + i}"),
            "factual":        row,
            "counterfactual": best_cf,
            "factual_prob":   float(row.get("risk_prob", 0.0)),
            "cf_prob":        cf_prob,
            "cf_valid":       cf_prob < 0.5,
            "treatments":     cf_generator._extract_treatments(row, best_cf),
            "num_changes":    len(cf_generator._extract_treatments(row, best_cf)),
        })
    return results


# ─── 스트리밍 CSV 저장 ───────────────────────────────────────────────────────────

def _stream_write(
    batch_results: List[dict],
    output_wide_path: Optional[str],
    output_long_path: Optional[str],
    write_wide_header: bool,
    write_long_header: bool,
):
    """배치 결과를 즉시 CSV에 append (메모리 절약)"""
    if output_wide_path:
        wide_rows = [build_treatment_rows(r) for r in batch_results]
        pd.DataFrame(wide_rows).to_csv(
            output_wide_path, mode="a", index=False,
            header=write_wide_header, encoding="utf-8-sig",
        )
    if output_long_path:
        long_rows = []
        for r in batch_results:
            long_rows.extend(build_action_rows(r))
        if long_rows:
            pd.DataFrame(long_rows).to_csv(
                output_long_path, mode="a", index=False,
                header=write_long_header, encoding="utf-8-sig",
            )


# ─── 2-Stage 대규모 CF 생성 ─────────────────────────────────────────────────────

def generate_cf_two_stage(
    top_df: pd.DataFrame,
    cf_generator: TabDiffCFGenerator,
    batch_size: int = 256,
    # Stage 1: 전체 고객 fast CF
    stage1_candidates: int = 3,
    stage1_ddim_steps: int = 50,
    stage1_refine_steps: int = 0,
    # Stage 2: 미달성 고객 deep CF
    stage2_candidates: int = 5,
    stage2_ddim_steps: int = 50,
    stage2_refine_steps: int = 10,
    # 공통
    checkpoint_dir: Optional[str] = None,
    output_wide_path: Optional[str] = None,
    output_long_path: Optional[str] = None,
    resume: bool = True,
    verbose: bool = True,
) -> List[dict]:
    """
    2단계 CF 생성 — 대규모(80만 명+) 최적화 전략

    Stage 1: 모든 고객에 DDIM 50스텝 fast CF (후보 3개, refine 없음)
             → 전체를 빠르게 커버
    Stage 2: Stage 1에서 CF가 무효(cf_prob >= 0.5)인 고객에만
             추가 후보 + refine 적용
             → 비용을 미달성 비율만큼만 추가 부담

    처리 시간 예시 (A100 기준, batch=512, DDIM 50스텝):
      800K 전체  1단계: ~10분
      ~200K 재시도 2단계: ~7분
      합계: ≈ 17분 (vs DDPM 400스텝 batch=32: 수십 시간)
    """
    os.makedirs(checkpoint_dir, exist_ok=True) if checkpoint_dir else None

    # ── 재시작: 이미 처리된 고객 스킵 ────────────────────────────────────────
    done_ids: set = set()
    if resume and output_wide_path and os.path.exists(output_wide_path):
        done_df = pd.read_csv(output_wide_path, usecols=["customer_id"])
        done_ids = set(done_df["customer_id"].values)
        print(f"  재시작 감지: {len(done_ids):,}명 이미 완료 — 나머지만 처리")

    pending = top_df[~top_df["customer_id"].isin(done_ids)].reset_index(drop=True) \
              if done_ids else top_df.copy()
    if len(pending) == 0:
        print("  모든 고객 처리 완료 (resume)")
        return []

    all_results: List[dict] = []
    write_wide_header = len(done_ids) == 0
    write_long_header = len(done_ids) == 0

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1 — 전체 고객 Fast CF
    # ══════════════════════════════════════════════════════════════════════════
    orig_cfg = dict(cf_generator.cfg)
    cf_generator.cfg = dict(orig_cfg)
    cf_generator.cfg["ddim_steps"]    = stage1_ddim_steps
    cf_generator.cfg["refine_steps"]  = stage1_refine_steps
    cf_generator.cfg["num_cf_samples"] = stage1_candidates

    t0 = time.time()
    n_batches = math.ceil(len(pending) / batch_size)
    pbar_desc = f"[1단계] Fast CF (DDIM {stage1_ddim_steps}스텝, 후보 {stage1_candidates}개)"
    pbar = tqdm(total=len(pending), desc=pbar_desc, unit="명") if verbose else None

    stage1_results: Dict[str, dict] = {}   # customer_id → result
    first_write = True

    for b_idx in range(n_batches):
        s = b_idx * batch_size
        e = min(s + batch_size, len(pending))
        batch_df = pending.iloc[s:e]

        for r in _run_batch(batch_df, cf_generator, stage1_candidates, s):
            stage1_results[r["customer_id"]] = r
            all_results.append(r)

        batch_list = list(stage1_results.values())[-len(batch_df):]
        _stream_write(batch_list, output_wide_path, output_long_path,
                      write_wide_header and first_write,
                      write_long_header and first_write)
        first_write = False

        if pbar:
            pbar.update(len(batch_df))

    if pbar:
        pbar.close()

    elapsed1 = time.time() - t0
    n_valid1  = sum(1 for r in stage1_results.values() if r["cf_valid"])
    n_invalid = len(stage1_results) - n_valid1
    n_stage1    = len(stage1_results)
    valid1_pct  = n_valid1 / max(n_stage1, 1) * 100
    elapsed1_s  = int(elapsed1)
    print(f"  1단계 완료: {n_stage1:,}명 | 유효 {n_valid1:,}명 "
          f"({valid1_pct:.1f}%) | {elapsed1_s}초")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2 — 미달성 고객만 Deep CF
    # ══════════════════════════════════════════════════════════════════════════
    invalid_ids = [cid for cid, r in stage1_results.items() if not r["cf_valid"]]

    if invalid_ids and stage2_refine_steps > 0:
        print(f"\n  [2단계] Deep CF 재시도: {len(invalid_ids):,}명 "
              f"(후보 {stage2_candidates}개, refine {stage2_refine_steps}스텝)")

        cf_generator.cfg["ddim_steps"]     = stage2_ddim_steps
        cf_generator.cfg["refine_steps"]   = stage2_refine_steps
        cf_generator.cfg["num_cf_samples"] = stage2_candidates

        invalid_df = top_df[top_df["customer_id"].isin(set(invalid_ids))].reset_index(drop=True)
        n_batches2 = math.ceil(len(invalid_df) / batch_size)
        pbar2_desc = (f"[2단계] Deep CF "
                      f"(refine {stage2_refine_steps}스텝, 후보 {stage2_candidates}개)")
        pbar2 = tqdm(total=len(invalid_df), desc=pbar2_desc, unit="명") if verbose else None

        t2 = time.time()
        improved = 0
        for b_idx in range(n_batches2):
            s = b_idx * batch_size
            e = min(s + batch_size, len(invalid_df))
            batch_df = invalid_df.iloc[s:e]

            for r in _run_batch(batch_df, cf_generator, stage2_candidates, s):
                cid = r["customer_id"]
                # Stage 2가 더 좋으면 교체 (낮은 cf_prob = 더 활성에 가까움)
                if r["cf_prob"] < stage1_results[cid]["cf_prob"]:
                    stage1_results[cid] = r
                    improved += 1

                # all_results에도 반영
                for idx, existing in enumerate(all_results):
                    if existing["customer_id"] == cid:
                        all_results[idx] = stage1_results[cid]
                        break

            if pbar2:
                pbar2.update(len(batch_df))

        if pbar2:
            pbar2.close()

        # 2단계 결과로 wide/long CSV 전체 재기록
        if output_wide_path:
            wide_rows = [build_treatment_rows(r) for r in stage1_results.values()]
            pd.DataFrame(wide_rows).to_csv(
                output_wide_path, index=False, encoding="utf-8-sig"
            )
        if output_long_path:
            long_rows = []
            for r in stage1_results.values():
                long_rows.extend(build_action_rows(r))
            if long_rows:
                pd.DataFrame(long_rows).to_csv(
                    output_long_path, index=False, encoding="utf-8-sig"
                )

        elapsed2  = time.time() - t2
        n_valid2   = sum(1 for r in stage1_results.values() if r["cf_valid"])
        valid1_pct = n_valid1 / max(n_stage1, 1) * 100
        valid2_pct = n_valid2 / max(n_stage1, 1) * 100
        elapsed2_s = int(elapsed2)
        print(f"  2단계 완료: 유효율 {valid1_pct:.1f}% → {valid2_pct:.1f}% "
              f"(개선 {improved:,}명) | {elapsed2_s}초")

    # config 복원
    cf_generator.cfg = orig_cfg
    return all_results


# ─── 단순 배치 CF 생성 (하위 호환) ──────────────────────────────────────────────

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
    단일 패스 배치 CF 생성 (2-stage 없이 단순 처리).
    소규모 테스트나 --fast 모드에 적합.
    대규모(10만 명+) 처리에는 generate_cf_two_stage 사용 권장.
    """
    os.makedirs(checkpoint_dir, exist_ok=True) if checkpoint_dir else None

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
    write_wide_header = len(done_ids) == 0
    write_long_header = len(done_ids) == 0

    t0   = time.time()
    pbar = tqdm(total=len(pending), desc="배치 CF 생성", unit="명") if verbose else None

    for b_idx in range(n_batches):
        s = b_idx * batch_size
        e = min(s + batch_size, len(pending))
        batch_df = pending.iloc[s:e]

        batch_results = _run_batch(batch_df, cf_generator, num_candidates, s)
        all_results.extend(batch_results)

        _stream_write(batch_results, output_wide_path, output_long_path,
                      write_wide_header, write_long_header)
        write_wide_header = False
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
