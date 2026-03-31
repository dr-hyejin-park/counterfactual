"""
대규모 고객 데이터 청크 로딩

800~900만 명 규모 데이터를 메모리 초과 없이 처리합니다.
CSV / Parquet 모두 지원, 청크 단위로 순차 yield.
"""

import os
import numpy as np
import pandas as pd
from typing import Iterator, Optional, Tuple


def iter_chunks(
    data_path: str,
    chunk_size: int = 100_000,
    usecols: Optional[list] = None,
    dtype_map: Optional[dict] = None,
) -> Iterator[pd.DataFrame]:
    """
    대용량 파일을 chunk_size 행씩 yield

    지원 형식:
      - CSV  (.csv)
      - Parquet (.parquet, .pq) — 열 단위 압축으로 훨씬 빠름

    Args:
        data_path:  파일 경로
        chunk_size: 한 번에 읽을 행 수 (메모리 ~ chunk_size × n_cols × 8 bytes)
        usecols:    필요한 열만 읽어 메모리 절약 (None = 전체)
        dtype_map:  열별 dtype 지정 (메모리 최적화용)

    Example:
        for chunk in iter_chunks("customers.csv", chunk_size=500_000):
            process(chunk)
    """
    ext = os.path.splitext(data_path)[1].lower()

    if ext in (".parquet", ".pq"):
        # Parquet: 열 단위 압축 → 대규모 데이터에 적합
        import pyarrow.parquet as pq
        pf     = pq.ParquetFile(data_path)
        offset = 0
        for batch in pf.iter_batches(batch_size=chunk_size, columns=usecols):
            df = batch.to_pandas()
            if dtype_map:
                df = df.astype({k: v for k, v in dtype_map.items() if k in df.columns})
            yield df
            offset += len(df)

    else:
        # CSV
        reader = pd.read_csv(
            data_path,
            chunksize=chunk_size,
            usecols=usecols,
            dtype=dtype_map,
            low_memory=False,
        )
        for chunk in reader:
            yield chunk


def count_rows(data_path: str) -> int:
    """파일 총 행 수를 메모리 효율적으로 계산"""
    ext = os.path.splitext(data_path)[1].lower()
    if ext in (".parquet", ".pq"):
        import pyarrow.parquet as pq
        return pq.read_metadata(data_path).num_rows
    else:
        # 첫 행만 읽어 컬럼 확인, 이후 행 수 계산
        with open(data_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f) - 1  # 헤더 제외


def csv_to_parquet(
    csv_path: str,
    parquet_path: str,
    chunk_size: int = 500_000,
    compression: str = "snappy",
) -> None:
    """
    대용량 CSV → Parquet 변환
    - Snappy 압축: ~3-5배 용량 절감 + 더 빠른 읽기
    - 메모리 효율: 청크 단위로 변환

    Example:
        csv_to_parquet("9M_customers.csv", "9M_customers.parquet")
        # CSV 9GB → Parquet ~2-3GB, 읽기 속도 10x 향상
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    for i, chunk in enumerate(iter_chunks(csv_path, chunk_size=chunk_size)):
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression=compression)
        writer.write_table(table)
        if (i + 1) % 10 == 0:
            print(f"  변환 중: {(i+1)*chunk_size:,}행 처리됨...")

    if writer:
        writer.close()
    print(f"  Parquet 저장 완료: {parquet_path}")
