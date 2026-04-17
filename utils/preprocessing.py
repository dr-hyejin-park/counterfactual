"""
데이터 전처리 파이프라인
수치형 피처: StandardScaler 정규화
범주형 피처: One-hot 인코딩 → 연속 확산 공간으로 변환
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List, Optional
import pickle
import os

from config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    CATEGORICAL_VALUES, TARGET_COLUMN, ALL_FEATURES
)


class HyundaiCardPreprocessor:
    """
    현대카드 데이터 전처리기

    TabDiff 모델 입력을 위해 혼합형 데이터(수치+범주)를
    연속 벡터로 변환합니다.

    인코딩 방식:
    - 수치형: StandardScaler → [-2, 2] 범위의 연속값
    - 범주형: One-hot encoding (softmax target으로 사용)

    최종 인코딩 벡터 구조:
    [수치형 피처(정규화)] + [범주형 피처 one-hot 벡터들]
    """

    def __init__(self):
        self.num_scaler = StandardScaler()
        self.cat_encoders: Dict[str, List[str]] = {}   # 피처명 → 값 목록
        self.feature_dims: Dict[str, int] = {}          # 피처명 → 인코딩 차원
        self.num_dim: int = 0
        self.cat_dim: int = 0
        self.total_dim: int = 0
        self.is_fitted = False

        # 범주형 피처별 원핫 슬라이스 인덱스
        self.cat_slices: Dict[str, Tuple[int, int]] = {}

    def fit(self, df: pd.DataFrame) -> "HyundaiCardPreprocessor":
        """학습 데이터로 전처리기 피팅"""
        # 수치형 피처 스케일러 학습
        self.num_scaler.fit(df[NUMERICAL_FEATURES].values.astype(float))
        self.num_dim = len(NUMERICAL_FEATURES)

        # 범주형 피처 인코더 설정
        offset = self.num_dim
        for feat in CATEGORICAL_FEATURES:
            values = CATEGORICAL_VALUES[feat]
            self.cat_encoders[feat] = values
            self.feature_dims[feat] = len(values)
            self.cat_slices[feat] = (offset, offset + len(values))
            offset += len(values)

        self.cat_dim = offset - self.num_dim
        self.total_dim = self.num_dim + self.cat_dim
        self.is_fitted = True

        # Plausibility 평가를 위한 피처별 분포 범위 저장 (5th/95th percentile)
        num_array = df[NUMERICAL_FEATURES].values.astype(float)
        self.num_pct05 = np.percentile(num_array, 5, axis=0)   # (num_dim,)
        self.num_pct95 = np.percentile(num_array, 95, axis=0)  # (num_dim,)

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """DataFrame → 연속 벡터 (n_samples, total_dim)"""
        assert self.is_fitted, "fit() 먼저 호출 필요"

        # 수치형 변환
        num_array = self.num_scaler.transform(df[NUMERICAL_FEATURES].values.astype(float))

        # 범주형 원핫 변환
        cat_parts = []
        for feat in CATEGORICAL_FEATURES:
            values = self.cat_encoders[feat]
            onehot = np.zeros((len(df), len(values)), dtype=np.float32)
            for i, val in enumerate(df[feat].values):
                if val in values:
                    onehot[i, values.index(val)] = 1.0
                else:
                    onehot[i, 0] = 1.0  # 알 수 없는 값 → 첫 번째 카테고리
            cat_parts.append(onehot)

        cat_array = np.concatenate(cat_parts, axis=1)
        return np.concatenate([num_array, cat_array], axis=1).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """연속 벡터 → DataFrame (원래 스케일로 복원)"""
        assert self.is_fitted

        # 수치형 역변환
        num_raw = self.num_scaler.inverse_transform(X[:, :self.num_dim])

        # 범주형 역변환 (argmax로 카테고리 선택)
        cat_rows = {feat: [] for feat in CATEGORICAL_FEATURES}
        for feat in CATEGORICAL_FEATURES:
            start, end = self.cat_slices[feat]
            logits = X[:, start:end]
            indices = np.argmax(logits, axis=1)
            values = self.cat_encoders[feat]
            cat_rows[feat] = [values[idx] for idx in indices]

        # 수치형 피처별 클리핑 규칙 (피처명 → (반올림 소수점, 최소, 최대, int 변환))
        clip_rules = {
            # ── 무실적 위험 피처 ──
            "months_since_last_txn":    (1,   0,    12,    False),
            "avg_spending_3m":          (0,   0,  5000,    False),
            "avg_spending_prev3m":      (0,   0,  5000,    False),
            "spending_trend_ratio":     (3,   0,     3,    False),
            "monthly_txn_count_3m":     (0,   0,   200,    True),
            "num_missed_months":        (0,   0,     6,    True),
            "days_since_app_login":     (0,   0,   365,    True),
            "loyalty_points_balance":   (0,   0, 300000,   False),
            "num_active_benefits":      (0,   0,    20,    True),
            "years_as_customer":        (1,   0,    30,    False),
            # ── 추가 발급 피처 (이전 버전 호환) ──
            "age":                      (0,  18,    80,    True),
            "annual_income":            (0, 500, 30000,    False),
            "credit_score":             (0, 300,   900,    False),
            "num_existing_cards":       (0,   0,    10,    True),
            "monthly_spending":         (0,   0,  5000,    False),
            "total_loan_amount":        (0,   0, 100000,   False),
            "monthly_transactions":     (0,   0,   200,    True),
            "num_delinquencies":        (0,   0,    10,    True),
            "utilization_rate":         (1,   0,   200,    False),
        }
        num_df = pd.DataFrame(num_raw, columns=NUMERICAL_FEATURES)
        for col in NUMERICAL_FEATURES:
            if col in clip_rules:
                dec, lo, hi, to_int = clip_rules[col]
                num_df[col] = num_df[col].round(dec).clip(lo, hi)
                if to_int:
                    num_df[col] = num_df[col].astype(int)

        cat_df = pd.DataFrame(cat_rows)
        return pd.concat([num_df, cat_df], axis=1)

    def get_numerical_slice(self) -> Tuple[int, int]:
        """수치형 피처의 인코딩 벡터 슬라이스"""
        return (0, self.num_dim)

    def get_categorical_slice(self, feat: str) -> Tuple[int, int]:
        """특정 범주형 피처의 인코딩 벡터 슬라이스"""
        return self.cat_slices[feat]

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "HyundaiCardPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """피처와 타겟 분리"""
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]
    return X, y


def get_feature_stats(df: pd.DataFrame) -> Dict:
    """피처별 통계 정보 반환 (CF 품질 평가용)"""
    stats = {}
    for feat in NUMERICAL_FEATURES:
        stats[feat] = {
            "min": df[feat].min(),
            "max": df[feat].max(),
            "mean": df[feat].mean(),
            "std": df[feat].std(),
        }
    return stats
