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

        # 원래 스케일로 클리핑
        num_df = pd.DataFrame(num_raw, columns=NUMERICAL_FEATURES)
        num_df["age"] = num_df["age"].round(0).clip(18, 80).astype(int)
        num_df["annual_income"] = num_df["annual_income"].round(0).clip(500, 30000)
        num_df["credit_score"] = num_df["credit_score"].round(0).clip(300, 900)
        num_df["num_existing_cards"] = num_df["num_existing_cards"].round(0).clip(0, 10).astype(int)
        num_df["monthly_spending"] = num_df["monthly_spending"].round(0).clip(0, 5000)
        num_df["years_as_customer"] = num_df["years_as_customer"].round(1).clip(0, 30)
        num_df["total_loan_amount"] = num_df["total_loan_amount"].round(0).clip(0, 100000)
        num_df["monthly_transactions"] = num_df["monthly_transactions"].round(0).clip(0, 200).astype(int)
        num_df["num_delinquencies"] = num_df["num_delinquencies"].round(0).clip(0, 10).astype(int)
        num_df["utilization_rate"] = num_df["utilization_rate"].round(1).clip(0, 200)

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
