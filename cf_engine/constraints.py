"""
행동 가능성 제약 (Actionability Constraints)
논문: "actionable counterfactual" — 현실적으로 실행 가능한 변화만 허용

제약 유형:
1. 불변 (Immutable): 변경 불가 (나이, 지역, 학력 등 인구통계)
2. 단조 증가 (Increasing only): 값이 올라가는 방향만 허용 (신용점수 개선 등)
3. 단조 감소 (Decreasing only): 값이 내려가는 방향만 허용 (연체 횟수 감소 등)
4. 범위 제한 (Bounded): 최대 변화 비율 제한
"""

import numpy as np
import torch
from typing import Dict, List, Optional

from config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, CATEGORICAL_VALUES,
    IMMUTABLE_FEATURES, INCREASING_ONLY_FEATURES, DECREASING_ONLY_FEATURES,
    CF_CONFIG,
)


class ActionabilityConstraints:
    """
    반사실적 생성 중 각 역확산 스텝마다 적용되는 제약 조건

    인코딩된 벡터 공간에서 동작하며 전처리기(preprocessor)와 연동됩니다.
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.num_dim = preprocessor.num_dim
        self.total_dim = preprocessor.total_dim

        # 수치형 피처 인덱스
        self.num_feature_indices: Dict[str, int] = {
            feat: i for i, feat in enumerate(NUMERICAL_FEATURES)
        }

        # 불변 피처 → 수치형 인덱스 (인코딩 벡터에서의 위치)
        self.immutable_num_idx: List[int] = [
            self.num_feature_indices[f]
            for f in IMMUTABLE_FEATURES
            if f in self.num_feature_indices
        ]

        # 불변 피처 → 범주형 슬라이스
        self.immutable_cat_slices: List[tuple] = [
            preprocessor.get_categorical_slice(f)
            for f in IMMUTABLE_FEATURES
            if f in CATEGORICAL_FEATURES
        ]

        # 단조 증가 피처 인덱스
        self.increasing_idx: List[int] = [
            self.num_feature_indices[f]
            for f in INCREASING_ONLY_FEATURES
            if f in self.num_feature_indices
        ]

        # 단조 감소 피처 인덱스
        self.decreasing_idx: List[int] = [
            self.num_feature_indices[f]
            for f in DECREASING_ONLY_FEATURES
            if f in self.num_feature_indices
        ]

    def apply(
        self, x_cf: torch.Tensor, x_factual: torch.Tensor
    ) -> torch.Tensor:
        """
        반사실적 벡터에 행동 가능성 제약 적용

        Args:
            x_cf:      생성 중인 반사실적 벡터 (batch, total_dim)
            x_factual: 원본 고객 벡터 (batch, total_dim)
        Returns:
            x_cf: 제약이 적용된 반사실적 벡터
        """
        x_cf = x_cf.clone()

        # 1. 수치형 불변 피처 고정
        for idx in self.immutable_num_idx:
            x_cf[:, idx] = x_factual[:, idx]

        # 2. 범주형 불변 피처 고정 (원핫 슬라이스 통째로 복원)
        for start, end in self.immutable_cat_slices:
            x_cf[:, start:end] = x_factual[:, start:end]

        # 3. 단조 증가 제약: CF 값이 원본보다 작으면 원본으로 클리핑
        for idx in self.increasing_idx:
            x_cf[:, idx] = torch.maximum(x_cf[:, idx], x_factual[:, idx])

        # 4. 단조 감소 제약: CF 값이 원본보다 크면 원본으로 클리핑
        for idx in self.decreasing_idx:
            x_cf[:, idx] = torch.minimum(x_cf[:, idx], x_factual[:, idx])

        return x_cf

    def __call__(self, x_cf: torch.Tensor, x_factual: torch.Tensor) -> torch.Tensor:
        return self.apply(x_cf, x_factual)


def compute_proximity(
    x_cf: np.ndarray,
    x_factual: np.ndarray,
    preprocessor,
) -> float:
    """
    반사실적 품질 지표: 원본과의 근접도 (L2 거리, 낮을수록 좋음)
    수치형 피처에 대해서만 계산 (정규화된 공간)
    """
    num_dim = preprocessor.num_dim
    diff = x_cf[:, :num_dim] - x_factual[:, :num_dim]
    return float(np.linalg.norm(diff, axis=1).mean())


def compute_sparsity(
    x_cf: np.ndarray,
    x_factual: np.ndarray,
    preprocessor,
    threshold: float = 0.05,
) -> float:
    """
    반사실적 품질 지표: 희소성 (변경된 피처 수가 적을수록 좋음)
    threshold 이상 변화한 수치형 피처 비율
    """
    num_dim = preprocessor.num_dim
    changed = np.abs(x_cf[:, :num_dim] - x_factual[:, :num_dim]) > threshold
    return float(changed.mean())


def compute_validity(
    x_cf_decoded: "pd.DataFrame",
    classifier,
    preprocessor,
    device: str = "cpu",
) -> float:
    """
    반사실적 품질 지표: 유효성 (목표 클래스로 예측되는 비율)
    """
    import torch
    X_cf_enc = preprocessor.transform(x_cf_decoded)
    X_tensor = torch.tensor(X_cf_enc, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = classifier.predict_proba(X_tensor).cpu().numpy()
    return float((probs > 0.5).mean())
