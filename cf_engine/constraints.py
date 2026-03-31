"""
행동 가능성 제약 (Actionability Constraints) — Wake-up 캠페인 버전

무실적 위험 고객 활성화 맥락의 제약:
1. 불변 (Immutable):   과거 실적·카드 등급 등 마케팅으로 변경 불가
2. 단조 감소 (↓ only): 마지막 거래 경과 월·앱 미사용 일수 등 줄여야 활성화
3. 단조 증가 (↑ only): 사용금액·거래 건수·혜택 수 등 늘려야 활성화
"""

import numpy as np
import torch
from typing import Dict, List, Tuple

from config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, CATEGORICAL_VALUES,
    IMMUTABLE_FEATURES, INCREASING_ONLY_FEATURES, DECREASING_ONLY_FEATURES,
    CF_CONFIG,
)


class ActionabilityConstraints:
    """
    반사실적 생성 중 각 역확산 스텝마다 적용되는 행동 가능성 제약
    인코딩된 벡터 공간에서 동작하며 preprocessor와 연동됩니다.
    """

    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.num_dim = preprocessor.num_dim
        self.total_dim = preprocessor.total_dim

        # 수치형 피처 → 인덱스 매핑
        self.num_feature_indices: Dict[str, int] = {
            feat: i for i, feat in enumerate(NUMERICAL_FEATURES)
        }

        # 불변 피처 → 수치형 인덱스
        self.immutable_num_idx: List[int] = [
            self.num_feature_indices[f]
            for f in IMMUTABLE_FEATURES
            if f in self.num_feature_indices
        ]

        # 불변 피처 → 범주형 원핫 슬라이스
        self.immutable_cat_slices: List[Tuple[int, int]] = [
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

    def apply(self, x_cf: torch.Tensor, x_factual: torch.Tensor) -> torch.Tensor:
        """
        반사실적 벡터에 행동 가능성 제약 적용

        Args:
            x_cf:      생성 중인 반사실적 벡터 (batch, total_dim)
            x_factual: 원본 고객 벡터         (batch, total_dim)
        Returns:
            x_cf: 제약이 적용된 반사실적 벡터
        """
        x_cf = x_cf.clone()

        # 1. 수치형 불변 피처 고정
        for idx in self.immutable_num_idx:
            x_cf[:, idx] = x_factual[:, idx]

        # 2. 범주형 불변 피처 고정
        for start, end in self.immutable_cat_slices:
            x_cf[:, start:end] = x_factual[:, start:end]

        # 3. 단조 증가 제약 (CF가 원본보다 낮아지면 원본으로 클리핑)
        for idx in self.increasing_idx:
            x_cf[:, idx] = torch.maximum(x_cf[:, idx], x_factual[:, idx])

        # 4. 단조 감소 제약 (CF가 원본보다 높아지면 원본으로 클리핑)
        for idx in self.decreasing_idx:
            x_cf[:, idx] = torch.minimum(x_cf[:, idx], x_factual[:, idx])

        return x_cf

    def __call__(self, x_cf: torch.Tensor, x_factual: torch.Tensor) -> torch.Tensor:
        return self.apply(x_cf, x_factual)


def compute_proximity(x_cf: np.ndarray, x_factual: np.ndarray, preprocessor) -> float:
    """CF 품질: 원본과의 L2 거리 (정규화 공간, 낮을수록 좋음)"""
    num_dim = preprocessor.num_dim
    diff = x_cf[:, :num_dim] - x_factual[:, :num_dim]
    return float(np.linalg.norm(diff, axis=1).mean())


def compute_sparsity(
    x_cf: np.ndarray, x_factual: np.ndarray, preprocessor, threshold: float = 0.05
) -> float:
    """CF 품질: 변경된 피처 비율 (낮을수록 좋음)"""
    num_dim = preprocessor.num_dim
    changed = np.abs(x_cf[:, :num_dim] - x_factual[:, :num_dim]) > threshold
    return float(changed.mean())


def compute_validity(x_cf_decoded, classifier, preprocessor, device: str = "cpu") -> float:
    """CF 품질: 목표 클래스로 예측되는 비율"""
    import torch
    from config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES, CF_CONFIG
    X_cf_enc = preprocessor.transform(x_cf_decoded[NUMERICAL_FEATURES + CATEGORICAL_FEATURES])
    X_tensor = torch.tensor(X_cf_enc, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = classifier.predict_proba(X_tensor).cpu().numpy()
    target = CF_CONFIG["target_class"]
    # target=0: 활성 고객 → prob < 0.5 가 목표
    if target == 0:
        return float((probs < 0.5).mean())
    else:
        return float((probs > 0.5).mean())
