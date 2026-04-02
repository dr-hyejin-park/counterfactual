"""
추가 카드 발급 예측 분류기
- 신경망 기반 (gradient 계산 가능 → classifier guidance에 활용)
- 학습 데이터: 전처리된 현대카드 고객 데이터
- 출력: P(추가 발급 = 1 | 고객 특성)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
import os


class CardIssuanceClassifier(nn.Module):
    """
    추가 카드 발급 예측 신경망 분류기

    TabDiff counterfactual guidance에서 사용:
    guidance_fn(x_t) = log σ(classifier(x_t))
    → 분류기 gradient가 역확산 방향을 클래스 1로 유도
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))  # binary output

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """logit 반환 (sigmoid 적용 전)"""
        return self.network(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """추가 발급 확률 P(y=1|x) 반환"""
        return torch.sigmoid(self.forward(x))

    def log_prob_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        log P(y=1|x) 반환 — classifier guidance에서 사용
        수식: log σ(f(x))
        """
        return F.logsigmoid(self.forward(x))


class ClassifierTrainer:
    """분류기 학습 관리자"""

    def __init__(
        self,
        model: CardIssuanceClassifier,
        lr: float = 1e-3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=80
        )
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(X_batch)
        self.scheduler.step()
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).float()
            logits = self.model(X_batch)
            loss = F.binary_cross_entropy_with_logits(logits, y_batch)
            total_loss += loss.item() * len(X_batch)
            preds = (logits > 0).float()
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
        return total_loss / total, correct / total

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int = 80,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> None:
        # 입력 차원 사전 검증 — 불일치 시 명확한 오류 메시지 제공
        actual_dim = X_train.shape[1]
        expected_dim = self.model.input_dim
        if actual_dim != expected_dim:
            raise ValueError(
                f"입력 차원 불일치: 학습 데이터는 {actual_dim}차원이지만 "
                f"모델은 {expected_dim}차원을 기대합니다.\n"
                f"  원인: config.py의 피처 정의와 맞지 않는 기존 체크포인트가 "
                f"checkpoints/ 폴더에 남아 있을 수 있습니다.\n"
                f"  해결: checkpoints/ 폴더를 삭제하고 train.py를 다시 실행하세요."
            )
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        best_val_acc = 0.0
        best_state = None

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if verbose and epoch % 20 == 0:
                print(f"  Epoch {epoch:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc*100:.1f}%")

        # 최적 가중치 복원
        if best_state is not None:
            self.model.load_state_dict(best_state)
        if verbose:
            print(f"  최적 Val Accuracy: {best_val_acc*100:.1f}%")

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.model.input_dim,
            "hidden_dims": self.model.hidden_dims,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs,
        }, path)

    @staticmethod
    def load(path: str, device: str = "cpu") -> "CardIssuanceClassifier":
        ckpt = torch.load(path, map_location=device)
        hidden_dims = tuple(ckpt.get("hidden_dims", (128, 64, 32)))
        model = CardIssuanceClassifier(
            input_dim=ckpt["input_dim"],
            hidden_dims=hidden_dims,
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        return model
