"""
TabDiff 모델 및 분류기 학습 스크립트

실행:
  python train.py [--epochs_tabdiff N] [--epochs_clf N] [--quick]

빠른 테스트: python train.py --quick
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_CONFIG, TABDIFF_CONFIG, CLASSIFIER_CONFIG,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
)
from data.hyundai_card_data import generate_hyundai_card_data, print_data_summary
from utils.preprocessing import HyundaiCardPreprocessor, split_features_target
from models.tabdiff import TabDiff
from models.classifier import CardIssuanceClassifier, ClassifierTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="TabDiff CF 모델 학습")
    parser.add_argument("--epochs_tabdiff", type=int, default=TABDIFF_CONFIG["num_epochs"])
    parser.add_argument("--epochs_clf", type=int, default=CLASSIFIER_CONFIG["num_epochs"])
    parser.add_argument("--quick", action="store_true", help="빠른 테스트 (에폭 최소화)")
    parser.add_argument("--n_samples", type=int, default=DATA_CONFIG["n_samples"])
    parser.add_argument("--seed", type=int, default=DATA_CONFIG["random_seed"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    return parser.parse_args()


def train_tabdiff(
    model: TabDiff,
    X_train: np.ndarray,
    X_val: np.ndarray,
    num_epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> TabDiff:
    """TabDiff DDPM 학습"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float("inf")
    best_state = None

    print(f"\n[TabDiff 학습] {num_epochs} 에폭, 배치 {batch_size}, LR {lr}")
    print(f"  입력 차원: {model.input_dim}, 학습 샘플: {len(X_train)}")

    for epoch in range(1, num_epochs + 1):
        # ── 학습 ────────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            loss = model.compute_loss(x_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # ── 검증 ────────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (x_batch,) in val_loader:
                x_batch = x_batch.to(device)
                loss = model.compute_loss(x_batch)
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % max(1, num_epochs // 10) == 0:
            print(f"  Epoch {epoch:4d}/{num_epochs} | "
                  f"Train: {np.mean(train_losses):.4f} | "
                  f"Val: {val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"  최적 Val Loss: {best_val_loss:.4f}")
    return model


def main():
    args = parse_args()
    device = args.device

    if args.quick:
        args.epochs_tabdiff = 10
        args.epochs_clf = 15
        args.n_samples = 2000
        print("[빠른 모드] 에폭 최소화, 샘플 2000개")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. 데이터 생성 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1단계: 현대카드 고객 데이터 생성")
    print("=" * 60)
    df = generate_hyundai_card_data(n_samples=args.n_samples, random_seed=args.seed)
    print_data_summary(df)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/hyundai_card_customers.csv", index=False)
    print("데이터 저장: data/hyundai_card_customers.csv")

    # ── 2. 전처리 ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("2단계: 데이터 전처리")
    print("=" * 60)

    X_df, y = split_features_target(df)
    y_np = y.values

    # 학습/검증/테스트 분할 (70/10/20)
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df, y_np, test_size=0.2, random_state=args.seed, stratify=y_np
    )
    X_train_df, X_val_df, y_train, y_val = train_test_split(
        X_train_df, y_train, test_size=0.125, random_state=args.seed, stratify=y_train
    )  # 0.125 * 0.8 = 0.1

    # 전처리기 학습
    preprocessor = HyundaiCardPreprocessor()
    X_train_enc = preprocessor.fit_transform(X_train_df)
    X_val_enc = preprocessor.transform(X_val_df)
    X_test_enc = preprocessor.transform(X_test_df)

    print(f"  학습: {len(X_train_enc)}  검증: {len(X_val_enc)}  테스트: {len(X_test_enc)}")
    print(f"  인코딩 차원: {preprocessor.total_dim}")
    print(f"  수치형: {preprocessor.num_dim}, 범주형(원핫): {preprocessor.cat_dim}")

    preprocessor.save(f"{args.output_dir}/preprocessor.pkl")
    print(f"  전처리기 저장: {args.output_dir}/preprocessor.pkl")

    # 테스트 데이터도 저장
    test_df = X_test_df.copy()
    test_df[TARGET_COLUMN] = y_test
    test_df.insert(0, "customer_id", [
        df.loc[X_test_df.index[i], "customer_id"] for i in range(len(X_test_df))
    ])
    test_df.to_csv(f"{args.output_dir}/test_customers.csv", index=False)

    # ── 3. 분류기 학습 ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("3단계: 추가 발급 예측 분류기 학습")
    print("=" * 60)

    clf_model = CardIssuanceClassifier(
        input_dim=preprocessor.total_dim,
        hidden_dims=tuple(CLASSIFIER_CONFIG["hidden_dims"]),
        dropout=CLASSIFIER_CONFIG["dropout"],
    )
    clf_trainer = ClassifierTrainer(
        model=clf_model,
        lr=CLASSIFIER_CONFIG["learning_rate"],
        device=device,
    )
    clf_trainer.train(
        X_train=X_train_enc,
        y_train=y_train,
        X_val=X_val_enc,
        y_val=y_val,
        num_epochs=args.epochs_clf,
        batch_size=CLASSIFIER_CONFIG["batch_size"],
        verbose=True,
    )

    # 테스트 평가
    X_test_tensor = torch.tensor(X_test_enc, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_probs = clf_model.predict_proba(X_test_tensor).cpu().numpy()
    test_preds = (test_probs > 0.5).astype(int)
    test_acc = (test_preds == y_test).mean()
    print(f"\n  테스트 Accuracy: {test_acc*100:.1f}%")

    clf_trainer.save(f"{args.output_dir}/classifier.pt")
    print(f"  분류기 저장: {args.output_dir}/classifier.pt")

    # ── 4. TabDiff 학습 ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("4단계: TabDiff DDPM 모델 학습")
    print("=" * 60)

    tabdiff = TabDiff(
        input_dim=preprocessor.total_dim,
        num_timesteps=TABDIFF_CONFIG["num_timesteps"],
        beta_start=TABDIFF_CONFIG["beta_start"],
        beta_end=TABDIFF_CONFIG["beta_end"],
        hidden_dim=TABDIFF_CONFIG["hidden_dim"],
        num_heads=TABDIFF_CONFIG["num_heads"],
        num_layers=TABDIFF_CONFIG["num_layers"],
        dropout=TABDIFF_CONFIG["dropout"],
        device=device,
    ).to(device)

    print(f"  파라미터 수: {sum(p.numel() for p in tabdiff.parameters()):,}")

    tabdiff = train_tabdiff(
        model=tabdiff,
        X_train=X_train_enc,
        X_val=X_val_enc,
        num_epochs=args.epochs_tabdiff,
        batch_size=TABDIFF_CONFIG["batch_size"],
        lr=TABDIFF_CONFIG["learning_rate"],
        device=device,
    )

    torch.save({
        "model_state": tabdiff.state_dict(),
        "input_dim": preprocessor.total_dim,
        "config": TABDIFF_CONFIG,
    }, f"{args.output_dir}/tabdiff.pt")
    print(f"  TabDiff 저장: {args.output_dir}/tabdiff.pt")

    print("\n" + "=" * 60)
    print("학습 완료!")
    print(f"  체크포인트 디렉토리: {args.output_dir}/")
    print("  다음 단계: python run_demo.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
