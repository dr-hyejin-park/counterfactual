"""
TabDiff: 표 형 데이터용 Denoising Diffusion Probabilistic Model
논문: "Tabular Diffusion Based Actionable Counterfactual Explanations
       for Network Intrusion Detection" 기반 구현

아키텍처:
- 표준 DDPM (Ho et al. 2020) + Transformer 기반 노이즈 예측 네트워크
- 혼합형 데이터(수치+원핫 인코딩된 범주형)를 연속 공간에서 확산
- sinusoidal timestep embedding 사용
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ─── Positional / Timestep Embedding ────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    """DDPM 스타일 sinusoidal timestep 임베딩"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ─── Transformer 기반 노이즈 예측 네트워크 ────────────────────────────────────

class TabDiffTransformer(nn.Module):
    """
    표형 데이터의 노이즈를 예측하는 Transformer 네트워크
    ε_θ(x_t, t) → 예측 노이즈

    입력: [노이즈 추가된 데이터 x_t, timestep t]
    출력: 예측 노이즈 ε (x_t와 같은 차원)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # timestep 임베딩
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # 입력 프로젝션 (tabular features → hidden_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Transformer 인코더 레이어들
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm (더 안정적인 학습)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 출력 프로젝션
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) — 노이즈 추가된 데이터
            t: (batch,) — timestep (정수)
        Returns:
            eps_pred: (batch, input_dim) — 예측 노이즈
        """
        # timestep 임베딩
        t_emb = self.time_embed(t)  # (batch, hidden_dim)

        # 입력 프로젝션
        h = self.input_proj(x)      # (batch, hidden_dim)

        # timestep 정보 주입 (additive)
        h = h + t_emb

        # Transformer는 sequence 입력 필요 → (batch, 1, hidden_dim)으로 처리
        # 피처를 토큰 시퀀스로 취급하기 위해 (batch, 1, hidden_dim) 사용
        h = h.unsqueeze(1)          # (batch, 1, hidden_dim)
        h = self.transformer(h)     # (batch, 1, hidden_dim)
        h = h.squeeze(1)            # (batch, hidden_dim)

        return self.output_proj(h)  # (batch, input_dim)


# ─── DDPM 확산 프로세스 ──────────────────────────────────────────────────────

class TabDiff(nn.Module):
    """
    표형 데이터용 DDPM (Denoising Diffusion Probabilistic Model)

    [Forward process] q(x_t | x_0):
        x_t = sqrt(α̅_t) * x_0 + sqrt(1 - α̅_t) * ε,  ε ~ N(0, I)

    [Reverse process] p_θ(x_{t-1} | x_t):
        예측 노이즈 ε_θ(x_t, t)로 x_0 추정 후 샘플링

    [학습 목표]:
        L_simple = E[||ε - ε_θ(x_t, t)||²]
    """

    def __init__(
        self,
        input_dim: int,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        hidden_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_timesteps = num_timesteps
        self.device_str = device

        # 노이즈 스케줄 (linear schedule)
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # 버퍼로 등록 (학습 파라미터 아님)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        # 역확산 posterior variance: β̃_t = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(posterior_variance.clamp(min=1e-20)))

        # 노이즈 예측 네트워크
        self.denoiser = TabDiffTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    # ── Forward (Noising) ────────────────────────────────────────────────────

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: x_0에 t 스텝만큼 노이즈 추가
        q(x_t | x_0) = N(sqrt(α̅_t)*x_0, (1-α̅_t)*I)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_a = self.sqrt_alphas_cumprod[t][:, None]       # (batch, 1)
        sqrt_1a = self.sqrt_one_minus_alphas_cumprod[t][:, None]  # (batch, 1)
        return sqrt_a * x_0 + sqrt_1a * noise

    # ── 학습 손실 계산 ────────────────────────────────────────────────────────

    def compute_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Simple DDPM loss: MSE(ε_true, ε_θ(x_t, t))"""
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        eps_pred = self.denoiser(x_t, t)
        return F.mse_loss(eps_pred, noise)

    # ── Reverse (Denoising) 단일 스텝 ────────────────────────────────────────

    @torch.no_grad()
    def p_sample(
        self, x_t: torch.Tensor, t: int, guidance_fn=None, guidance_scale: float = 0.0
    ) -> torch.Tensor:
        """
        역확산 한 스텝: p_θ(x_{t-1} | x_t)
        classifier-free guidance 또는 classifier guidance 지원
        """
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)

        # 노이즈 예측
        eps_pred = self.denoiser(x_t, t_tensor)

        # x_0 추정 (DDPM 공식)
        sqrt_recip_a = self.sqrt_alphas_cumprod[t] ** (-1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t]
        x_0_pred = sqrt_recip_a * (x_t - sqrt_1ma * eps_pred)

        # posterior mean
        beta_t = self.betas[t]
        sqrt_a_prev = self.alphas_cumprod_prev[t] ** 0.5
        sqrt_1ma_prev = (1.0 - self.alphas_cumprod_prev[t]) ** 0.5
        sqrt_1ma_t = (1.0 - self.alphas_cumprod[t]) ** 0.5

        coef1 = beta_t * sqrt_a_prev / (1.0 - self.alphas_cumprod[t])
        coef2 = (1.0 - self.alphas_cumprod_prev[t]) * self.alphas[t] ** 0.5 / (1.0 - self.alphas_cumprod[t])
        mean = coef1 * x_0_pred + coef2 * x_t

        # 노이즈 추가 (t=0이면 노이즈 없음)
        if t > 0:
            var = self.posterior_variance[t]
            noise = torch.randn_like(x_t)
            x_prev = mean + var ** 0.5 * noise
        else:
            x_prev = mean

        return x_prev

    def p_sample_with_guidance(
        self,
        x_t: torch.Tensor,
        t: int,
        guidance_fn,
        guidance_scale: float,
    ) -> torch.Tensor:
        """
        x₀-prediction classifier guidance를 포함한 역확산 스텝

        [기존 방식의 문제]
        분류기는 깨끗한 x₀로 학습됐지만, 기존 방식은 노이즈가 낀 x_t에서
        gradient를 계산 → 분류기가 out-of-distribution 입력을 받아 신호 매우 약함

        [개선: x₀-prediction guidance]
        1. ε_θ(x_t, t)로 깨끗한 x₀_pred를 먼저 복원
        2. 분류기 훈련 분포(x₀)에서 gradient 계산: ∇_{x₀} log p(y|x₀_pred)
        3. chain rule로 x_t 공간으로 역전파: grad_xt = grad_x0 / √ᾱ_t

        수식: ε̃ = ε_θ(x_t,t) - guidance_scale * √(1-ᾱ_t)/√ᾱ_t * ∇_{x₀} log p(y|x₀_pred)
        """
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        x_t_d = x_t.detach()

        # ── 1단계: denoiser로 eps 예측 (no_grad) ────────────────────────────
        with torch.no_grad():
            eps_pred = self.denoiser(x_t_d, t_tensor)

        sqrt_a    = self.sqrt_alphas_cumprod[t]           # √ᾱ_t
        sqrt_1ma  = self.sqrt_one_minus_alphas_cumprod[t]  # √(1-ᾱ_t)

        # ── 2단계: x₀_pred 복원 (clean space) ───────────────────────────────
        # x₀_pred = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t
        x_0_pred = (x_t_d - sqrt_1ma * eps_pred) / sqrt_a

        # ── 3단계: 분류기 훈련 분포(x₀)에서 gradient 계산 ───────────────────
        x_0_guided = x_0_pred.detach().requires_grad_(True)
        log_prob   = guidance_fn(x_0_guided)              # log p(y=target | x₀_pred)
        grad_x0    = torch.autograd.grad(log_prob.sum(), x_0_guided)[0]

        # ── 4단계: chain rule로 x_t 공간의 eps 보정 ─────────────────────────
        # ∂x₀_pred/∂x_t = 1/√ᾱ_t  →  grad_xt = grad_x0 / √ᾱ_t
        # eps 보정: ε̃ = ε - scale * √(1-ᾱ_t) * (grad_x0 / √ᾱ_t)
        grad_xt   = grad_x0.detach() / sqrt_a
        eps_guided = eps_pred - guidance_scale * sqrt_1ma * grad_xt

        # ── 5단계: eps_guided로 x₀ 재추정 → posterior mean ──────────────────
        x_0_guided_final = (x_t_d - sqrt_1ma * eps_guided) / sqrt_a

        beta_t = self.betas[t]
        coef1  = beta_t * self.alphas_cumprod_prev[t] ** 0.5 / (1.0 - self.alphas_cumprod[t])
        coef2  = (1.0 - self.alphas_cumprod_prev[t]) * self.alphas[t] ** 0.5 / (1.0 - self.alphas_cumprod[t])
        mean   = coef1 * x_0_guided_final + coef2 * x_t_d

        if t > 0:
            noise  = torch.randn_like(x_t_d)
            x_prev = mean + self.posterior_variance[t] ** 0.5 * noise
        else:
            x_prev = mean

        return x_prev.detach()

    # ── 전체 샘플 생성 ────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, batch_size: int, device: str) -> torch.Tensor:
        """표준 DDPM 샘플링 (순수 생성, guidance 없음)"""
        x_t = torch.randn(batch_size, self.input_dim, device=device)
        for t in reversed(range(self.num_timesteps)):
            x_t = self.p_sample(x_t, t)
        return x_t

    def partial_noise(
        self, x_0: torch.Tensor, t_start: int
    ) -> torch.Tensor:
        """
        x_0에 t_start 스텝까지 노이즈 추가 (반사실적 생성 시작점)
        논문: factual instance를 부분 노이즈 처리하여 reverse 시작
        """
        t_tensor = torch.full((x_0.shape[0],), t_start, device=x_0.device, dtype=torch.long)
        return self.q_sample(x_0, t_tensor)

    def generate_counterfactual_trajectory(
        self,
        x_factual: torch.Tensor,
        t_start: int,
        guidance_fn,
        guidance_scale: float,
        constraint_fn=None,
        refine_steps: int = 0,
        refine_lr: float = 0.05,
    ) -> torch.Tensor:
        """
        반사실적 생성 역확산 루프 (x₀-prediction guidance + 후처리 정제)

        Args:
            x_factual:      원본 고객 데이터 (인코딩됨)
            t_start:        역확산 시작 timestep (T_cf)
            guidance_fn:    log p(y=target|x₀) 반환하는 함수
            guidance_scale: guidance 강도 λ
            constraint_fn:  제약 함수 (optional)
            refine_steps:   diffusion 후 gradient ascent 정제 횟수 (0=생략)
            refine_lr:      정제 스텝 학습률
        Returns:
            x_cf: 반사실적 데이터
        """
        x_t = self.partial_noise(x_factual, t_start)

        for t in reversed(range(t_start)):
            x_t = self.p_sample_with_guidance(x_t, t, guidance_fn, guidance_scale)
            if constraint_fn is not None:
                x_t = constraint_fn(x_t, x_factual)

        # ── 후처리 정제: clean space에서 gradient ascent ─────────────────────
        # diffusion 이후에도 분류기 경계를 넘지 못한 경우를 위한 보완
        if refine_steps > 0:
            x_t = self._gradient_refine(x_t, x_factual, guidance_fn,
                                         refine_steps, refine_lr, constraint_fn)

        return x_t

    def _gradient_refine(
        self,
        x_cf: torch.Tensor,
        x_factual: torch.Tensor,
        guidance_fn,
        steps: int,
        lr: float,
        constraint_fn=None,
    ) -> torch.Tensor:
        """
        diffusion 결과를 clean space에서 gradient ascent로 추가 정제.
        분류기 경계 근처에 있는 샘플들을 목표 클래스 쪽으로 밀어주는 역할.
        proximity 패널티를 함께 적용해 원본과 너무 멀어지지 않도록 제어.
        """
        x = x_cf.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=lr)

        for _ in range(steps):
            optimizer.zero_grad()
            log_p       = guidance_fn(x)                          # log p(y=target|x)
            proximity   = ((x - x_factual.detach()) ** 2).mean()  # 원본과의 거리
            loss        = -(log_p.mean()) + 0.1 * proximity       # 목표 방향 + proximity 패널티
            loss.backward()
            optimizer.step()

            # 정제 중에도 제약 유지
            with torch.no_grad():
                if constraint_fn is not None:
                    x.data = constraint_fn(x.data, x_factual.detach())

        return x.detach()
