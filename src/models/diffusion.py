import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    steps = timesteps + 1

    t = torch.linspace(
        0,
        timesteps,
        steps,
        dtype=torch.float64
    ) / timesteps

    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()

    alphas_cumprod = (
        -((t * (end - start) + start) / tau).sigmoid() + v_end
    ) / (v_end - v_start)

    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1.0 - (
        alphas_cumprod[1:] / alphas_cumprod[:-1]
    )

    return torch.clamp(betas, 0, 0.999)


def extract(a, t, x_shape):
    out = a.gather(0, t)

    return out.reshape(
        t.shape[0],
        *((1,) * (len(x_shape) - 1))
    )


class GaussianDiffusion(nn.Module):

    def __init__(
        self,
        model,
        image_size,
        timesteps=1000,
        objective="pred_v",
        beta_schedule="sigmoid",
        auto_normalize=False,
        min_snr_loss_weight=False,
        min_snr_gamma=5
    ):
        super().__init__()

        assert model.channels == model.out_dim
        assert not model.random_or_learned_sinusoidal_cond

        assert objective in {
            "pred_noise",
            "pred_x0",
            "pred_v"
        }

        self.model = model
        self.channels = model.channels
        self.self_condition = model.self_condition

        self.image_size = image_size
        self.objective = objective

        if beta_schedule == "sigmoid":
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(
                f"Unknown beta schedule: {beta_schedule}"
            )

        alphas = 1.0 - betas

        alphas_cumprod = torch.cumprod(
            alphas,
            dim=0
        )

        alphas_cumprod_prev = F.pad(
            alphas_cumprod[:-1],
            (1, 0),
            value=1.0
        )

        self.num_timesteps = int(timesteps)

        def register_buffer(name, value):
            self.register_buffer(name, value.float())

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        register_buffer(
            "sqrt_alphas_cumprod",
            torch.sqrt(alphas_cumprod)
        )

        register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod)
        )

        register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod)
        )

        register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )

        register_buffer(
            "posterior_variance",
            posterior_variance
        )

        register_buffer(
            "posterior_log_variance_clipped",
            torch.log(
                posterior_variance.clamp(min=1e-20)
            )
        )

        register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )

        register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )

        snr = alphas_cumprod / (1.0 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()

        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(
                max=min_snr_gamma
            )

        if objective == "pred_noise":
            loss_weight = maybe_clipped_snr / snr

        elif objective == "pred_x0":
            loss_weight = maybe_clipped_snr

        else:
            loss_weight = (
                maybe_clipped_snr / (snr + 1.0)
            )

        register_buffer(
            "loss_weight",
            loss_weight
        )

        if auto_normalize:
            self.normalize = lambda x: x * 2.0 - 1.0
            self.unnormalize = lambda x: (x + 1.0) * 0.5
        else:
            self.normalize = lambda x: x
            self.unnormalize = lambda x: x

    @property
    def device(self):
        return self.betas.device

    # --------------------------------------------------
    # Forward diffusion: add noise
    # --------------------------------------------------

    def q_sample(self, x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        return (
            extract(
                self.sqrt_alphas_cumprod,
                t,
                x_start.shape
            ) * x_start

            + extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                x_start.shape
            ) * noise
        )

    # --------------------------------------------------
    # Velocity target
    # --------------------------------------------------

    def predict_v(self, x_start, t, noise):

        return (
            extract(
                self.sqrt_alphas_cumprod,
                t,
                x_start.shape
            ) * noise

            - extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                x_start.shape
            ) * x_start
        )

    # --------------------------------------------------
    # Reconstruct x0 from predicted velocity
    # --------------------------------------------------

    def predict_start_from_v(self, x_t, t, v):

        return (
            extract(
                self.sqrt_alphas_cumprod,
                t,
                x_t.shape
            ) * x_t

            - extract(
                self.sqrt_one_minus_alphas_cumprod,
                t,
                x_t.shape
            ) * v
        )

    # --------------------------------------------------
    # Reverse prediction: reconstruct x0 from noise
    # --------------------------------------------------

    def predict_start_from_noise(self, x_t, t, noise):

        return (
            extract(
                self.sqrt_recip_alphas_cumprod,
                t,
                x_t.shape
            ) * x_t

            - extract(
                self.sqrt_recipm1_alphas_cumprod,
                t,
                x_t.shape
            ) * noise
        )

    # --------------------------------------------------
    # Training loss
    # --------------------------------------------------

    def p_losses(self, x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        x = self.q_sample(
            x_start,
            t,
            noise
        )

        if self.model.training:
            model_out = self.model(x, t, None)
        else:
            model_out, feat_out = self.model(x, t, None)

        if self.objective == "pred_noise":
            target = noise

        elif self.objective == "pred_x0":
            target = x_start

        else:
            target = self.predict_v(
                x_start,
                t,
                noise
            )

        loss = F.mse_loss(
            model_out,
            target,
            reduction="none"
        )

        loss = loss.reshape(
            loss.shape[0],
            -1
        ).mean(dim=1)

        loss = loss * extract(
            self.loss_weight,
            t,
            loss.shape
        )

        if self.model.training:
            return loss.mean()

        return x, model_out, feat_out

    # --------------------------------------------------
    # Main forward
    # --------------------------------------------------

    def forward(self, img):

        batch, channels, height, width = img.shape

        assert height == self.image_size
        assert width == self.image_size
        assert channels == self.channels

        t = torch.randint(
            0,
            self.num_timesteps,
            (batch,),
            device=img.device
        ).long()

        img = self.normalize(img)

        return self.p_losses(img, t)