import torch
from src.losses.pac_loss import PACLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pac = PACLoss(weight=10.0).to(device)

original_rcams = torch.rand(2, 6, 32, 32, device=device)
denoised_rcams = torch.rand(2, 6, 32, 32, device=device)

image_labels = torch.tensor(
    [
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 0]
    ],
    dtype=torch.float32,
    device=device
)

original_rcams.requires_grad_(True)
denoised_rcams.requires_grad_(True)

loss = pac(
    original_rcams,
    denoised_rcams,
    image_labels
)

loss.backward()

print("PAC loss:", loss.item())
print("Original CAM gradients:", original_rcams.grad)
print("Denoised CAM gradients:", denoised_rcams.grad.abs().mean().item())
