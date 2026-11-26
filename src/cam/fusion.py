import torch
import torch.nn.functional as F


def extract_cams(model, images, image_labels=None, size=(512, 512)):
    vit_outputs = model.vit(images)
    flipped_outputs = model.vit(torch.flip(images, dims=(3,)))

    cams = (
        vit_outputs["rcams"]
        + torch.flip(flipped_outputs["rcams"], dims=(3,))
    )

    if image_labels is not None:
        cams = cams * image_labels.view(
            images.shape[0],
            -1,
            1,
            1,
        )

    cams = F.interpolate(
        cams,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    return model._max_norm(cams)


def make_pseudo_labels(cams, image_labels, threshold=0.0):
    scores = cams.to(image_labels.device)

    mask = image_labels.to(scores.device).bool().view(-1, scores.shape[1], 1, 1)
    inactive = torch.full_like(scores, -1e6)
    scores = torch.where(mask, scores, inactive)

    if threshold > 0.0:
        background = torch.full(
            (scores.shape[0], 1, scores.shape[2], scores.shape[3]),
            threshold,
            device=scores.device,
            dtype=scores.dtype,
        )
        scores = torch.cat((background, scores), dim=1)
        pseudo = scores.argmax(dim=1) - 1
        pseudo = pseudo.clamp(min=0)
    else:
        pseudo = scores.argmax(dim=1)

    return pseudo.to(torch.uint8).cpu().numpy()
