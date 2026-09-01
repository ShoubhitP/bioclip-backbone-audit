import torch
import torch.nn.functional as F


def extract_preprojection_features(model, images):
    model.eval()

    with torch.no_grad():
        x = model.visual._embeds(images)
        x = model.visual.transformer(x)
        pooled, _ = model.visual._pool(x)

        features = F.normalize(pooled, p=2, dim=1)

    return features