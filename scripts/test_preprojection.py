import torch
import open_clip

MODEL_ID = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"

model, _, _ = open_clip.create_model_and_transforms(MODEL_ID)
model.eval()

image_size = model.visual.image_size
print("image_size:", image_size)

if isinstance(image_size, tuple):
    h, w = image_size
else:
    h = w = image_size

x = torch.randn(1, 3, h, w)

with torch.no_grad():
    # Manually reproduce the visual forward pass up to projection
    z = model.visual._embeds(x)
    z = model.visual.transformer(z)
    pooled, tokens = model.visual._pool(z)

    manual_projected = pooled @ model.visual.proj
    normal_output = model.visual(x)

print("pooled shape:", pooled.shape)
print("proj shape:", model.visual.proj.shape)
print("manual projected shape:", manual_projected.shape)
print("normal output shape:", normal_output.shape)

max_diff = (manual_projected - normal_output).abs().max().item()

print("max abs difference:", max_diff)
print(
    "allclose:",
    torch.allclose(
        manual_projected,
        normal_output,
        atol=1e-6,
        rtol=1e-5,
    ),
)