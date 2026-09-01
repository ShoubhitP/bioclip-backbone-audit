import torch
import open_clip
from src.data import load_flowers102
from src.features import extract_preprojection_features
from collections import Counter


MODEL_ID = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
model, _, _ = open_clip.create_model_and_transforms(MODEL_ID)

images = torch.randn(4, 3, 224, 224)

features = extract_preprojection_features(model, images)

print("features shape:", features.shape)
print(
    "feature norms:",
    torch.linalg.vector_norm(features, dim=1)
)

support_pool, test_dataset = load_flowers102(root="data")
print("support_pool length:", len(support_pool))
print("test_dataset length:", len(test_dataset))

labels = [label for _, label in support_pool]
counts = Counter(labels)

print("num classes:", len(counts))
print("min per class:", min(counts.values()))
print("max per class:", max(counts.values()))
