from collections import Counter
import open_clip
import torch
from src.data import load_flowers102
from src.features import extract_preprojection_features
from src.fewshot import sample_support_indices

MODEL_ID = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
# To test BioCLIP instead:
# MODEL_ID = "hf-hub:imageomics/bioclip-2"

model, _, _ = open_clip.create_model_and_transforms(MODEL_ID)
model.eval()

images = torch.randn(4, 3, 224, 224)

features = extract_preprojection_features(model, images)

print("features shape:", features.shape)
print(
    "feature norms:",
    torch.linalg.vector_norm(features, dim=1),
)

support_pool, test_dataset = load_flowers102("data")

print("support_pool length:", len(support_pool))
print("test_dataset length:", len(test_dataset))

labels = [label for _, label in support_pool]

counts = Counter(labels)

print("num classes:", len(counts))
print("min per class:", min(counts.values()))
print("max per class:", max(counts.values()))

a = sample_support_indices(labels, k=5, seed=42)
b = sample_support_indices(labels, k=5, seed=42)
c = sample_support_indices(labels, k=5, seed=43)

print("number selected:", len(a))
print("same seed identical:", a == b)
print("different seed identical:", a == c)

selected_labels = [labels[i] for i in a]
selected_counts = Counter(selected_labels)

print("selected num classes:", len(selected_counts))
print("selected min per class:", min(selected_counts.values()))
print("selected max per class:", max(selected_counts.values()))