import open_clip

BIOCLIP_ID = "hf-hub:imageomics/bioclip-2"
LAION_ID = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"

model, _, _ = open_clip.create_model_and_transforms(BIOCLIP_ID)
model.eval()

print(type(model.visual))
print(hasattr(model.visual, "proj"))
if hasattr(model.visual, "proj"):
    print(model.visual.proj.shape)