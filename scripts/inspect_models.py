import open_clip
import gc

BIOCLIP_ID = "hf-hub:imageomics/bioclip-2"
LAION_ID = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"

bioModel, _, _ = open_clip.create_model_and_transforms(BIOCLIP_ID)
bioModel.eval()

print(type(bioModel.visual))
print(hasattr(bioModel.visual, "proj"))
if hasattr(bioModel.visual, "proj"):
    print(bioModel.visual.proj.shape)

for key, tensor in bioModel.state_dict().items():
    if key == "visual.proj":
        print(key, tensor.shape)

for key in bioModel.state_dict().keys():
    if "replay" in key.lower():
        print(key)

for key in bioModel.state_dict().keys():
    if "projector" in key.lower():
        print(key)

del bioModel
gc.collect()

laionModel, _, _ = open_clip.create_model_and_transforms(LAION_ID)
laionModel.eval()

print(type(laionModel.visual))
print(hasattr(laionModel.visual, "proj"))
if hasattr(laionModel.visual, "proj"):
    print(laionModel.visual.proj.shape)

for key, tensor in laionModel.state_dict().items():
    if key == "visual.proj":
        print(key, tensor.shape)
