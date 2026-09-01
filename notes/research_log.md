# Research Log

## Project Question

How does BioCLIP 2's biological specialization change the transfer behavior of the shared visual backbone relative to its LAION initialization?

---

## Decisions

### Why compare BioCLIP 2 rather than BioCLIP 2.5?

BioCLIP 2 retains the ViT-L/14 architecture of the LAION checkpoint used to initialize it, enabling a cleaner paired comparison of the visual backbone before and after biological specialization.

BioCLIP 2.5 uses a different ViT-H/14 backbone, which would introduce architecture as an additional confounding variable. For this study, the goal is to keep the model architecture fixed and compare representations before and after BioCLIP 2's biological training.

### Why use pre-projection features?

BioCLIP 2 used separate visual projectors for biological and replay data during training, while the visual backbone itself was shared.

The primary experiment therefore focuses on the shared visual backbone rather than the released projected CLIP embedding.

This matters because using the final projected representation could mix together two effects:

1. changes learned by the shared visual backbone;
2. specialization introduced by the biological projection head.

The primary representation for this project will therefore be the pooled visual feature immediately before `visual.proj`.

### Why compare against the LAION initialization?

BioCLIP 2 was initialized from a LAION-2B ViT-L/14 CLIP checkpoint.

This gives a particularly useful control:

- LAION CLIP represents the model before biological specialization;
- BioCLIP 2 represents the related model after biological specialization.

The goal is not to claim that every difference is caused by one specific component such as experience replay. Instead, the experiment measures how the shared visual representation differs after the overall BioCLIP 2 training procedure.

---

# Experiment 0 — Checkpoint Inspection

## Goal

Before running any classification experiments, verify the actual structure of the released checkpoints.

Specifically:

1. determine the type of visual encoder used by BioCLIP 2;
2. determine whether the released checkpoint exposes a visual projection layer;
3. determine the dimensions of that projection;
4. search the released BioCLIP 2 state dictionary for evidence of additional replay/projector parameters;
5. confirm that the LAION initialization has the same visual architecture and projection dimensions.

This establishes whether a direct pre-projection backbone comparison is structurally valid.

---

## Prediction

### Prediction 1 — Projection dimensions

The BioCLIP 2 visual backbone was expected to produce a 1024-dimensional representation and project it into a 768-dimensional CLIP embedding.

Therefore, I expected:

```text
visual.proj.shape = [1024, 768]
```

because a batch of visual features with shape

```text
[B, 1024]
```

multiplied by a projection matrix with shape

```text
[1024, 768]
```

produces:

```text
[B, 768]
```

### Prediction 2 — Number of released visual projectors

BioCLIP 2 used separate biological and replay visual projectors during training.

However, I expected the released inference checkpoint to expose only one main visual projector, most likely the biological projector, rather than both training-specific heads.

This was a hypothesis to verify rather than an assumption.

### Prediction 3 — LAION checkpoint structure

Because BioCLIP 2 was initialized from the LAION ViT-L/14 checkpoint, I expected the LAION model to use the same OpenCLIP `VisionTransformer` architecture and to expose the same `[1024, 768]` visual projection.

Matching at this structural point is important because the main experiment will compare the two models' 1024-dimensional pre-projection features.

---

## Procedure

Both models were loaded through OpenCLIP.

BioCLIP 2:

```python
BIOCLIP_ID = "hf-hub:imageomics/bioclip-2"
```

LAION initialization:

```python
LAION_ID = "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
```

After loading, each model was placed into evaluation mode using:

```python
model.eval()
```

The following properties were inspected:

```python
type(model.visual)
hasattr(model.visual, "proj")
model.visual.proj.shape
```

The model's `state_dict()` was also searched for projection-related parameter names.

---

## Observation

### BioCLIP 2

The released BioCLIP 2 model reported:

```text
<class 'open_clip.transformer.VisionTransformer'>
True
torch.Size([1024, 768])
visual.proj torch.Size([1024, 768])
```

Therefore:

- the visual encoder is OpenCLIP's `VisionTransformer`;
- the visual encoder exposes a `proj` parameter;
- the projection matrix has shape `[1024, 768]`.

A broad search for keys containing `"proj"` returned many additional parameters such as:

```text
visual.transformer.resblocks.*.attn.in_proj_weight
visual.transformer.resblocks.*.attn.out_proj.weight
visual.transformer.resblocks.*.mlp.c_proj.weight
```

These are internal Transformer projections and are not additional final visual projection heads.

For example, attention layers contained matrices such as:

```text
attn.in_proj_weight: [3072, 1024]
```

The value `3072` corresponds to:

```text
3 × 1024
```

because the attention layer constructs query, key, and value projections.

The Transformer MLP also contained projection matrices involving dimensions `1024` and `4096`, corresponding to the feed-forward expansion and contraction inside each Transformer block.

The visual encoder contained residual blocks numbered from `0` through `23`, confirming 24 Transformer blocks.

No obviously named `"replay"` parameter or additional `"projector"` parameter was identified in the released state dictionary.

The only obvious top-level final visual projection found was:

```text
visual.proj
```

### LAION CLIP

The LAION checkpoint reported:

```text
<class 'open_clip.transformer.VisionTransformer'>
True
torch.Size([1024, 768])
visual.proj torch.Size([1024, 768])
```

This exactly matches the corresponding BioCLIP 2 visual architecture at the level inspected.

---

## Interpretation

Experiment 0 confirms that BioCLIP 2 and its LAION initialization are structurally comparable at the point relevant to this project.

Both models use:

```text
OpenCLIP VisionTransformer
```

with the following conceptual flow:

```text
image
  ↓
ViT visual backbone
  ↓
1024-dimensional pooled visual representation
  ↓
visual.proj [1024 × 768]
  ↓
768-dimensional CLIP embedding
```

The main experiment should therefore compare the models at the 1024-dimensional representation immediately before `visual.proj`.

This avoids allowing differences in the released biological projection head to directly determine the primary comparison.

The inspection also suggests that the publicly released BioCLIP 2 checkpoint exposes one obvious top-level visual projection head. Although BioCLIP 2 used a separate replay projector during training, no explicitly named replay projector was identified in the released state dictionary.

Therefore, any later optional projector analysis should be framed as:

```text
pre-projection backbone feature
vs.
released biological projected feature
```

rather than assuming that both biological and replay projectors are publicly available.

This experiment establishes structural comparability, but it does not yet prove that the feature extraction code is stopping at exactly the correct point in both models' forward passes.

That must be verified next.

---

# Experiment 0B — Pre-Projection Feature Extraction

## Goal

Identify and extract the 1024-dimensional pooled visual representation immediately before `visual.proj`.

Then verify that applying `visual.proj` manually to this representation reproduces OpenCLIP's normal 768-dimensional image representation up to numerical precision.

Conceptually, the validation will test:

```text
our extracted 1024-D feature
        ↓
manual multiplication by visual.proj
        ↓
768-D result
```

against:

```text
OpenCLIP normal image forward path
        ↓
768-D result
```

If the two results match numerically, it provides strong evidence that the correct pre-projection extraction point has been identified.

## Prediction

Not yet written.

## Observation

Not yet run.

## Interpretation

Not yet written.
