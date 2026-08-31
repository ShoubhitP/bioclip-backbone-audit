# Research Log

## Project Question

How does BioCLIP 2's biological specialization change the transfer behavior of the shared visual backbone relative to its LAION initialization?

---

## Decisions

### Why compare BioCLIP 2 rather than BioCLIP 2.5?

BioCLIP 2 retains the ViT-L/14 architecture of the LAION checkpoint used to initialize it, enabling a cleaner paired comparison of the backbone before and after biological specialization.

### Why use pre-projection features?

BioCLIP 2 used separate visual projectors for biological and replay data during training. The primary experiment therefore focuses on the shared visual backbone rather than allowing the biological projection head to confound the comparison.

---

## Experiment 0 — Checkpoint Inspection

### Prediction

Not yet written.

### Observation

Not yet run.

### Interpretation

Not yet written.
