from collections import defaultdict
import random


def sample_support_indices(labels, k, seed):
    class_indices = defaultdict(list)

    for idx, lbl in enumerate(labels):
        class_indices[lbl].append(idx)

    rng = random.Random(seed)

    support_indices = []

    for lbl in sorted(class_indices):
        indices = class_indices[lbl]

        if len(indices) < k:
            raise ValueError(
                f"Not enough samples for class {lbl}. "
                f"Required: {k}, available: {len(indices)}"
            )

        sampled_indices = rng.sample(indices, k)
        support_indices.extend(sampled_indices)

    return support_indices