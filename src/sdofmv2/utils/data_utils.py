import pandas as pd
import torch
from torch.utils.data._utils.collate import default_collate

def safe_collate(batch):
    """
    Intercepts the batch to fix datetime objects before the default
    collator tries to turn them into tensors.
    """
    for sample in batch:
        # Check if the sample is a tuple (data, metadata)
        if isinstance(sample, tuple) and len(sample) == 2:
            metadata = sample[1]
            # Convert datetime64 to strings or unix timestamps
            if "timestamps_input" in metadata:
                metadata["timestamps_input"] = [str(t) for t in metadata["timestamps_input"]]
            if "timestamps_targets" in metadata:
                metadata["timestamps_targets"] = [str(t) for t in metadata["timestamps_targets"]]

    return default_collate(batch)
