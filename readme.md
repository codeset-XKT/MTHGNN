# MTHGNN

**Multiscale Temporal Hypergraph Neural Network for Cross-Subject EEG Emotion Recognition**

MTHGNN models emotion-related EEG activity through dynamic hypergraphs, global-local hypergraph convolution, reliability-guided temporal aggregation, and structural and temporal contrastive learning. The paper has been accepted by *IEEE Transactions on Affective Computing*.

![Overall architecture of MTHGNN](assets/framework.png)

## Architecture

1. **Dynamic hypergraph construction:** learn sample-adaptive electrode-to-hyperedge assignments at each time step.
2. **Global-local hypergraph convolution:** combine whole-brain higher-order interactions with hyperedge-level local representations.
3. **Noise-robust temporal modeling:** use a Transformer and reliability-guided attention to aggregate the EEG sequence.
4. **Multi-view contrastive learning:** expose structural and temporal view embeddings for the two contrastive objectives alongside supervised classification.

## Model file

`Model.py` implements the MTHGNN architecture. Import `MTHGNN` to build the model.

The model expects an already segmented differential-entropy tensor with shape `[batch, time_steps, 62, 5]`. Its `forward` method returns class logits with shape `[batch, num_classes]`.

```python
import torch
from Model import MTHGNN

model = MTHGNN(
    in_channels=5,
    num_electrodes=62,
    num_layers=2,
    hid_channels=32,
    num_hyperedges=10,
    num_classes=3,  # SEED: 3; SEED-IV: 4; SEED-V: 5
)

segments = torch.randn(2, 10, 62, 5)
logits = model(segments)
print(logits.shape)  # torch.Size([2, 3])
```

The model uses an entropy-scaling coefficient of 5, a reliability mixing coefficient of 0.5, and a stabilizing term of `1e-4`. `encode_views` and `encode_temporal_views` provide the embeddings used by structural and temporal contrastive learning, respectively.
