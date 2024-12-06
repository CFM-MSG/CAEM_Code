# Counterfactually-Augmented Event Matching for De-biased Temporal Sentence Grounding

Code for the paper "Counterfactually-Augmented Event Matching for De-biased Temporal Sentence Grounding (ACM MM 24)".

## Installation
You can build the conda environment simply by,
```bash
conda env create -f environment.yml
```

## Dataset Preparation
#### Pretrained Models and Visual Features
You can download the pretrained models of our method at [Google Drive](https://drive.google.com/file/d/1UHe1YzoKQS_9jdK_vaRuNgwPN-HvAhnp/view?usp=sharing)

Please put the pretrained models into the directories `CAEM/checkpoints/Charades`

For video features, you can follow [here](https://github.com/microsoft/VideoX/tree/master/MS-2D-TAN) for more details. Please save the video feature files as `CAEM/data/Video_Feature/i3d_charades_features.hdf5`

#### Word Embeddings
We provide the part in the code that downloads word embeddings to the corresponding directory. You don't need to do anything else.

## Quick Start
```
conda activate env_caem
cd CAEM
```

### Charades-CD

Train:
```
CUDA_VISIBLE_DEVICES=0 python train.py --cfg experiments/charades-cd/CAEM.yaml --verbose
```

Evaluate:
```
CUDA_VISIBLE_DEVICES=0 python test.py --cfg experiments/charades-cd/CAEM.yaml --verbose --split test_iid
```