# Unsupervised Multimodal Intent Discovery via MLLM-Guided Concept Generation and Semantic Propagation

This repository provides the official PyTorch implementation of:

[Unsupervised Multimodal Intent Discovery via MLLM-Guided Concept Generation and Semantic Propagation](https://arxiv.org/abs/2607.21908) (**Accepted at ACM Multimedia 2026, MM '26**).

## 1. Introduction

Unsupervised multimodal intent discovery aims to uncover latent intents from unlabeled multimodal dialogues, but remains challenging due to the lack of explicit semantic supervision. Existing methods often provide limited interpretability, as their refinement mainly relies on geometric similarity rather than high-level semantic guidance. To address these limitations, we propose **MCSP**, a fully unsupervised method that introduces semantic refinement based on concepts into multimodal intent discovery.

## 2. Dependencies

We recommend using Anaconda to create an environment:

```bash
conda create -n mcsp python=3.9.23 pip=25.2 -y
conda activate mcsp

python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

## 3. Usage

### 3.1 Data

The data can be downloaded through the following links: [Download data from Google Drive](https://drive.google.com/drive/folders/1nCkhkz72F6ucseB73XVbqCaDG-pjhpSS)

MCSP loads TSV files and pre-extracted video and audio features. Prepare them in the following structure:

```text
DATA_ROOT/
├── MIntRec/
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   ├── video_data/
│   │   └── swin_feats.pkl
│   └── audio_data/
│       └── wavlm_feats.pkl
├── MIntRec2.0/
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   ├── video_data/
│   │   └── swin_roi.pkl
│   └── audio_data/
│       └── wavlm_feats.pkl
└── MELD-DA/
    ├── train.tsv
    ├── dev.tsv
    ├── test.tsv
    ├── video_data/
    │   └── swin_feats.pkl
    └── audio_data/
        └── wavlm_feats.pkl
```

Set `DATA_ROOT` to the parent directory of the datasets. The feature files must be pickle dictionaries whose sample IDs match the TSV files. Place a Hugging Face-compatible [BERT-base-uncased checkpoint](https://huggingface.co/google-bert/bert-base-uncased) at `uncased_L-12_H-768_A-12/` in the repository root. If you use another location, update both [`configs/__init__.py`](configs/__init__.py) and `pretrained_bert_model` in the selected MCSP configuration.

### 3.2 Configuration Files

The dataset configurations are under [`configs/`](configs):

| Dataset | Configuration | Video features | Audio features |
| --- | --- | --- | --- |
| MIntRec | `mcsp_MIntRec.py` | `swin_feats.pkl` | `wavlm_feats.pkl` |
| MIntRec2.0 | `mcsp_MIntRec2.py` | `swin_roi.pkl` | `wavlm_feats.pkl` |
| MELD-DA | `mcsp_MELD-DA.py` | `swin_feats.pkl` | `wavlm_feats.pkl` |

Set `use_llm=False` to use the bundled concept bank at `methods/unsupervised/MCSP/intent_concepts.json`, which covers seeds `0`–`4` for all three datasets and requires no API key or remote model calls. Set `use_llm=True` to generate new concepts online, which requires `MCSP_API_KEY` via the environment and a video-capable `llm_model_name` supported by the endpoint in `manager.py`.

### 3.3 Run Training / Testing

Run commands from the repository root. Each script trains and then evaluates seeds `0` through `4`:

```bash
export DATA_ROOT=/path/to/DATA_ROOT
export GPU_ID=0

# MIntRec
bash examples/run_mcsp.sh

# MIntRec2.0
bash examples/run_mcsp_mintrec2.sh

# MELD-DA
bash examples/run_mcsp_meld.sh
```

## 4. Model

The overview of MCSP:

![MCSP framework: multimodal pretraining, MLLM-guided concept generation, and semantic propagation](assets/framework.png)

The method has three stages:

1. **Multimodal unsupervised pretraining:** performs mask-based contrastive learning on multimodal inputs to establish a robust joint embedding space.
2. **MLLM-guided concept generation:** leverages MLLM-guided reasoning to generate semantic concepts from high-quality representative samples.
3. **Semantic propagation:** integrates evidence diffusion and concept-driven supervision to refine the embedding manifold for accurate intent identification.

## 5. Experimental Results

![Experimental results on MIntRec, MIntRec2.0, and MELD-DA](assets/experimental_results.png)


## 6. Citation

If you are insterested in this work, and want to use the codes or results in this repository, please star this repository and cite by:

```bibtex
@misc{gu2026unsupervised,
  title={Unsupervised Multimodal Intent Discovery via MLLM-Guided Concept Generation and Semantic Propagation},
  author={Yunjin Gu and Qianrui Zhou and Hua Xu},
  year={2026},
  eprint={2607.21908},
  archivePrefix={arXiv},
  primaryClass={cs.MM},
  url={https://arxiv.org/abs/2607.21908}
}
```

## 7. Acknowledgements

Parts of this implementation build on [UMC](https://github.com/thuiar/UMC). We thank the authors and contributors for sharing their work.

For questions or reproducibility issues, please open a GitHub issue with your environment, command, configuration, and relevant error log.
