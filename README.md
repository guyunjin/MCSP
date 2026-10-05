# Unsupervised Multimodal Intent Discovery via MLLM-Guided Concept Generation and Semantic Propagation

This repository contains the official PyTorch implementation of [*Unsupervised Multimodal Intent Discovery via MLLM-Guided Concept Generation and Semantic Propagation*](https://arxiv.org/abs/2607.21908) (**Accepted at ACM Multimedia 2026 (MM '26)**).

## 1. Introduction

Unsupervised multimodal intent discovery seeks latent intents from unlabeled multimodal dialogue data. MCSP first identifies representative samples from initial clusters, obtains high-level semantic concepts through MLLM-guided contrastive reasoning or an included concept bank, and then propagates these concepts over a semantically weighted graph. The propagated pseudo-labels are used to refine multimodal representations.
The implementation supports text, video, and audio features and includes configurations for MIntRec, MIntRec2.0, and MELD-DA.

## 2. Dependencies

The original experiments use Python 3.8, PyTorch 1.8.1, and CUDA 11.1. Create an environment whose PyTorch build matches the CUDA version available on your machine.

```bash
conda create -n mcsp python=3.8 -y
conda activate mcsp

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

The optional online MLLM concept-generation path requires an OpenAI-compatible client. It is included in `requirements.txt`; use a version compatible with your Python runtime if your environment requires a constraint.

## 3. Data and Backbone Preparation

Obtain each dataset from its original release and follow its license and access conditions:

- [MIntRec](https://github.com/thuiar/MIntRec)
- [MIntRec2.0](https://github.com/thuiar/MIntRec2.0)
- [MELD](https://github.com/declare-lab/MELD)

`--data_path` must point to the directory that contains the dataset directories. The loader expects the following structure; feature-file names are supplied on the command line.

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

The code resolves the text encoder from the local path configured in [`configs/__init__.py`](configs/__init__.py). Download the uncased BERT-base checkpoint from the [official BERT release](https://github.com/google-research/bert) and either place it at `uncased_L-12_H-768_A-12/` in the repository root or change that mapping to the checkpoint directory on your machine. The directory must be readable by Hugging Face `from_pretrained` and contain the model weights, configuration, and tokenizer files.

This repository does not redistribute datasets, extracted features, pretrained model weights, or API credentials.

## 4. Usage

### 4.1 Offline reproduction

By default, `use_llm=False`. MCSP then loads the precomputed concepts in [`methods/unsupervised/MCSP/intent_concepts.json`](methods/unsupervised/MCSP/intent_concepts.json). The file contains entries for seeds `0` through `4` for MIntRec, MIntRec2.0, and MELD-DA.

The first run must create a pretraining checkpoint. In the selected configuration file, set:

```python
'pretrain': True,
'train': True,
'save_model': True,
```

For MIntRec, run:

```bash
python run.py \
  --dataset MIntRec \
  --data_path /path/to/DATA_ROOT \
  --multimodal_method mcsp \
  --method mcsp \
  --text_backbone bert-base-uncased \
  --config_file_name mcsp_MIntRec \
  --video_feats_path swin_feats.pkl \
  --audio_feats_path wavlm_feats.pkl \
  --seed 0 \
  --gpu_id 0 \
  --save_results \
  --results_file_name mintrec_mcsp.csv \
  --output_path outputs/MIntRec
```

For later runs using the saved pretraining checkpoint, reset `pretrain` to `False` and keep `train=True`. Use the same dataset, seed, and `--output_path` so that the checkpoint remains at:

```text
outputs/MIntRec/mcsp_mcsp_MIntRec_bert-base-uncased_0/models/pretrain/pytorch_model.bin
```

To run the other supported datasets, use the matching configuration and feature file:

| Dataset | Configuration | Video features | Audio features |
| --- | --- | --- | --- |
| MIntRec | `mcsp_MIntRec` | `swin_feats.pkl` | `wavlm_feats.pkl` |
| MIntRec2.0 | `mcsp_MIntRec2` | `swin_roi.pkl` | `wavlm_feats.pkl` |
| MELD-DA | `mcsp_MELD-DA` | `swin_feats.pkl` | `wavlm_feats.pkl` |

The scripts under [`examples/`](examples) run the five seeds used by the included concept bank. Set `DATA_ROOT` and, optionally, `GPU_ID` before execution:

```bash
DATA_ROOT=/path/to/DATA_ROOT GPU_ID=0 bash examples/run_mcsp.sh
```

### 4.2 Optional MLLM concept generation

Set `use_llm=True` in the selected MCSP configuration and provide an API key through an environment variable:

```bash
export MCSP_API_KEY='your-api-key'
```

The online path may send representative text and video evidence to the configured OpenAI-compatible endpoint. Review the endpoint, model name, video paths, dataset terms, privacy requirements, and billing before enabling it. Keep credentials out of configuration files, commits, logs, and issue reports.

## 5. Outputs

Each run writes its artifacts below the selected `--output_path`, including model checkpoints, training metrics, timing summaries, representative samples, and final high-quality sample assignments. With `--save_results`, aggregate test metrics are appended to the specified CSV file under `results/`.

## 6. Citation

If you use this code or paper, please cite:

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

This implementation builds on and adapts components from [UMC](https://github.com/thuiar/UMC). We thank its authors and contributors for making their work available.

For questions or reproducibility issues, please open a GitHub issue with the operating system, package versions, command, configuration file, and relevant error log.
