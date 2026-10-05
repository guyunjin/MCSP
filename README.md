# Unsupervised Multimodal Intent Discovery via MLLM-Guided Concept Generation and Semantic Propagation

This repository provides the official PyTorch implementation of:

[Unsupervised Multimodal Intent Discovery via MLLM-Guided Concept Generation and Semantic Propagation](https://arxiv.org/abs/2607.21908) (**Accepted at ACM Multimedia 2026, MM '26**).

## 1. Introduction

Unsupervised multimodal intent discovery aims to identify latent intents from unlabeled text, video, and audio. We propose MCSP, a framework that combines MLLM-guided concept generation with semantic propagation. MCSP selects reliable cluster representatives, generates interpretable intent concepts through contrastive reasoning, and uses these concepts to guide graph propagation and representation learning.

This repository includes implementations and configurations for MIntRec, MIntRec2.0, and MELD-DA.

## 2. Dependencies

The recorded server environment uses **Python 3.9.23**, **PyTorch 2.8.0**, and **CUDA 12.8**. We recommend using Anaconda to create an environment:

```bash
conda create -n mcsp python=3.9.23 pip=25.2 -y
conda activate mcsp

python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
```

The existing server environment is named `umc`; it does not need to be renamed. Keep the pinned NLP dependency versions together. The package versions come from the server environment report; a fresh installation and an end-to-end run of this release have not yet been independently verified. See [environment notes](docs/environment.md) for the full record and a GPU computation check.

## 3. Usage

### 3.1 Data

The data source is the same as [HIER](https://github.com/thuiar/HIER):

[Download data from Google Drive](https://drive.google.com/drive/folders/1nCkhkz72F6ucseB73XVbqCaDG-pjhpSS)

MCSP loads TSV utterance files and pre-extracted Swin video and WavLM audio features. Prepare them in the following structure:

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

Set `DATA_ROOT` to the parent directory of the datasets. The feature files must be pickle dictionaries whose sample IDs match the TSV files. If the downloaded data is in another format, prepare the TSV files and feature dictionaries before running MCSP.

For the paper's evaluation protocol, merge the original training, validation, and test partitions and repartition the samples at a 4:1 training-to-test ratio. The loader combines the supplied `train.tsv` and `dev.tsv`; use files that follow this protocol when reproducing the paper results.

Place a Hugging Face-compatible [BERT-base-uncased checkpoint](https://huggingface.co/google-bert/bert-base-uncased) at `uncased_L-12_H-768_A-12/` in the repository root. If you use another location, update both [`configs/__init__.py`](configs/__init__.py) and `pretrained_bert_model` in the selected MCSP configuration.

### 3.2 Configuration Files

The dataset configurations are under [`configs/`](configs):

| Dataset | Configuration | Video features | Audio features |
| --- | --- | --- | --- |
| MIntRec | `mcsp_MIntRec.py` | `swin_feats.pkl` | `wavlm_feats.pkl` |
| MIntRec2.0 | `mcsp_MIntRec2.py` | `swin_roi.pkl` | `wavlm_feats.pkl` |
| MELD-DA | `mcsp_MELD-DA.py` | `swin_feats.pkl` | `wavlm_feats.pkl` |

For a first run, set these values in the corresponding configuration:

```python
'pretrain': True,
'train': True,
'save_model': True,
'use_llm': False,
```

The configuration overrides command-line values with the same name. Configure `pretrain`, `train`, and `save_model` in the Python file before running an example script.

With `use_llm=False`, the code loads the included [concept bank](methods/unsupervised/MCSP/intent_concepts.json), which covers seeds `0` through `4` for all three datasets. This workflow does not require an API key or remote model calls.

To generate new concepts online, set `use_llm=True`, provide `MCSP_API_KEY` through the environment, and configure a video-capable `llm_model_name` supported by the endpoint in [`manager.py`](methods/unsupervised/MCSP/manager.py). Review the raw-video paths and provider settings in [`mllm_reasoning.py`](methods/unsupervised/MCSP/mllm_reasoning.py). The included concept bank is the default workflow.

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

For a single MIntRec run:

```bash
python run.py \
  --dataset MIntRec \
  --data_path "$DATA_ROOT" \
  --multimodal_method mcsp \
  --method mcsp \
  --text_backbone bert-base-uncased \
  --config_file_name mcsp_MIntRec \
  --video_feats_path swin_feats.pkl \
  --audio_feats_path wavlm_feats.pkl \
  --seed 0 \
  --gpu_id "$GPU_ID" \
  --save_results \
  --results_file_name mintrec_mcsp.csv \
  --output_path outputs/MIntRec
```

To reuse a saved pretraining checkpoint, set `pretrain=False` and `train=True`. For testing only, set `pretrain=False` and `train=False`, then run the same command with the dataset, seed, backbone, and output path used during training. Testing requires both the pretraining and final checkpoints:

```text
<output_path>/mcsp_mcsp_<dataset>_bert-base-uncased_<seed>/models/
├── pretrain/
│   └── pytorch_model.bin
└── pytorch_model.bin
```

Metrics, timing summaries, representative samples, and prediction artifacts are written below `--output_path`. Aggregate test metrics are saved under `results/` when `--save_results` is enabled.

## 4. Model

The overview of MCSP:

![MCSP framework: multimodal pretraining, MLLM-guided concept generation, and semantic propagation](assets/framework.png)

The framework has three stages:

1. **Multimodal unsupervised pretraining:** learn a joint representation using modality-masked contrastive views.
2. **MLLM-guided concept generation:** select reliable representatives and derive interpretable intent concepts through contrastive reasoning.
3. **Semantic propagation:** adjust graph edges with concept similarity, propagate concept labels, and refine representations using high-confidence samples.

## 5. Experimental Results

Results reported in the paper are shown below. Scores are averaged over five runs with seeds `0` through `4` and a known number of intent categories. The Qwen and Gemini variants use Qwen3-VL and Gemini-3.0-Pro, respectively, for concept generation.

| Dataset | Method | ACC | ARI | NMI | FMI | Avg. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MIntRec | MCSP-Qwen | 44.54 | 24.85 | 49.07 | 29.61 | 37.02 |
| MIntRec | MCSP-Gemini | 45.21 | 25.59 | 48.42 | 30.28 | 37.37 |
| MIntRec2.0 | MCSP-Qwen | 29.54 | 15.00 | 37.37 | 18.96 | 25.22 |
| MIntRec2.0 | MCSP-Gemini | 29.19 | 15.03 | 37.33 | 18.92 | 25.12 |
| MELD-DA | MCSP-Qwen | 34.62 | 22.07 | 21.49 | 33.74 | 27.98 |
| MELD-DA | MCSP-Gemini | 34.57 | 21.57 | 21.61 | 33.16 | 27.73 |

These are the published experiment results. The included cached concept bank does not record the generating MLLM, so it should not be assumed to identify either variant in this table.

## 6. Citation

If you use this code or the results in your research, please cite:

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

Parts of this implementation build on [UMC](https://github.com/thuiar/UMC), with Transformer components adapted from [Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer) and [fairseq](https://github.com/facebookresearch/fairseq). We thank the authors and contributors for sharing their work.

For questions or reproducibility issues, please open a GitHub issue with your environment, command, configuration, and relevant error log.
