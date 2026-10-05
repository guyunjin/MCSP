# Environment Record

The authors supplied this record from their existing `umc` Conda environment on October 5, 2026. The package pins in `requirements.txt` follow that record and the imports used by this repository.

| Component | Recorded value |
| --- | --- |
| Python | 3.9.23 |
| pip | 25.2 |
| PyTorch | 2.8.0+cu128 |
| torchvision package | 0.23.0 |
| CUDA runtime reported by PyTorch | 12.8 |
| cuDNN reported by PyTorch | 91002 (9.10.2) |
| GPU | 4 × NVIDIA Tesla V100-PCIE-32GB |
| NVIDIA driver | 560.35.03 |
| `torch.cuda.is_available()` | `True` |
| `python -m pip check` | `No broken requirements found.` |

The report establishes the installed versions and GPU visibility. It does not establish that a fresh environment installation, CUDA computation, or an end-to-end experiment with the release copy has passed.

## Dependency Scope

The requirements include the modules imported by the code and the recorded versions of the NLP dependencies used by Sentence Transformers. The server's unrelated Google SDKs and package-management tools are omitted. CUDA runtime libraries and Triton are installed through PyTorch's package dependencies.

The old NLP versions must remain compatible with each other:

- `sentence-transformers==2.2.2` imports `huggingface_hub.cached_download`; the recorded `huggingface-hub==0.10.1` provides that interface.
- `transformers==4.20.1` provides the `AdamW` import used by the training code, and is paired with `tokenizers==0.12.1` in the server record.
- `torchvision==0.23.0` is paired with `torch==2.8.0` and is also a dependency of Sentence Transformers.
- `openai==2.9.0` is needed for online MLLM concept generation; the default cached-concept workflow makes no remote model calls.

The PyTorch requirement uses `torch==2.8.0`. The CUDA wheel source is specified separately in the README so the installation selects the reported CUDA 12.8 build.

## GPU Computation Check

Run this in the environment that will run MCSP. It checks a CUDA matrix multiplication and backward pass without starting model training.

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("Compiled architectures:", torch.cuda.get_arch_list())
assert torch.cuda.is_available(), "CUDA is unavailable"
print("GPU:", torch.cuda.get_device_name(0))
print("GPU capability:", torch.cuda.get_device_capability(0))

x = torch.arange(16, dtype=torch.float32, device="cuda").reshape(4, 4)
x.requires_grad_(True)
y = x @ x.T
y.sum().backward()
torch.cuda.synchronize()
cpu_x = x.detach().cpu()
torch.testing.assert_close(y.detach().cpu(), cpu_x @ cpu_x.T)
torch.testing.assert_close(x.grad.cpu(), 2 * cpu_x.sum(dim=0).expand_as(cpu_x))
print("GPU matrix multiplication and backward: PASS")
PY
```

The official PyTorch 2.8.0 CUDA 12.8 build configuration includes compute capability 7.0, used by V100. The driver version in the report falls within NVIDIA's CUDA 12.x minor-compatibility range. The computation check above still needs to be run on the target server.

## Sources

- [PyTorch 2.8.0 installation commands](https://pytorch.org/get-started/previous-versions/)
- [PyTorch 2.8.0 CUDA build configuration](https://github.com/pytorch/pytorch/blob/v2.8.0/.ci/manywheel/build_cuda.sh)
- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [Sentence Transformers 2.2.2 imports](https://github.com/UKPLab/sentence-transformers/blob/v2.2.2/sentence_transformers/SentenceTransformer.py)
- [Hugging Face Hub 0.10.1 exports](https://github.com/huggingface/huggingface_hub/blob/v0.10.1/src/huggingface_hub/__init__.py)
- [Transformers 4.20.1 exports](https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/__init__.py)
