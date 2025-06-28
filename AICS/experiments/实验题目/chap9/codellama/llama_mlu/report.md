# Cambricon PyTorch Model Migration Report
## Cambricon PyTorch Changes
| No. |  File  |  Description  |
| 1 | generation.py:12 | add "import torch_mlu" |
| 2 | generation.py:23 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 3 | generation.py:24 | change "device = "cuda"" to "device = "mlu" " |
| 4 | generation.py:77 | change "if device == "cuda":" to "if device == "mlu": " |
| 5 | generation.py:78 | change "torch.distributed.init_process_group("nccl")" to "torch.distributed.init_process_group("cncl") " |
| 6 | generation.py:87 | change "if device == "cuda":" to "if device == "mlu": " |
| 7 | generation.py:88 | change "torch.cuda.set_device(local_rank)" to "torch.mlu.set_device(local_rank) " |
| 8 | generation.py:115 | change "if device == "cuda":" to "if device == "mlu": " |
| 9 | generation.py:116 | change "if torch.cuda.is_bf16_supported():" to "if torch.mlu.is_bf16_supported(): " |
| 10 | generation.py:117 | change "torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)" to "torch.set_default_tensor_type(torch.mlu.BFloat16Tensor) " |
| 11 | generation.py:119 | change "torch.set_default_tensor_type(torch.cuda.HalfTensor)" to "torch.set_default_tensor_type(torch.mlu.HalfTensor) " |
| 12 | model.py:9 | add "import torch_mlu" |
| 13 | model.py:18 | change "if torch.cuda.is_available():" to "if torch.mlu.is_available(): " |
| 14 | model.py:19 | change "device = "cuda"" to "device = "mlu" " |
| 15 | model.py:77 | change "if not torch.cuda.is_available():" to "if not torch.mlu.is_available(): " |
| 16 | model.py:287 | change "self.freqs_cis = self.freqs_cis.to("cuda" if device == "cuda" else "cpu")" to "self.freqs_cis = self.freqs_cis.to("mlu" if device == "mlu" else "cpu") " |
