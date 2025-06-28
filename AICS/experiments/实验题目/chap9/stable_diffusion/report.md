# Cambricon PyTorch Model Migration Report
## Cambricon PyTorch Changes
| No. |  File  |  Description  |
| 1 | scripts/txt2img.py:3 | add "import torch_mlu" |
| 2 | scripts/txt2img.py:28 | change "def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):" to "def load_model_from_config(config, ckpt, device=torch.device("mlu"), verbose=False): " |
| 3 | scripts/txt2img.py:43 | change "if device == torch.device("cuda"):" to "if device == torch.device("mlu"): " |
| 4 | scripts/txt2img.py:44 | change "model.cuda()" to "model.mlu() " |
| 5 | scripts/txt2img.py:184 | change "choices=["cpu", "cuda"]," to "choices=["cpu", "mlu"], " |
| 6 | scripts/txt2img.py:218 | change "device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")" to "device = torch.device("mlu") if opt.device == "mlu" else torch.device("cpu") " |
| 7 | scripts/img2img.py:5 | add "import torch_mlu" |
| 8 | scripts/img2img.py:44 | change "model.cuda()" to "model.mlu() " |
| 9 | scripts/img2img.py:190 | change "device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")" to "device = torch.device("mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 10 | scripts/img2img.py:234 | change "with precision_scope("cuda"):" to "with precision_scope("mlu"): " |
| 11 | scripts/gradio/depth2img.py:2 | add "import torch_mlu" |
| 12 | scripts/gradio/depth2img.py:25 | change ""cuda") if torch.cuda.is_available() else torch.device("cpu")" to ""mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 13 | scripts/gradio/depth2img.py:58 | change ""cuda") if torch.cuda.is_available() else torch.device("cpu")" to ""mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 14 | scripts/gradio/depth2img.py:68 | change "torch.autocast("cuda"):" to "torch.autocast("mlu"): " |
| 15 | scripts/gradio/inpainting.py:3 | add "import torch_mlu" |
| 16 | scripts/gradio/inpainting.py:34 | change ""cuda") if torch.cuda.is_available() else torch.device("cpu")" to ""mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 17 | scripts/gradio/inpainting.py:71 | change ""cuda") if torch.cuda.is_available() else torch.device("cpu")" to ""mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 18 | scripts/gradio/inpainting.py:85 | change "torch.autocast("cuda"):" to "torch.autocast("mlu"): " |
| 19 | scripts/gradio/superresolution.py:2 | add "import torch_mlu" |
| 20 | scripts/gradio/superresolution.py:26 | change ""cuda") if torch.cuda.is_available() else torch.device("cpu")" to ""mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 21 | scripts/gradio/superresolution.py:58 | change ""cuda") if torch.cuda.is_available() else torch.device("cpu")" to ""mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 22 | scripts/gradio/superresolution.py:71 | change "torch.autocast("cuda"):" to "torch.autocast("mlu"): " |
| 23 | scripts/streamlit/depth2img.py:2 | add "import torch_mlu" |
| 24 | scripts/streamlit/depth2img.py:25 | change "device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")" to "device = torch.device("mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 25 | scripts/streamlit/depth2img.py:55 | change "device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")" to "device = torch.device("mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 26 | scripts/streamlit/depth2img.py:65 | change "torch.autocast("cuda"):" to "torch.autocast("mlu"): " |
| 27 | scripts/streamlit/inpainting.py:3 | add "import torch_mlu" |
| 28 | scripts/streamlit/inpainting.py:34 | change "device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")" to "device = torch.device("mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 29 | scripts/streamlit/inpainting.py:70 | change "device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")" to "device = torch.device("mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 30 | scripts/streamlit/inpainting.py:83 | change "torch.autocast("cuda"):" to "torch.autocast("mlu"): " |
| 31 | scripts/streamlit/stableunclip.py:3 | add "import torch_mlu" |
| 32 | scripts/streamlit/stableunclip.py:72 | change "init_image = load_img(key=key).cuda()" to "init_image = load_img(key=key).mlu() " |
| 33 | scripts/streamlit/stableunclip.py:109 | change "with precision_scope("cuda"):" to "with precision_scope("mlu"): " |
| 34 | scripts/streamlit/stableunclip.py:266 | change "model.cuda()" to "model.mlu() " |
| 35 | scripts/streamlit/superresolution.py:2 | add "import torch_mlu" |
| 36 | scripts/streamlit/superresolution.py:26 | change "device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")" to "device = torch.device("mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 37 | scripts/streamlit/superresolution.py:56 | change "device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")" to "device = torch.device("mlu") if torch.mlu.is_available() else torch.device("cpu") " |
| 38 | scripts/streamlit/superresolution.py:68 | change "torch.autocast("cuda"):" to "torch.autocast("mlu"): " |
| 39 | ldm/util.py:3 | add "import torch_mlu" |
| 40 | ldm/util.py:13 | change "with torch.cuda.amp.autocast(enabled=True," to "with torch.mlu.amp.autocast(enabled=True, " |
| 41 | ldm/modules/ema.py:1 | add "import torch_mlu" |
| 42 | ldm/modules/attention.py:3 | add "import torch_mlu" |
| 43 | ldm/modules/attention.py:175 | change "with torch.autocast(enabled=False, device_type = 'cuda'):" to "with torch.autocast(enabled=False, device_type = 'mlu'): " |
| 44 | ldm/modules/karlo/diffusers_pipeline.py:18 | add "import torch_mlu" |
| 45 | ldm/modules/karlo/diffusers_pipeline.py:209 | change "device = torch.device(f"cuda:{gpu_id}")" to "device = torch.device(f"mlu:{gpu_id}") " |
| 46 | ldm/modules/karlo/kakao/template.py:8 | add "import torch_mlu" |
| 47 | ldm/modules/karlo/kakao/template.py:81 | change "clip.cuda()" to "clip.mlu() " |
| 48 | ldm/modules/karlo/kakao/template.py:108 | change "prior.cuda()" to "prior.mlu() " |
| 49 | ldm/modules/karlo/kakao/template.py:124 | change "decoder.cuda()" to "decoder.mlu() " |
| 50 | ldm/modules/karlo/kakao/template.py:137 | change "sr.cuda()" to "sr.mlu() " |
| 51 | ldm/modules/karlo/kakao/sampler.py:10 | add "import torch_mlu" |
| 52 | ldm/modules/karlo/kakao/sampler.py:63 | change "prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device="cuda")" to "prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device="mlu") " |
| 53 | ldm/modules/karlo/kakao/sampler.py:66 | change "decoder_cf_scales_batch = torch.tensor(decoder_cf_scales_batch, device="cuda")" to "decoder_cf_scales_batch = torch.tensor(decoder_cf_scales_batch, device="mlu") " |
| 54 | ldm/modules/karlo/kakao/sampler.py:82 | change "tok, mask = tok.to(device="cuda"), mask.to(device="cuda")" to "tok, mask = tok.to(device="mlu"), mask.to(device="mlu") " |
| 55 | ldm/modules/karlo/kakao/sampler.py:102 | change "with torch.no_grad(), torch.cuda.amp.autocast():" to "with torch.no_grad(), torch.mlu.amp.autocast(): " |
| 56 | ldm/modules/karlo/kakao/sampler.py:210 | change "prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device="cuda")" to "prior_cf_scales_batch = torch.tensor(prior_cf_scales_batch, device="mlu") " |
| 57 | ldm/modules/karlo/kakao/sampler.py:213 | change "decoder_cf_scales_batch = torch.tensor(decoder_cf_scales_batch, device="cuda")" to "decoder_cf_scales_batch = torch.tensor(decoder_cf_scales_batch, device="mlu") " |
| 58 | ldm/modules/karlo/kakao/sampler.py:229 | change "tok, mask = tok.to(device="cuda"), mask.to(device="cuda")" to "tok, mask = tok.to(device="mlu"), mask.to(device="mlu") " |
| 59 | ldm/modules/karlo/kakao/sampler.py:249 | change "with torch.no_grad(), torch.cuda.amp.autocast():" to "with torch.no_grad(), torch.mlu.amp.autocast(): " |
| 60 | ldm/modules/karlo/kakao/models/sr_64_256.py:7 | add "import torch_mlu" |
| 61 | ldm/modules/karlo/kakao/models/prior_model.py:7 | add "import torch_mlu" |
| 62 | ldm/modules/karlo/kakao/models/decoder_model.py:7 | add "import torch_mlu" |
| 63 | ldm/modules/karlo/kakao/models/clip.py:10 | add "import torch_mlu" |
| 64 | ldm/modules/midas/utils.py:6 | add "import torch_mlu" |
| 65 | ldm/modules/midas/api.py:4 | add "import torch_mlu" |
| 66 | ldm/modules/midas/midas/base_model.py:1 | add "import torch_mlu" |
| 67 | ldm/modules/midas/midas/blocks.py:1 | add "import torch_mlu" |
| 68 | ldm/modules/midas/midas/vit.py:1 | add "import torch_mlu" |
| 69 | ldm/modules/midas/midas/dpt_depth.py:1 | add "import torch_mlu" |
| 70 | ldm/modules/midas/midas/midas_net_custom.py:5 | add "import torch_mlu" |
| 71 | ldm/modules/midas/midas/midas_net.py:5 | add "import torch_mlu" |
| 72 | ldm/modules/image_degradation/utils_image.py:5 | add "import torch_mlu" |
| 73 | ldm/modules/image_degradation/bsrgan.py:15 | add "import torch_mlu" |
| 74 | ldm/modules/image_degradation/bsrgan_light.py:4 | add "import torch_mlu" |
| 75 | ldm/modules/distributions/distributions.py:1 | add "import torch_mlu" |
| 76 | ldm/modules/encoders/modules.py:1 | add "import torch_mlu" |
| 77 | ldm/modules/encoders/modules.py:46 | change "def get_unconditional_conditioning(self, bs, device="cuda"):" to "def get_unconditional_conditioning(self, bs, device="mlu"): " |
| 78 | ldm/modules/encoders/modules.py:62 | change "def __init__(self, version="google/t5-v1_1-large", device="cuda", max_length=77," to "def __init__(self, version="google/t5-v1_1-large", device="mlu", max_length=77, " |
| 79 | ldm/modules/encoders/modules.py:99 | change "def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77," to "def __init__(self, version="openai/clip-vit-large-patch14", device="mlu", max_length=77, " |
| 80 | ldm/modules/encoders/modules.py:143 | change "device='cuda' if torch.cuda.is_available() else 'cpu'," to "device='mlu' if torch.mlu.is_available() else 'cpu', " |
| 81 | ldm/modules/encoders/modules.py:186 | change "def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77," to "def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="mlu", max_length=77, " |
| 82 | ldm/modules/encoders/modules.py:244 | change "def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="cuda", max_length=77," to "def __init__(self, arch="ViT-H-14", version="laion2b_s32b_b79k", device="mlu", max_length=77, " |
| 83 | ldm/modules/encoders/modules.py:299 | change "def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="cuda"," to "def __init__(self, clip_version="openai/clip-vit-large-patch14", t5_version="google/t5-v1_1-xl", device="mlu", " |
| 84 | ldm/modules/diffusionmodules/upscaling.py:1 | add "import torch_mlu" |
| 85 | ldm/modules/diffusionmodules/upscaling.py:68 | change "def __init__(self, noise_schedule_config, max_noise_level=1000, to_cuda=False):" to "def __init__(self, noise_schedule_config, max_noise_level=1000, to_mlu=False): " |
| 86 | ldm/modules/diffusionmodules/model.py:3 | add "import torch_mlu" |
| 87 | ldm/modules/diffusionmodules/util.py:13 | add "import torch_mlu" |
| 88 | ldm/modules/diffusionmodules/util.py:143 | change "torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):" to "torch.mlu.amp.autocast(**ctx.gpu_autocast_kwargs): " |
| 89 | ldm/data/util.py:1 | add "import torch_mlu" |
| 90 | ldm/models/autoencoder.py:1 | add "import torch_mlu" |
| 91 | ldm/models/diffusion/plms.py:3 | add "import torch_mlu" |
| 92 | ldm/models/diffusion/plms.py:13 | change "def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):" to "def __init__(self, model, schedule="linear", device=torch.device("mlu"), **kwargs): " |
| 93 | ldm/models/diffusion/sampling_util.py:1 | add "import torch_mlu" |
| 94 | ldm/models/diffusion/ddim.py:3 | add "import torch_mlu" |
| 95 | ldm/models/diffusion/ddim.py:11 | change "def __init__(self, model, schedule="linear", device=torch.device("cuda"), **kwargs):" to "def __init__(self, model, schedule="linear", device=torch.device("mlu"), **kwargs): " |
| 96 | ldm/models/diffusion/ddpm.py:9 | add "import torch_mlu" |
| 97 | ldm/models/diffusion/dpm_solver/sampler.py:2 | add "import torch_mlu" |
| 98 | ldm/models/diffusion/dpm_solver/sampler.py:13 | change "def __init__(self, model, device=torch.device("cuda"), **kwargs):" to "def __init__(self, model, device=torch.device("mlu"), **kwargs): " |
| 99 | ldm/models/diffusion/dpm_solver/dpm_solver.py:1 | add "import torch_mlu" |
