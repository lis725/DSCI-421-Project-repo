import torch
import torch.nn as nn


def maybe_wrap_model(model, device, multi_gpu=False):
    model = model.to(device)
    if device == "cuda":
        n_gpu = torch.cuda.device_count()
        names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
        print(f"[GPU] visible_cuda_devices={n_gpu} names={names}")
        if multi_gpu and n_gpu > 1:
            print(f"[MultiGPU] enabled DataParallel across {n_gpu} GPUs")
            model = nn.DataParallel(model)
        elif multi_gpu:
            print("[MultiGPU] requested, but fewer than 2 GPUs are visible; using single GPU")
    return model


def state_dict_for_save(model):
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def clean_state_dict(state_dict):
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict
