import torch


def resolve_device(name: str) -> torch.device:
    """Resolve a device string to a torch.device.

    `"auto"` picks CUDA if available, else CPU. Other values pass through to
    torch.device() directly ("cpu", "cuda", "cuda:1", etc).
    """
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
