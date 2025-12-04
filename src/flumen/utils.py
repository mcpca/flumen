import torch


def print_gpu_info():
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"CUDA is available, {n_gpus} devices can be used.")
        current_dev = torch.cuda.current_device()

        for id in range(n_gpus):
            msg = f"Device {id}: {torch.cuda.get_device_name(id)}"

            if id == current_dev:
                msg += " [Current]"

            print(msg)


def pack_model_inputs(x0, t, u, delta):
    t = torch.Tensor(t)
    x0 = torch.Tensor(x0)
    u = torch.Tensor(u)

    if x0.ndim < 2:
        x0 = x0.unsqueeze(0)
        u = u.unsqueeze(0)

    skips = torch.floor(t / delta).int()
    tau = (t - delta * skips) / delta

    return x0, u, skips.squeeze(), tau
