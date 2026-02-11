import torch


def main():
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA:   {torch.cuda.is_available()} ({torch.version.cuda})")
    if torch.cuda.is_available():
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
        x = torch.randn(256, 256, device="cuda")
        print(f"Tensor on GPU: {x.device}, shape={x.shape}")


if __name__ == "__main__":
    main()
