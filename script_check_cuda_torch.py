import torch

def main():
    print('torch', torch.__version__)
    print('cuda_available', torch.cuda.is_available())
    print('torch_cuda', torch.version.cuda)
    print('device_count', torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        print('device_name', torch.cuda.get_device_name(0))
    else:
        print('device_name', None)

if __name__ == '__main__':
    main()
