import torch
import time

def test_device(device_name):
    print(f"\n{'='*50}")
    print(f"Testing on {device_name.upper()}")
    print(f"{'='*50}")
    
    device = torch.device(device_name)
    print(f"Device: {device}")
    
    # Create a simple neural network
    model = torch.nn.Sequential(
        torch.nn.Linear(1000, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10)
    )
    model = model.to(device)
    
    # Create random input
    batch_size = 32
    num_iterations = 100
    input_tensor = torch.randn(batch_size, 1000, device=device)
    
    # Warm up
    with torch.no_grad():
        model(input_tensor)
    
    # Benchmark
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            output = model(input_tensor)
    end = time.time()
    
    elapsed = end - start
    throughput = (num_iterations * batch_size) / elapsed
    
    print(f"Time for {num_iterations} iterations: {elapsed:.4f}s")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    print("PyTorch CUDA Test")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Test CPU
    test_device("cpu")
    
    # Test GPU if available
    if torch.cuda.is_available():
        test_device("cuda")
    else:
        print("\nGPU not available - CUDA test skipped")
