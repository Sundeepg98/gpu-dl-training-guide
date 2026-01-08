import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

class SimpleModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_on_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.001
    
    X = torch.randn(1000, 1, 28, 28)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SimpleModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    end_time = time.time()
    print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
    
    if torch.cuda.is_available():
        print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

def multi_gpu_example():
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = SimpleModel()
        model = nn.DataParallel(model)
        model = model.cuda()
    else:
        print("Multi-GPU not available")

def mixed_precision_example():
    try:
        from torch.cuda.amp import autocast, GradScaler
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = SimpleModel().to(device)
        optimizer = optim.Adam(model.parameters())
        scaler = GradScaler()
        
        data = torch.randn(32, 1, 28, 28).to(device)
        target = torch.randint(0, 10, (32,)).to(device)
        
        with autocast():
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print("Mixed precision training step completed")
    except ImportError:
        print("Mixed precision requires newer PyTorch version")

if __name__ == "__main__":
    train_on_gpu()
    print("\n" + "="*50 + "\n")
    multi_gpu_example()
    print("\n" + "="*50 + "\n")
    mixed_precision_example()