import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class TrafficDataset(Dataset):
    """Synthetic traffic dataset for initial development"""
    
    def __init__(self, num_samples=1000, seq_length=24):
        """
        Generate synthetic traffic patterns
        Features: [north, south, east, west, pedestrians]
        """
        self.seq_length = seq_length
        self.num_samples = num_samples
        
        # Generate realistic traffic patterns
        np.random.seed(42)
        
        # Base patterns for different times of day
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            # Random time of day effect
            time_of_day = np.random.choice(['morning', 'afternoon', 'evening', 'night'])
            
            # Generate sequence
            sequence = []
            for t in range(seq_length):
                # Base traffic based on time
                if time_of_day == 'morning':
                    base_ns = np.random.poisson(15)  # North-South heavy
                    base_ew = np.random.poisson(8)   # East-West light
                    peds = np.random.poisson(5)
                elif time_of_day == 'evening':
                    base_ns = np.random.poisson(12)
                    base_ew = np.random.poisson(10)
                    peds = np.random.poisson(7)
                elif time_of_day == 'night':
                    base_ns = np.random.poisson(3)
                    base_ew = np.random.poisson(2)
                    peds = np.random.poisson(1)
                else:  # afternoon
                    base_ns = np.random.poisson(8)
                    base_ew = np.random.poisson(8)
                    peds = np.random.poisson(4)
                
                # Add some noise and trends
                trend = 0.1 * t if t < 12 else 0.1 * (24 - t)
                
                north = max(0, int(base_ns + trend + np.random.normal(0, 2)))
                south = max(0, int(base_ns + trend + np.random.normal(0, 2)))
                east = max(0, int(base_ew + trend + np.random.normal(0, 1.5)))
                west = max(0, int(base_ew + trend + np.random.normal(0, 1.5)))
                pedestrians = max(0, int(peds + np.random.normal(0, 1)))
                
                sequence.append([north, south, east, west, pedestrians])
            
            self.data.append(sequence)
            
            # Label: predict the next time step after the sequence
            next_step = [
                max(0, int(base_ns + np.random.normal(0, 2))),
                max(0, int(base_ns + np.random.normal(0, 2))),
                max(0, int(base_ew + np.random.normal(0, 1.5))),
                max(0, int(base_ew + np.random.normal(0, 1.5))),
                max(0, int(peds + np.random.normal(0, 1)))
            ]
            self.labels.append(next_step)
        
        self.data = torch.FloatTensor(self.data)  # Shape: (num_samples, seq_length, 5)
        self.labels = torch.FloatTensor(self.labels)  # Shape: (num_samples, 5)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class TrafficLSTMPredictor(nn.Module):
    """LSTM-based traffic prediction model"""
    
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, output_size=5):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state for prediction
        last_hidden = hidden[-1]  # Shape: (batch_size, hidden_size)
        
        return self.fc(last_hidden)

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    """Training function"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses

def main():
    """Main execution function"""
    print("ATLAS Traffic Prediction Model")
    print("=" * 50)
    
    # 1. Create dataset
    print("1. Creating synthetic traffic dataset...")
    full_dataset = TrafficDataset(num_samples=2000, seq_length=24)
    
    # Split into train/validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # 2. Create model
    print("\n2. Creating LSTM model...")
    model = TrafficLSTMPredictor(
        input_size=5,
        hidden_size=64,
        num_layers=2,
        output_size=5
    )
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Train model
    print("\n3. Training model...")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader,
        epochs=10,
        lr=0.001
    )
    
    # 4. Make predictions
    print("\n4. Making sample predictions...")
    model.eval()
    with torch.no_grad():
        # Get a batch for demonstration
        sample_batch, sample_target = next(iter(val_loader))
        predictions = model(sample_batch)
        
        print(f"   Sample batch shape: {sample_batch.shape}")
        print(f"   Predictions shape: {predictions.shape}")
        
        # Show first prediction
        print(f"\n   First sample prediction:")
        print(f"   Actual: {sample_target[0].numpy()}")
        print(f"   Predicted: {predictions[0].numpy()}")
        print(f"   Error: {np.abs(sample_target[0].numpy() - predictions[0].numpy())}")
    
    # 5. Plot results
    print("\n5. Plotting training results...")
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('ATLAS Traffic Prediction Model Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('docs/figures/training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 50)
    print("✅ Model training complete!")
    print(f"   Final training loss: {train_losses[-1]:.4f}")
    print(f"   Final validation loss: {val_losses[-1]:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), 'models/traffic_predictor_v1.pth')
    print("   Model saved to 'models/traffic_predictor_v1.pth'")

if __name__ == "__main__":
    main()