import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence
import torch.optim as optim
#from torchvision.models.video.resnet import model_urls
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

#from codes.LSTMl3 import train_dataset

#using M3 chip

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using:{device}")


writer = SummaryWriter(log_dir="runs/lstm_level4")

class Level4Dataset(Dataset):
    def __init__(self, root_dir):
        self.sequences = []
        self.labels = []

        for label_dir, label in [('1',0),('2',1)]:
            folder = os.path.join(root_dir, label_dir)
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                seq = np.loadtxt(path)
                self.sequences.append(torch.tensor(seq,dtype=torch.float32))
                self.labels.append(label)

    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = Level4Dataset("../dataset/level4/reference")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim = 64, hidden_dim = 32, num_classes = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
            #packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first = True, enforce_sorted=False)
            out, (hn,_)= self.lstm(x)
            out = self.fc(hn[-1])
            return out


model = LSTMClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()


#training
for epoch in range(50):
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        out = model(X_batch)
        loss = loss_fn(out, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    writer.add_scalar("Loss/train", loss.item(), epoch)
    print(f"Epoch {epoch+1} :loss: {loss.item():.4f}")

os.makedirs("../model", exist_ok=True)
torch.save(model.state_dict(), "../model/lstm_level4.pth")
print("✅ successfully saved as lstm_level4.pth")

#testing
model = LSTMClassifier().to(device)
model.load_state_dict(torch.load("../model/lstm_level4.pth"))
model.eval()
#fig, ax = plt.subplots(figsize=(8,4))

test_dir = "../dataset/level4/test"

for file in sorted(os.listdir(test_dir)):
    seq = np.loadtxt(os.path.join(test_dir,file))
    X = torch.tensor(seq,dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(X)
        pred = torch.argmax(output, dim = 1).item()

    #ax.plot(seq[:,0],label = f"{file} -> class {pred}" )

    print(f"{file}: class {pred}")
