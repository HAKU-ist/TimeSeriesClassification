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

#using M3 chip

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"using:{device}")


writer = SummaryWriter(log_dir="runs/lstm_level3")

class Level3Dataset(Dataset):
    def __init__(self, directory, label_map):
        self.sequences = []
        self.labels = []

        for file, label in label_map.items():
            path = os.path.join(directory, file)
            seq = np.loadtxt(path)
            self.sequences.append(torch.tensor(seq,dtype=torch.float32))
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_batch(batch):
    sequences,labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in sequences])
    return padded_seqs, torch.tensor(labels), lengths

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim = 3, hidden_dim = 64, num_classes = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, batch_first = True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x, lengths):
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first = True, enforce_sorted=False)
            packed_out, (hn,_)= self.lstm(packed)
            out = self.fc(hn[-1])
            return out

label_map = {
    "1.dat":0,
    "2.dat":1,
}
train_dataset = Level3Dataset("../dataset/level3/reference", label_map)
train_loader = DataLoader(train_dataset, batch_size = 2, collate_fn=collate_batch)


model = LSTMClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss()


#training
for epoch in range(50):
    for X_batch, y_batch, lengths in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        lengths = lengths.to(device)

        out = model(X_batch, lengths)
        loss = loss_fn(out, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"epoch: {epoch}, loss: {loss}")
    writer.add_scalar("Loss/train", loss.item(), epoch)
    print(f"Epoch {epoch+1} :loss: {loss.item():.4f}")

os.makedirs("../model", exist_ok=True)
torch.save(model.state_dict(), "../model/lstm_level3.pth")
print("✅ successfully saved as lstm_level3.pth")

#testing
model = LSTMClassifier().to(device)
model.load_state_dict(torch.load("../model/lstm_level3.pth"))
model.eval()
#fig, ax = plt.subplots(figsize=(8,4))

test_dir = "../dataset/level3/test"

for file in sorted(os.listdir(test_dir)):
    seq = np.loadtxt(os.path.join(test_dir,file))
    X = torch.tensor(seq,dtype=torch.float32).unsqueeze(0).to(device)
    X = X.to(device)
    lengths = torch.tensor([X.shape[1]]).to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        output = model(X, lengths)
        pred = torch.argmax(output, dim = 1).item()

    #ax.plot(seq[:,0],label = f"{file} -> class {pred}" )

    print(f"{file}: class {pred}")
