import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_STORES = 55
NUM_ITEMS = 4100

class simpleNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, storeEmb_size=100, itemEmb_size=200):
        super(simpleNN, self).__init__()

        self.store_embeddings = nn.Embedding(NUM_STORES, storeEmb_size)
        self.item_embeddings = nn.Embedding(NUM_ITEMS, itemEmb_size)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.input_size = input_size - 2 + storeEmb_size + itemEmb_size
        self.fc1 = nn.Linear(self.input_size, 2 * hidden_size)
        self.fc2 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.fc4 = nn.Linear(2 * hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, 1)

        self.layers = [self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]

    def forward(self, stores, items, input):
        storeEmbs = self.store_embeddings(stores)
        itemEmbs = self.item_embeddings(items)
        inputs = torch.cat((storeEmbs, itemEmbs, input), dim=1)

        out = F.relu(self.fc1(inputs))
        for layer in self.layers:
            out = F.relu(layer(out))
        out = self.fc7(out)
        return out





