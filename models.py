import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_STORES = 55
NUM_ITEMS = 4100

class NN1(nn.Module):
    def __init__(self, input_size, hidden_size=512, storeEmb_size=200, itemEmb_size=300):
        super(NN1, self).__init__()

        self.store_embeddings = nn.Embedding(NUM_STORES, storeEmb_size)
        self.item_embeddings = nn.Embedding(NUM_ITEMS, itemEmb_size)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.input_size = input_size - 2 + storeEmb_size + itemEmb_size
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, 1)

        self.layers = [self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]

    def forward(self, stores, items, input):
        storeEmbs = self.store_embeddings(stores)
        itemEmbs = self.item_embeddings(items)
        inputs = torch.cat((storeEmbs, itemEmbs, input), dim=1)

        outs = []

        out = F.relu(self.fc1(inputs))
        outs.append(out)

        for layer in self.layers:
            out = F.relu(layer(out))
            if layer == self.fc4:
                out += outs[1]
            outs.append(out)
        out = self.fc7(out)
        return out



class simpleNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, storeEmb_size=100, itemEmb_size=200, monthEmb_size=10, dowEmb_size=5):
        super(simpleNN, self).__init__()

        self.store_embeddings = nn.Embedding(NUM_STORES, storeEmb_size)
        self.item_embeddings = nn.Embedding(NUM_ITEMS, itemEmb_size)
        self.month_embeddings = nn.Embedding(13, monthEmb_size)
        self.dow_embeddings = nn.Embedding(8, dowEmb_size)

        self.input_size = input_size - 4 + storeEmb_size + itemEmb_size + monthEmb_size + dowEmb_size
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, stores, items, month, dow, input):
        storeEmbs = self.store_embeddings(stores)
        itemEmbs = self.item_embeddings(items)
        monthEmbs = self.month_embeddings(month)
        dowEmbs = self.dow_embeddings(dow)
        inputs = torch.cat((storeEmbs, itemEmbs, monthEmbs, dowEmbs, input), dim=1)

        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out




