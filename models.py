import torch
import torch.nn as nn
import torch.nn.functional as F
from arguments import *
import torch.nn.init as init

NUM_STORES = 55
NUM_ITEMS = 4100
args = solicit_args()

class NN1(nn.Module):
    def __init__(self, input_size, hidden_size=256, storeEmb_size=60, itemEmb_size=300, monthEmb_size=10, dowEmb_size=10):
        super(NN1, self).__init__()
        self.store_embeddings = nn.Embedding(NUM_STORES, storeEmb_size)
        self.item_embeddings = nn.Embedding(NUM_ITEMS, itemEmb_size)
        self.month_embeddings = nn.Embedding(13, monthEmb_size)
        self.dow_embeddings = nn.Embedding(8, dowEmb_size)
        self.day_embeddings = nn.Embedding(32, 2)
        self.input_embeddings = nn.Linear(input_size - 5,
                                          storeEmb_size + itemEmb_size + monthEmb_size + dowEmb_size + 2)
        init.normal(self.input_embeddings.weight, mean=0, std=0.001)

        self.input_size = storeEmb_size + itemEmb_size + monthEmb_size + dowEmb_size + 2
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, hidden_size)
        self.fc7 = nn.Linear(hidden_size, 1)

        self.layers = [self.fc2, self.fc3, self.fc4, self.fc5, self.fc6]

    def forward(self, varList):
        [batchPerishable, batchStores, batchItems, batchMonth, batchDay, batchDow, batchPro, batchYear,
         batchHis] = varList
        storeEmbs = self.store_embeddings(batchStores)
        itemEmbs = self.item_embeddings(batchItems)
        monthEmbs = self.month_embeddings(batchMonth)
        dowEmbs = self.dow_embeddings(batchDow)
        dayEmbs = self.day_embeddings(batchDay)
        inputs = torch.cat((batchPerishable, batchPro, batchYear, batchHis), dim=1)
        inputEmbs = self.input_embeddings(inputs)

        inputs = torch.cat((storeEmbs, itemEmbs, monthEmbs, dowEmbs, dayEmbs), dim=1) + inputEmbs

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
    def __init__(self, input_size, hidden_size=256, storeEmb_size=60, itemEmb_size=300, monthEmb_size=10, dowEmb_size=10):
        super(simpleNN, self).__init__()

        self.store_embeddings = nn.Embedding(NUM_STORES, storeEmb_size)
        self.item_embeddings = nn.Embedding(NUM_ITEMS, itemEmb_size)
        self.month_embeddings = nn.Embedding(13, monthEmb_size)
        self.dow_embeddings = nn.Embedding(8, dowEmb_size)
        self.day_embeddings = nn.Embedding(32, 2)
        self.input_embeddings = nn.Linear(input_size - 5, storeEmb_size + itemEmb_size + monthEmb_size + dowEmb_size + 2)
        init.normal(self.input_embeddings.weight, mean=0, std=0.001)



        self.input_size = storeEmb_size + itemEmb_size + monthEmb_size + dowEmb_size + 2
        # self.input_size = storeEmb_size + itemEmb_size

        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 1)

    def forward(self, varList):
        [batchPerishable, batchStores, batchItems, batchMonth, batchDay, batchDow, batchPro, batchYear, batchHis] = varList
        storeEmbs = self.store_embeddings(batchStores)
        itemEmbs = self.item_embeddings(batchItems)
        monthEmbs = self.month_embeddings(batchMonth)
        dowEmbs = self.dow_embeddings(batchDow)
        dayEmbs = self.day_embeddings(batchDay)
        inputs = torch.cat((batchPerishable, batchPro, batchYear, batchHis), dim=1)
        inputEmbs = self.input_embeddings(inputs)

        inputs = torch.cat((storeEmbs, itemEmbs, monthEmbs, dowEmbs, dayEmbs), dim=1) + inputEmbs
        # inputs = torch.cat((storeEmbs, itemEmbs), dim=1)  # + inputEmbs

        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, hidden_size):
        super(BasicBlock, self).__init__()
        self.l1 = nn.Linear(hidden_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        residual = x

        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        out += x
        return out


class resNext(object):
    def __init__(self, input_size, hidden_size=256, storeEmb_size=60, itemEmb_size=300, monthEmb_size=10, dowEmb_size=10):
        super(resNext, self).__init__()
        self.store_embeddings = nn.Embedding(NUM_STORES, storeEmb_size)
        self.item_embeddings = nn.Embedding(NUM_ITEMS, itemEmb_size)
        self.month_embeddings = nn.Embedding(13, monthEmb_size)
        self.dow_embeddings = nn.Embedding(8, dowEmb_size)
        self.day_embeddings = nn.Embedding(32, 100)

        self.input_size = input_size - 5 + storeEmb_size + itemEmb_size + monthEmb_size + dowEmb_size + 100
        self.fc = nn.Linear(self.input_size, hidden_size)

        layers = []
        for i in range(args.num_blocks):
            layers.append(BasicBlock(hidden_size))
            mid = nn.Linear(hidden_size, hidden_size/2)
            layers.append(mid)
            hidden_size = hidden_size/2

        layers.append(nn.Linear(hidden_size, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, varList):
        [batchPerishable, batchStores, batchItems, batchMonth, batchDay, batchDow, batchPro, batchYear, batchHis] = varList
        storeEmbs = self.store_embeddings(batchStores)
        itemEmbs = self.item_embeddings(batchItems)
        monthEmbs = self.month_embeddings(batchMonth)
        dowEmbs = self.dow_embeddings(batchDow)
        dayEmbs = self.day_embeddings(batchDay)
        inputs = (batchPerishable, batchPro, batchYear, batchHis)

        inputs = torch.cat((storeEmbs, itemEmbs, monthEmbs, dowEmbs, dayEmbs) + inputs, dim=1)

        out = F.relu(self.fc(inputs))
        out = self.layers(out)
        return out



