import torch
import torch.nn as nn
import torch.nn.functional as F
from arguments import *

NUM_STORES = 55
NUM_ITEMS = 4100
args = solicit_args()

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
    def __init__(self, input_size, hidden_size=256, storeEmb_size=60, itemEmb_size=300, monthEmb_size=10, dowEmb_size=10):
        super(simpleNN, self).__init__()

        self.store_embeddings = nn.Embedding(NUM_STORES, storeEmb_size)
        self.item_embeddings = nn.Embedding(NUM_ITEMS, itemEmb_size)
        self.month_embeddings = nn.Embedding(13, monthEmb_size)
        self.dow_embeddings = nn.Embedding(8, dowEmb_size)
        self.day_embeddings = nn.Embedding(32, 10)

        self.input_size = input_size - 5 + storeEmb_size + itemEmb_size + monthEmb_size + dowEmb_size + 10
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
        inputs = (batchPerishable, batchPro, batchYear, batchHis)

        inputs = torch.cat((storeEmbs, itemEmbs, monthEmbs, dowEmbs, dayEmbs) + inputs, dim=1)

        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))
        out = self.fc4(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input, ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

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

        return


