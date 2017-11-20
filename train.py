import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torch.autograd import Variable
import numpy as np
from data_analysis import *

train_data = pd.read_csv('data/Jan17Jul_split/train_data.csv', nrows=5000)
dev_data = pd.read_csv('data/Jan17Jul_split/dev_data.csv', nrows=5000)
train_labels = pd.read_csv('data/Jan17Jul_split/train_labels.csv', nrows=5000)
dev_labels = pd.read_csv('data/Jan17Jul_split/dev_labels.csv', nrows=5000)


train_data = train_data.drop('id', 1)
dev_data = dev_data.drop('id', 1)

print 'Data readed in!'

columns = list(train_data)

train_data = train_data.as_matrix()
dev_data = dev_data.as_matrix()
train_labels = train_labels.as_matrix()
dev_labels = dev_labels.as_matrix()


num_examples = len(train_data)
num_dev = len(dev_data)

NUM_STORES = 55
NUM_ITEMS = 4100

class simpleNN(nn.Module):
    def __init__(self, input_size, hidden_size=256, storeEmb_size=30, itemEmb_size=100):
        super(simpleNN, self).__init__()

        self.store_embeddings = nn.Embedding(NUM_STORES, storeEmb_size)
        self.item_embeddings = nn.Embedding(NUM_ITEMS, itemEmb_size)

        self.input_size = input_size - 2 + storeEmb_size + itemEmb_size
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, stores, items, input):
        storeEmbs = self.store_embeddings(stores)
        itemEmbs = self.item_embeddings(items)
        inputs = torch.cat((storeEmbs, itemEmbs, input), dim=1)

        out = F.relu(self.fc1(inputs))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def solicit_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', help='iterations', type=int, default=2000)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=512)
    parser.add_argument('--dev_batch_size', help='dev_batch_size', type=int, default=1024)
    parser.add_argument('--iter2report', help='iterations to report', type=int, default=500)
    parser.add_argument('--version', help='version', type=str, default='test')
    return parser.parse_args()


def generate_dev_batch(dev_data, dev_labels):
    seq = np.random.choice(num_dev, args.dev_batch_size)
    batchD = dev_data[seq, :]
    batchStores = Variable(torch.LongTensor(batchD[:, 0].astype(np.int64)))
    batchItems = Variable(torch.LongTensor(batchD[:, 1].astype(np.int64)))
    batchD = Variable(torch.Tensor(batchD[:, 2:]))
    batchL = Variable(torch.Tensor(dev_labels[seq]))
    return batchStores, batchItems, batchD, batchL

def validation(model, dev_data, dev_labels, loss_function):
    model.eval()
    batchStores, batchItems, batchD, batchL = generate_dev_batch(dev_data, dev_labels)

    predictions = model(batchStores, batchItems, batchD)
    loss = loss_function(predictions, batchL)

    return loss


args = solicit_args()

losses = []
steps = []
dev_losses = []

loss_function = nn.MSELoss()
model = simpleNN(len(columns))
optimizer = optim.SGD(model.parameters(), lr=0.01)

for step in range(args.iterations):
    seq = np.random.choice(num_examples, args.batch_size)
    batchD = train_data[seq, :]
    batchStores = Variable(torch.LongTensor(batchD[:, 0].astype(np.int64)))
    batchItems = Variable(torch.LongTensor(batchD[:, 1].astype(np.int64)))
    batchD = Variable(torch.Tensor(batchD[:, 2:]))
    batchL = Variable(torch.Tensor(train_labels[seq]))

    model.zero_grad()

    predictions = model(batchStores, batchItems, batchD)
    loss = loss_function(predictions, batchL)

    if step % args.iter2report == 0:
        print 'Current step = ', step
        print 'Current loss = ', loss
        losses.append(loss.data[0])
        steps.append(step)

        dev_loss = validation(model, dev_data, dev_labels, loss_function)

        print 'Validation loss = ', dev_loss
        dev_losses.append(dev_loss.data[0])

    loss.backward()
    optimizer.step()



df = pd.DataFrame(data={'steps':steps, 'train_losses':losses, 'validation_losses':dev_losses})
df.to_csv('records/' + args.version + '_error.csv', index=False)





