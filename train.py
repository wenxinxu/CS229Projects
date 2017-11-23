from models import *
import torch.optim as optim
import argparse
from torch.autograd import Variable
import numpy as np
from data_analysis import *
import time
from utils import *

use_cuda = torch.cuda.is_available()
DECAY = [0.5, 0.75]


def solicit_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', help='iterations', type=int, default=80000)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=512)
    parser.add_argument('--dev_batch_size', help='dev_batch_size', type=int, default=1024)
    parser.add_argument('--iter2report', help='iterations to report', type=int, default=1000)
    parser.add_argument('--version', help='version', type=str, default='v1')
    parser.add_argument('--init_lr', help='initialized learning rate', type=float, default=0.01)
    return parser.parse_args()

args = solicit_args()


def load_train_dev():
    train_data = pd.read_csv('data/Jan17Jul_split/train_data.csv')
    dev_data = pd.read_csv('data/Jan17Jul_split/dev_data.csv')
    train_labels = pd.read_csv('data/Jan17Jul_split/train_labels.csv')
    dev_labels = pd.read_csv('data/Jan17Jul_split/dev_labels.csv')


    # train_data = train_data.drop('id', 1)
    # dev_data = dev_data.drop('id', 1)

    print 'Data readed in!'

    train_data = train_data.as_matrix()
    dev_data = dev_data.as_matrix()
    train_labels = train_labels.as_matrix()
    dev_labels = dev_labels.as_matrix()
    return train_data, dev_data, train_labels, dev_labels




def weighted_MSE(predictions, targets, weights):
    return torch.sum(torch.mul(weights, ((predictions - targets) ** 2))) / torch.sum(weights)


def generate_dev_batch(dev_data, dev_labels, num_dev):
    seq = np.random.choice(num_dev, args.dev_batch_size)
    batchD = dev_data[seq, :]
    batchPerishable = Variable(torch.FloatTensor(batchD[:, 0].astype(np.int64))).view((args.dev_batch_size, 1))
    batchStores = Variable(torch.LongTensor(batchD[:, 1].astype(np.int64)))
    batchItems = Variable(torch.LongTensor(batchD[:, 2].astype(np.int64)))
    batchD = Variable(torch.Tensor(batchD[:, 3:]))
    batchL = Variable(torch.Tensor(dev_labels[seq]))

    batchD = batchD.cuda() if use_cuda else batchD
    batchStores = batchStores.cuda() if use_cuda else batchStores
    batchItems = batchItems.cuda() if use_cuda else batchItems
    batchL = batchL.cuda() if use_cuda else batchL


    return batchPerishable, batchStores, batchItems, batchD, batchL

def validation(model, dev_data, dev_labels, loss_function, num_dev):
    model.eval()
    batchPerishable, batchStores, batchItems, batchD, batchL = generate_dev_batch(dev_data, dev_labels, num_dev)

    predictions = model(batchStores, batchItems, batchD)
    loss = loss_function(predictions, batchL, batchPerishable)
    # loss = loss_function(predictions, batchL)

    return loss


def train(train_data, dev_data, train_labels, dev_labels, model):


    num_examples = len(train_data)
    num_dev = len(dev_data)

    losses = []
    steps = []
    dev_losses = []

    loss_function = weighted_MSE
    # loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.init_lr * 0.01)
    start = time.time()

    for step in range(args.iterations):
        seq = np.random.choice(num_examples, args.batch_size)
        batchD = train_data[seq, :]
        batchPerishable = Variable(torch.FloatTensor(batchD[:, 0].astype(np.int64))).view((args.batch_size, 1))
        batchStores = Variable(torch.LongTensor(batchD[:, 1].astype(np.int64)))
        batchItems = Variable(torch.LongTensor(batchD[:, 2].astype(np.int64)))
        batchD = Variable(torch.Tensor(batchD[:, 3:]))
        batchL = Variable(torch.Tensor(train_labels[seq]))

        batchD = batchD.cuda() if use_cuda else batchD
        batchStores = batchStores.cuda() if use_cuda else batchStores
        batchItems = batchItems.cuda() if use_cuda else batchItems
        batchL = batchL.cuda() if use_cuda else batchL

        model.zero_grad()

        predictions = model(batchStores, batchItems, batchD)
        loss = loss_function(predictions, batchL, batchPerishable)
        # loss = loss_function(predictions, batchL)

        if step % args.iter2report == 0:
            print 'Current step = ', step
            print 'Current loss = ', loss
            print '%d%% complete %s' % ((step * 1.0/args.iterations * 100) , (timeSince(start,
                                                                                     step * 1.0/args.iterations)))
            losses.append(loss.data[0])
            steps.append(step)

            dev_loss = validation(model, dev_data, dev_labels, loss_function, num_dev)

            print 'Validation loss = ', dev_loss
            dev_losses.append(dev_loss.data[0])

        if step == args.iterations * DECAY[0] or args.iterations * DECAY[1]:
            lr = args.init_lr * 0.1 ** (1 + step / DECAY[1])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        loss.backward()
        optimizer.step()

    df = pd.DataFrame(data={'steps':steps, 'train_losses':losses, 'validation_losses':dev_losses})
    df.to_csv('records/' + args.version + '_error.csv', index=False)


if __name__ == "__main__":
    train_data, dev_data, train_labels, dev_labels = load_train_dev()
    model = simpleNN(train_data.shape[1] - 1)
    model = model.cuda() if use_cuda else model
    train(train_data, dev_data, train_labels, dev_labels, model)
    torch.save(model, 'records/' + args.version + '.pt')
    print 'Model saved to ' + 'records/' + args.version + '.pt'


