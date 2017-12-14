from models import *
import torch.optim as optim
from torch.autograd import Variable
from data_analysis import *
import time
from utils import *

use_cuda = torch.cuda.is_available()
HISTORY = 15 # How many days to look back
DECAY = [0.5, 0.75]
cols = []
columns = ['perishable', 'store_nbr', 'item_nbr', 'onpromotion', 'month', 'dow', 'day', 'year',
           'us0', 'us1', 'us2', 'us3', 'us4', 'us5', 'us6', 'us7', 'us8', 'us9', 'us10', 'us11', 'us12',
           'us13', 'us14']


def load_train_dev():
    if args.debug:
        train_data = pd.read_csv('data/april/df_train.csv', usecols=columns, nrows=2000)
        dev_data = pd.read_csv('data/april/df_dev.csv', usecols=columns, nrows=2000)
    else:
        train_data = pd.read_csv('data/april/df_train.csv', usecols=columns)
        dev_data = pd.read_csv('data/april/df_dev.csv', usecols=columns)
    train_labels = np.load('data/april/train_labels.npy')
    dev_labels = np.load('data/april/dev_labels.npy')

    print 'Data readed in!'

    train_data = train_data[columns].as_matrix()
    dev_data = dev_data[columns].as_matrix()
    return train_data, dev_data, train_labels, dev_labels


def weighted_MSE(predictions, targets, weights):
    # print torch.mul(weights, ((predictions - targets) ** 2))
    #
    # print predictions[0:5]
    # print 'predictions - targets = '
    # print targets[0:5]
    # print '------------'
    # print ((predictions - targets) ** 2)[0:5]
    targets = targets.view((-1, 1))
    diff = predictions - targets
    return torch.sum(torch.mul(weights, (torch.mul(diff, diff)))) / torch.sum(weights)


    # loss_function = nn.MSELoss()
    # return loss_function(predictions, targets)

def generate_dev_batch(dev_data, dev_labels, num_dev):
    seq = np.random.choice(num_dev, args.dev_batch_size)
    batchD = dev_data[seq, :]
    batchPerishable = Variable(torch.FloatTensor(batchD[:, 0].astype(np.int64))).view((args.dev_batch_size, 1))
    batchStores = Variable(torch.LongTensor(batchD[:, 1].astype(np.int64)))
    batchItems = Variable(torch.LongTensor(batchD[:, 2].astype(np.int64)))
    batchMonth = Variable(torch.LongTensor(batchD[:, 4].astype(np.int64)))
    batchDay = Variable(torch.LongTensor(batchD[:, 6].astype(np.int64)))
    batchDow = Variable(torch.LongTensor(batchD[:, 5].astype(np.int64)))

    batchPro = Variable(torch.Tensor(batchD[:, 3])).contiguous().view((args.dev_batch_size, 1))
    batchYear = Variable(torch.Tensor(batchD[:, 7])).contiguous().view((args.dev_batch_size, 1))
    batchHis = Variable(torch.Tensor(batchD[:, 8:]))
    batchL = Variable(torch.Tensor(dev_labels[seq]))

    varList = [batchPerishable, batchStores, batchItems, batchMonth, batchDay, batchDow, batchPro, batchYear, batchHis]
    for var in varList:
        if use_cuda:
            var = var.cuda()

    return varList, batchL

def validation(model, dev_data, dev_labels, loss_function, num_dev):
    model.eval()
    varList, batchL = generate_dev_batch(dev_data, dev_labels, num_dev)

    predictions = model(varList)
    loss = loss_function(predictions, batchL, varList[0])
    return loss


def train(train_data, dev_data, train_labels, dev_labels, model):

    num_examples = len(train_data)
    num_dev = len(dev_data)

    losses = []
    steps = []
    dev_losses = []

    loss_function = weighted_MSE
    # optimizer = optim.Adam(model.parameters(), lr=args.init_lr * 0.01)
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9)
    start = time.time()

    for step in range(args.iterations):
        seq = np.random.choice(num_examples, args.batch_size)
        batchD = train_data[seq, :]
        batchPerishable = Variable(torch.FloatTensor(batchD[:, 0].astype(np.int64))).view((args.batch_size, 1))
        batchStores = Variable(torch.LongTensor(batchD[:, 1].astype(np.int64)))
        batchItems = Variable(torch.LongTensor(batchD[:, 2].astype(np.int64)))
        batchMonth = Variable(torch.LongTensor(batchD[:, 4].astype(np.int64)))
        batchDay = Variable(torch.LongTensor(batchD[:, 6].astype(np.int64)))
        batchDow = Variable(torch.LongTensor(batchD[:, 5].astype(np.int64)))


        batchPro = Variable(torch.Tensor(batchD[:, 3])).contiguous().view((args.batch_size, 1))
        batchYear = Variable(torch.Tensor(batchD[:, 7])).contiguous().view((args.batch_size, 1))
        batchHis = Variable(torch.Tensor(batchD[:, 8:]))
        batchL = Variable(torch.Tensor(train_labels[seq]))

        varList = [batchPerishable, batchStores, batchItems, batchMonth, batchDay, batchDow, batchPro, batchYear, batchHis]
        for var in varList:
            if use_cuda:
                var = var.cuda()

        model.zero_grad()

        predictions = model(varList)
        # print 'Predcition = ', predictions


        loss = loss_function(predictions, batchL, batchPerishable)
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

        # if step == args.iter2report * 3:
        #     lr = 0.01
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        #     print 'Learning rate increased ... '

        if step == args.iterations * DECAY[0] or step == args.iterations * DECAY[1]:
            print 'Decay learning rate to ' + str(args.init_lr * 0.1 ** (1 + step / DECAY[1]))
            lr = args.init_lr * 0.1 ** (1 + step / DECAY[1])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        loss.backward()
        optimizer.step()

    df = pd.DataFrame(data={'steps':steps, 'train_losses':losses, 'validation_losses':dev_losses})
    df.to_csv('records/' + args.version + '_error.csv', index=False)



if __name__ == "__main__":
    train_data, dev_data, train_labels, dev_labels = load_train_dev()
    model = simpleNN(train_data.shape[1])
    # model = resNext(train_data.shape[1])
    model = model.cuda() if use_cuda else model
    train(train_data, dev_data, train_labels, dev_labels, model)
    torch.save(model, 'records/' + args.version + '.pt')
    print 'Model saved to ' + 'records/' + args.version + '.pt'


