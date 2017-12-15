from data_analysis import *
from torch.autograd import Variable
import torch
from arguments import *

use_cuda = torch.cuda.is_available()

test_df = pd.read_csv('data/april/test_data2.csv')
features = np.load('data/april/april_sales.npy')


grouped = test_df.groupby(['day'])
columns = ['perishable', 'store_nbr', 'item_nbr', 'onpromotion', 'month', 'dow', 'day', 'year',
           'us0', 'us1', 'us2', 'us3', 'us4', 'us5', 'us6', 'us7', 'us8', 'us9', 'us10', 'us11', 'us12',
           'us13', 'us14']
args = solicit_args()



def add_historical_features_test(df, features, days=15):
    '''
    This function help find the historical features for test dataset
    :param df:
    :param features:
    :return:
    '''
    histories = np.zeros((len(df), days))
    df.loc[:,'date'] = pd.to_datetime(df['date'])

    start = time.time()
    offset = -1
    for i, row in df.iterrows():
        if offset == -1:
            offset = i
        if i % 100000 == 0:
            print 'Processing the %i th row...' % i
            print 'Time cost = ', time.time() - start
            start = time.time()

        diff = (row['date'] - pd.to_datetime('2017-04-01')).days

        histories[i-offset, :] = features[row['store_nbr'], row['item_nbr'], diff - days:diff]
    cols = []

    for i in range(days):
        cols.append('us' + str(i))
    histories = pd.DataFrame(histories, columns=cols, index=df.index.get_values())

    df = pd.concat([df, histories], axis=1)
    print df.head()
    # print df.tail()
    return df


def generate_test_batch(dev_data, offset, end=False):
    if end:
        append = np.zeros((offset+args.dev_batch_size - len(dev_data), dev_data.shape[1]))
        dev_data = np.concatenate((dev_data, append), axis=0)

    seq = range(offset, offset+args.dev_batch_size)

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

    varList = [batchPerishable, batchStores, batchItems, batchMonth, batchDay, batchDow, batchPro, batchYear, batchHis]
    for var in varList:
        if use_cuda:
            var = var.cuda()

    return varList


def test(df, model):
    df = df[columns].as_matrix()
    num_iters = len(df) / args.dev_batch_size + 1

    offset = 0
    predictions = np.zeros(args.dev_batch_size * num_iters)

    for i in range(num_iters):
        if offset + args.dev_batch_size > len(df):
            assert i == num_iters - 1
            varList = generate_test_batch(df, offset, True)
        else:
            varList = generate_test_batch(df, offset)


        prediction = model(varList)
        predictions[offset:offset+args.dev_batch_size] = prediction.data.numpy()[:, 0]
        offset += args.dev_batch_size

    return predictions[0:len(df)]


model = torch.load('records/3.pt')
test_df['unit_sales'] = 0


for i, group in grouped:
    print 'Starting new iteration %i...'%i
    new_features = np.zeros((55, 4100, 1))
    df = add_historical_features_test(group, features)
    predictons = test(df, model)

    test_df.ix[grouped.groups[i], 'unit_sales'] = predictons

    idx1 = group['store_nbr'].tolist()
    idx2 = group['item_nbr'].tolist()

    new_features[idx1, idx2, 0] = predictons
    features = np.concatenate((features, new_features[:, :, :]), axis=2)

np.save('data/april/model3_features.npy', features)
print features.shape
test_df.to_csv('data/april/model3_predictions', index=False)
print test_df.head()
print test_df.tail()



