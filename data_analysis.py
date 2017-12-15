import pandas as pd
import gc
import collections
import time
import numpy as np

def readin_data(path='data/train.csv', skiprows=range(1, 101688780)):
    dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':'int8', 'unit_sales':'float32'}
    train = pd.read_csv(path, dtype=dtypes, parse_dates=['date'], skiprows=skiprows)
                    #skiprows=range(1, 101688780)) #Skip dates before 2017-01-01)
    print 'csv data loaded! Start preprocessing'
    return train


def item2idx(df):
    items = pd.read_csv('data/items.csv', usecols=['id', 'item_nbr']).as_matrix()
    items_dict = {}
    for i in range(len(items)):
        items_dict[items[i, 1]] = items[i, 0]

    def search(x):
        return items_dict[x]

    print 'Start to apply....'
    df['item_nbr'] = df['item_nbr'].apply(search)
    print 'End apply!!'
    return df


def perishable_dict(path='data/items.csv'):
    df = pd.read_csv(path, usecols=['item_nbr', 'perishable']).as_matrix()
    peri = collections.defaultdict(int)
    for i in range(len(df)):
        if df[i][1] == 1:
            peri[df[i][0]] = 1.25
        else:
            peri[df[i][0]] = 1
    return peri

def pre_processing(train, is_train=True):
    if is_train:
        train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
        train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
        train.loc[:, 'unit_sales'].fillna(0, inplace=True)

    train['date'] = pd.to_datetime(train['date'])
    train['dow'] = train['date'].dt.dayofweek
    train['month'] = train['date'].dt.month
    train['day'] = train['date'].dt.day

    train.loc[:, 'onpromotion'].fillna(0, inplace=True)
    # del train['date']

    print 'Adding perishable...'
    peri_dict = perishable_dict()
    train['perishable'] = train['item_nbr'].apply(lambda x:peri_dict[x])

    train = item2idx(train)
    # train = pd.get_dummies(train, columns=['month', 'dow'])

    if is_train:
        train= train[['unit_sales', 'perishable', 'store_nbr', 'item_nbr', 'onpromotion', 'month', 'dow', 'day', 'date']]
    else:
        train = train[['perishable', 'store_nbr', 'item_nbr', 'onpromotion', 'month', 'dow', 'day', 'date']]
    # print train.store_nbr.unique().shape # 54
    # print train.item_nbr.unique().shape # 4018
    print 'Finished preprocessing. For next step please start to split...'

    return train


def generate_features(path='data/april/april.csv', save_path='data/april/april_sales.npy'):
    '''
    This function generates a store * item * days (until Aug 15) numpy array and store it
    :param path: a path to csv with features ['unit_sales', 'date', 'store_nbr', 'item_nbr']
    :return:
    '''
    df = pd.read_csv(path, usecols=['unit_sales', 'date', 'store_nbr', 'item_nbr'])
    df['date'] = pd.to_datetime(df['date'])

    df['elapsed'] = (df['date'] - df.ix[0, 'date']).dt.days

    grouped = df.groupby(['store_nbr', 'item_nbr'])

    num_days = (df.ix[len(df) - 1, 'date'] - df.ix[0, 'date']).days + 1
    sales = np.zeros((55, 4100, num_days))


    print 'Start iteration...'
    for name, group in grouped:
        if len(group) != num_days:
            day_indices = group.loc[:, 'elapsed'].tolist()
            sales[name[0], name[1], day_indices] = group.loc[:, 'unit_sales']

        else:
            sales[name[0], name[1], :] = group.loc[:, 'unit_sales']
    np.save(save_path, sales)
    print 'Data saved!'


def split(df):
    '''
    This function splits the train and dev set. Dev set would be from Aug 1 to Aug 15
    :param df:
    :return: data will be pandas dataframe, labels will be numpy array
    '''
    train = df.query('month < 8')
    dev = df.query('month == 8')

    dev_label = dev['unit_sales'].as_matrix()
    dev_data = dev.drop('unit_sales', 1)
    del dev
    gc.collect()

    train_label = train['unit_sales'].as_matrix()
    train_data = train.drop('unit_sales', 1)
    del train
    gc.collect()

    print 'Data splitted! Labels will be numpy arrays. Next step please save the data and labels...'
    return train_data, train_label, dev_data, dev_label


def add_historical_features(df, feature1, feature2, days=15):
    '''
    This function adds historical features to the dataset
    :param df:
    :return:
    '''
    histories = np.zeros((len(df), days))
    df['date'] = pd.to_datetime(df['date'])
    start = time.time()
    for i, row in df.iterrows():
        if i % 100000 == 0:
            print 'Processing the %i th row...' %i
            print 'Time cost = ', time.time() - start
            start = time.time()
        if row['year'] == 0:
            diff = (row['date'] - pd.to_datetime('2016-07-01')).days
            histories[i,:] = feature1[row['store_nbr'], row['item_nbr'], diff-days:diff]
        else:
            diff = (row['date'] - pd.to_datetime('2017-04-01')).days
            histories[i, :] = feature2[row['store_nbr'], row['item_nbr'], diff - days:diff]
    cols = []

    for i in range(days):
        cols.append('us'+str(i))
    np.save('data/april/histories_dev.npy', histories)
    histories = pd.DataFrame(histories, columns=cols)
    df = pd.concat((df, histories), axis=1)
    print df.head()
    print df.tail()
    return df


if __name__ == "__main__":
    data = readin_data()
    data = pre_processing(data)
    train_data, train_label, dev_data, dev_label = split(data)

    train_data.to_csv('data/Jan17Jul_split/train_data.csv', index=False)
    train_label.to_csv('data/Jan17Jul_split/train_labels.csv', index=False, header='unit_sales')
    dev_label.to_csv('data/Jan17Jul_split/dev_labels.csv', index=False, header='unit_sales')
    dev_data.to_csv('data/Jan17Jul_split/dev_data.csv', index=False)

    print train_data.shape # (22237293, 5)
    print train_label.shape
    print dev_data.shape # (1570968, 5)
    print dev_label.shape