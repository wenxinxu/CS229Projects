import pandas as pd
import gc
import collections
import time

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
    print 'Finished preprocessing. Start to split...'

    return train


def split(df):
    # Need to change when including more training data
    train = df.iloc[0:22237293, :] # Aug
    dev = df.iloc[22237293:, :] # July
    dev_label = dev['unit_sales']
    dev_data = dev.drop('unit_sales', 1)
    del dev
    gc.collect()

    train_label = train['unit_sales']
    train_data = train.drop('unit_sales', 1)
    del train
    gc.collect()

    print 'Data splitted! Start to dump...'
    return train_data, train_label, dev_data, dev_label


def fill30days(df, save_path):
    '''
    Fill in the sales unit for the previous 30 days
    :param df_source: where to look for historical records
    :return:
    '''
    def find_unit_sales(date, store, item, df_source):
        '''
        Help find the unit sales of a certain day
        :param date:
        :param store:
        :param item:
        :return:
        '''
        d = df_source.loc[df_source['date'] == date, :]
        d = d.loc[d['store_nbr'] == store, :]
        d = d.loc[d['item_nbr'] == item, :]

        if len(d) != 0:
            return d['unit_sales']
        else:
            return 0

    df['date'] = pd.to_datetime(df['date'])

    df4 = pd.read_csv('data/april/4.csv')
    df5 = pd.read_csv('data/april/5.csv')
    df6 = pd.read_csv('data/april/6.csv')
    df7 = pd.read_csv('data/april/7.csv')
    df8 = pd.read_csv('data/april/8.csv')

    dfs = {4:df4, 5:df5, 6:df6, 7:df7, 8:df8}

    for i in range(1, 10):
        print 'Adding %i th days ago...' %i
        df['new_date'] = df['date'] - pd.to_timedelta(i, unit='d')
        print 'Start applying...'
        now = time.time()
        df['us'+str(i)] = df.apply(lambda row: find_unit_sales(row['new_date'], row['store_nbr'], row['item_nbr'],
                                    pd.concat((dfs[row['month'] - 1], dfs[row['month']]), axis=0)), axis=1)
        print 'Takes time = ', time.time() - now
        del df['new_date']

    print df.iloc[0:5, :]
    df.to_csv(save_path, index=False)

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