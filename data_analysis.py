import pandas as pd
import gc
import collections

def readin_data(path='data/train.csv', skiprows=range(1, 101688780)):
    dtypes = {'id':'int64', 'item_nbr':'int32', 'store_nbr':'int8', 'onpromotion':'int8', 'unit_sales':'float32'}
    train = pd.read_csv('data/train.csv', dtype=dtypes, parse_dates=['date'], skiprows=skiprows)
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

def pre_processing(train):
    train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
    train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
    train['dow'] = train['date'].dt.dayofweek
    train['month'] = train['date'].dt.month
    train.loc[:, 'unit_sales'].fillna(0, inplace=True)
    train.loc[:, 'onpromotion'].fillna(0, inplace=True)
    del train['date']

    print 'Adding perishable...'
    peri_dict = perishable_dict()
    train['perishable'] = train['item_nbr'].apply(lambda x:peri_dict[x])

    train = item2idx(train)
    train = pd.get_dummies(train, columns=['month', 'dow'])

    train= train[['unit_sales', 'perishable', 'store_nbr', 'item_nbr', 'onpromotion', 'month_1', 'month_2', 'month_3',
                          'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'dow_0', 'dow_1', 'dow_2',
                  'dow_3', 'dow_4', 'dow_5', 'dow_6']]

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