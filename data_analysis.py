import pandas as pd
import gc

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


def pre_processing(train):
    train.loc[(train.unit_sales<0),'unit_sales'] = 0 # eliminate negatives
    train['unit_sales'] =  train['unit_sales'].apply(pd.np.log1p) #logarithm conversion
    train['dow'] = train['date'].dt.dayofweek
    train['month'] = train['date'].dt.month
    train.loc[:, 'unit_sales'].fillna(0, inplace=True)
    train.loc[:, 'onpromotion'].fillna(0, inplace=True)
    del train['date']

    train = item2idx(train)
    train = pd.get_dummies(train, columns=['month', 'dow'])

    # print train.store_nbr.unique().shape # 54
    # print train.item_nbr.unique().shape # 4018
    print 'Finished preprocessing. Start to split...'

    return train


def split(df):
    # Need to change when including more training data
    dev = df.iloc[0:22237293, :] # Aug
    train = df.iloc[22237293:, :] # July
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
    print data.iloc[0:5, :]
    data = pre_processing(data)

    print data.iloc[0:5, :]
    train_data, train_label, dev_data, dev_label = split(data)

    train_data.to_csv('data/Jan17Jul_split/train_data.csv', index=False)
    train_label.to_csv('data/Jan17Jul_split/train_labels.csv', index=False, header='unit_sales')
    dev_label.to_csv('data/Jan17Jul_split/dev_labels.csv', index=False, header='unit_sales')
    dev_data.to_csv('data/Jan17Jul_split/dev_data.csv', index=False)

    print train_data.shape # (22237293, 5)
    print train_label.shape
    print dev_data.shape # (1570968, 5)
    print dev_label.shape