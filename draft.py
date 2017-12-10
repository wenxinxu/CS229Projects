import pandas as pd
from data_analysis import *

# data = pd.read_csv('data/april/april.csv') #14489514
# print len(data)

# data = readin_data('data/train.csv')
# data = pre_processing(data)
# data = data.loc[data['month'] >= 5, :]
# print data.iloc[0:5,:]
# data.to_csv('data/april/may.csv', index=False)
#
# df = pd.read_csv('data/april/may.csv', nrows=5)
# # df['date'] = pd.to_datetime(df['date'])
# # df['new_date'] = df['date'] - pd.to_timedelta(10, unit='d')
# print len(df.loc[df['store_nbr'] == 2, :])

df = pd.read_csv('data/april/may.csv')
df_source = pd.read_csv('data/april/april.csv')

df['date'] = pd.to_datetime(df['date'])
df_source['date'] = pd.to_datetime(df_source['date'])

# for i in range(4, 9):
#     print 'Processing the %i th month...' %i
#     df = df_source.loc[df_source['month'] == i, :]
#     df.to_csv('data/april/' + str(i) + '.csv', index=False)
# save_path = 'data/april/may_sales.csv'

# fill30days(df, save_path)
# df.info()
grouped = df_source.groupby(['store_nbr', 'item_nbr', 'month', 'day'])

def find_unit_sales(store, item, month, day):
    if (store, item, month, day) in grouped.groups:
        return grouped.get_group((store, item, month, day))['unit_sales']
    else:
        return 0


for i in range(1, 16):
    print 'Adding %i th days ago...' % i
    df['new_date'] = df['date'] - pd.to_timedelta(i, unit='d')
    df['new_day'] = df['new_date'].dt.day
    df['new_month'] = df['new_date'].dt.month

    del df['new_date']
    now = time.time()

    # df['us' + str(i)] = df.apply(lambda row: find_unit_sales(row['store_nbr'], row['item_nbr'],row['new_month'],
    #                                                              row['new_day']), axis=1)

    sales = []
    print 'Started iteration'
    for j, row in df.iterrows():
        if j % 100000 == 0:
            print 'Processing the %i th row..' %j
        sales.append(find_unit_sales(row['store_nbr'], row['item_nbr'],row['new_month'], row['new_day']))
    df['us' + str(i)] = pd.DataFrame(sales)

    print 'Takes time = ', time.time() - now
    del df['new_day']
    del df['new_month']
    df.to_csv('data/april/may_teacher_forcing.csv', index=False)

print df.iloc[0:5, :]
df.to_csv('data/april/may_teacher_forcing.csv', index=False)