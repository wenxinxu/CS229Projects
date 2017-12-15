import pandas as pd
from data_analysis import *
import numpy as np

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

# df = pd.read_csv('data/april/may.csv')
# df_source = pd.read_csv('data/april/april.csv')
#
# df['date'] = pd.to_datetime(df['date'])
# df_source['date'] = pd.to_datetime(df_source['date'])

# for i in range(4, 9):
#     print 'Processing the %i th month...' %i
#     df = df_source.loc[df_source['month'] == i, :]
#     df.to_csv('data/april/' + str(i) + '.csv', index=False)
# save_path = 'data/april/may_sales.csv'

# fill30days(df, save_path)
# df.info()
# grouped = df_source.groupby(['store_nbr', 'item_nbr', 'month', 'day'])
#
#
#
#
# for i in range(1, 16):
#     print 'Adding %i th days ago...' % i
#     df['new_date'] = df['date'] - pd.to_timedelta(i, unit='d')
#     df['new_day'] = df['new_date'].dt.day
#     df['new_month'] = df['new_date'].dt.month
#
#     del df['new_date']
#     now = time.time()
#
#     # df['us' + str(i)] = df.apply(lambda row: find_unit_sales(row['store_nbr'], row['item_nbr'],row['new_month'],
#     #                                                              row['new_day']), axis=1)
#
#     sales = []
#     print 'Started iteration'
#     for j, row in df.iterrows():
#         if j % 100000 == 0:
#             print 'Processing the %i th row..' %j
#         sales.append(find_unit_sales(row['store_nbr'], row['item_nbr'],row['new_month'], row['new_day']))
#     df['us' + str(i)] = pd.DataFrame(sales)
#
#     print 'Takes time = ', time.time() - now
#     del df['new_day']
#     del df['new_month']
#     df.to_csv('data/april/may_teacher_forcing.csv', index=False)
#
# print df.iloc[0:5, :]
# df.to_csv('data/april/may_teacher_forcing.csv', index=False)
#
# def find_unit_sales(store, item, date, grouped):
#     if (store, item, date) in grouped.groups:
#         return grouped.get_group((store, item, date))['unit_sales']
#     else:
#         return 0
#
# def fillDays(df):
#     df['date'] = pd.to_datetime(df['date'])
#     grouped = df.groupby(['store_nbr', 'item_nbr', 'date'])
#
#     for i in range(2, 16):
#         print 'Adding %i th days ago...' % i
#         sales = []
#         start = time.time()
#         for j in range(3100000, len(df)):
#             if df.loc[df.index[j], 'date'].month < 5:
#                 sales.append(0)
#                 continue
#             if j % 100000 == 0:
#                 print 'Processing the %i th row..' % j
#                 print 'Time cost = ', time.time() - start
#                 start = time.time()
#
#             new_date = df.loc[df.index[j], 'date'] - pd.to_timedelta(i, unit='d')
#             sale = find_unit_sales(df.loc[df.index[j], 'store_nbr'], df.loc[df.index[j], 'item_nbr'], new_date, grouped)
#             sales.append(sale)
#         np.save('data/april/f' + str(i) + '.npy', arr=sales)
#         print 'Saved the %i th feature...' %i
#


# fillDays(df)

# df = pd.read_csv('data/april/may.csv')
# b = pd.read_csv('data/april/april.csv')
#
# df['year'] = 1
# a = pd.merge(b, df, how='outer')
# print a.tail()
# print a.head()
# print len(a)
# a.to_csv('data/april/aa.csv')
# print grouped.get_group((15, 17)).head(50)

# df = pd.read_csv('data/april/julaug16.csv')
# print df.head()
# print df.tail()
# print(len(df))
# generate_features(path='data/april/julaug16.csv', save_path='data/april/julaug16_feature.npy')





# df = pd.read_csv('data/april/april.csv')
# # print(len(df))
# df = df.query('(month > 4) | (day > 15)')
# train_data, train_labels, dev_data, dev_labels = split(df)
# #
# #
# #
# #
# train2 = pd.read_csv('data/april/julaug16.csv')
# print train2.head()
# print train2.tail()
# train2 = train2.query('(month > 7) | (day > 15)')
#
# train_l2 = train2['unit_sales'].as_matrix()
# train_d2 = train2.drop('unit_sales', 1)
#
# train_data = pd.concat((train_d2, train_data), axis=0)
# train_labels = np.concatenate((train_l2, train_labels), axis=0)
#
# train_data.to_csv('data/april/train.csv', index=False)
# dev_data.to_csv('data/april/dev.csv', index=False)
# np.save('data/april/train_labels.npy', train_labels)
# np.save('data/april/dev_labels.npy', dev_labels)
#
# print train_data.head()
# print train_data.tail()
#
# print dev_data.head()
# print dev_data.tail()

# df = pd.read_csv('data/april/julaug16.csv')
# print df.head()
# print df.tail()
# df['date'] = pd.to_datetime(df['date'])
# year = df['date'].dt.year
# df = df.loc[year == 2016]
# df.to_csv('data/april/julaug16.csv', index=False)
# generate_features('data/april/julaug16.csv', 'data/april/julaug16_features.npy')

# train_data = pd.read_csv('data/april/train.csv')
# dev_data = pd.read_csv('data/april/dev.csv')
# feature2 = np.load('data/april/april_sales.npy')
# feature1 = np.load('data/april/julaug16_features.npy')
#
# df_train = add_historical_features(dev_data, feature1, feature2=feature2)
# df_train.to_csv('data/april/df_dev.csv', index=False)
a = pd.read_csv('data/april/test_data2.csv')
grouped = a.groupby(['day'])
for i, group in grouped:
        print '~~~~~~~~~~~~~~'
        print group.head(10)
        print group.tail(10)



