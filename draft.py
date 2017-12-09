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
# df_source = pd.read_csv('data/april/april.csv')

# for i in range(4, 9):
#     print 'Processing the %i th month...' %i
#     df = df_source.loc[df_source['month'] == i, :]
#     df.to_csv('data/april/' + str(i) + '.csv', index=False)
# save_path = 'data/april/may_sales.csv'

# fill30days(df, save_path)
# df.info()
df.describe()