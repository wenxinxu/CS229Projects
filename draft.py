import pandas as pd

train_data = pd.read_csv('data/Jan17Jul_split/train_data.csv', nrows=5)
dev_data = pd.read_csv('data/Jan17Jul_split/dev_data.csv', nrows=5)
train_labels = pd.read_csv('data/Jan17Jul_split/train_labels.csv', nrows=5)
dev_labels = pd.read_csv('data/Jan17Jul_split/dev_labels.csv', nrows=5)
aa = pd.read_csv('data/Jan17Jul_split/aa.csv', nrows=5)



print train_data
print train_labels
print aa