import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('records/4_error.csv').as_matrix()
plt.plot(df[:,0], df[:,1], label='Training loss')
plt.plot(df[:,0], df[:,2], label='Dev loss')
plt.title('Training curve of residual neural network: with historical features')
plt.legend()
plt.show()