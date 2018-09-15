from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pandas as pd
from time import time
from IPython.display import display


import visuals as vs
data = pd.read_csv("census.csv")


display(data.head(n=2))

income_raw =data["income"]

features_raw = data.drop("income",axis = 1)





skewed = ['capital-gain','capital-loss']

features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed]   = features_raw[skewed].apply(lambda x: np.log(x+1))

vs.distribution(features_log_transformed,transformed=True)
scaler  = MinMaxScaler()
numerical =['age','education-num','capital-gain','capital-loss','hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

display(features_log_minmax_transform.head(n=5))