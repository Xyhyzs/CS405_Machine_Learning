import numpy as np
import pandas as pd
from time import time
from IPython.display import display


import visuals as vs




data = pd.read_csv("census.csv")


display(data.head(n=2))

income_raw =data["income"]

features_raw = data.drop("income",axis = 1)

vs.distribution(data)