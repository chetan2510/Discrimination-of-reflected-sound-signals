import glob

import pandas as pd


path =r'C:\Users\Prashant\Desktop\Masters\Academics\Individual Project\Others\Machine Learning Project - Discrimination of reflected sound signals - CNN\DataConversion\Object 2'

filenames = glob.glob(path + "/*/*/*.csv")

dfs = []

for filename in filenames:

    dfs.append(pd.read_csv(filename, header=None, usecols=list(range(16384))))

data = pd.concat(dfs)

data.to_csv(r'C:\Users\Prashant\Desktop\Masters\Academics\Individual Project\Others\Machine Learning Project - Discrimination of reflected sound signals - CNN\DataConversion\object2.csv', index = False, header=False)