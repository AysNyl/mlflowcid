from sklearn import datasets
from pathlib import Path
import numpy as np
import seaborn as sns

RDIR = Path('C:/Users/Ayush/mlflowcid')

def iris():
    dt  = sns.load_dataset('iris')
    dt.to_csv(RDIR.joinpath('data/iris.csv'))

if __name__ == '__main__':
    iris()
