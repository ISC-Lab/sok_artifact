
print('Loading packages')
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import matplotlib

from sklearn.manifold import TSNE
from sklearn import mixture
from scipy import linalg

import time
print('Successfully loaded packages')

print('Loading dataframes')
papers_reference = pd.read_pickle('papers_reference')
codebook_reference = pd.read_pickle('codebook_reference')
print('Successfully loaded dataframes')