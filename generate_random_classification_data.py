# Last amended: 28th July, 2022
# My folder: C:\Users\Ashok\OneDrive\Documents\python
# Generates data for classification
#You have control over:
#        i)   How many datasets to generate
#        ii)  How many max/min classes to generate randomly
#        iii) How many max/min samples per dataset
#        iv)  How many max/min categorical variables
#        v)   How many max/min no of columns
#        vi)  How many NULLS to generate


# Set constants in para 1.1, save the
#  file and then run in Anaconda prompt as:

#       python generate_random_classification_data.py


# 1.0 Call libraries
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer
import os

# 1.1 ## Set these constants
#        Folder where your generated csv files will be saved
#        Folder will be generated if it does not exist
pathToFolder = r"C:\temp\fake\test"
# Files will be named as bda25001 to bda2500XX
rollNumberSeries = 25000    # Class roll numbers

noOfDatasetsToGen = 60      # So many diff csv files will be created

minSamples = 1000   # Per dataset
maxSamples = 8000   # Per dataset

minFeatures = 15    # Per dataset
maxFeatures = 30    # PEr dataset

minInfFeatures = 7  # Per dataset
maxInfFeatures = 10 # PEr dataset

min_dis = 2         # Min discrete columns
max_dis = 5         # MAx dicrete columns

min_nulls = 8       # Min number of nulls
max_nulls = 15      # MAx number of nulls


# 1.2 Random number generator
rng = np.random.default_rng()


# 2.0 Define function to generate a single dataset
def generate_data( ):
    # Parameters for make classifier
    params = {
              'n_samples' : rng.integers(low = minSamples, high = maxSamples),  # No of samples
              'n_features': rng.integers(low = minFeatures, high = maxFeatures),      # No of features
              'n_informative': rng.integers(low = minInfFeatures, high = maxInfFeatures),    # No of informative features
              'n_classes' : int(2 + np.around(rng.random()))  ,     # No of classes in target 2 or 3
              'flip_y' : rng.random()/10,                           # Introduce some noise in target
              'class_sep' : 0.5 + rng.random(),                     # Difficulty level of classification
              'scale' : None,                                       # No scaling of data
              'shuffle' : True                                      # Shuffle dataset
              }

    # Parameters for discretizer
    params1 = {
             'n_bins'  :  rng.integers(low = 3, high = 6),         # Discretise into how many bins
             'encode'  : 'ordinal',                                # Type of discretization: 1,2,1,1,3
             'strategy': 'uniform'
            }
    # How many discrete columns
    noOfDiscreteCols = rng.integers(low = min_dis, high = max_dis)             # No of columns to be discretized

    # Generate classification data
    dx = make_classification(**params)
    # Discretize some cols
    kb = KBinsDiscretizer(**params1)
    # Limit decimal places to 2
    X = np.round(dx[0],2)   # Predictirs
    y = dx[1]               # Target
    X_d = kb.fit_transform(X[:,  : noOfDiscreteCols])

    # Stack discrete data with continuous
    X = np.hstack([X_d,X[:,  noOfDiscreteCols:]])

    NoOfNulls = rng.integers(low = min_nulls, high = max_nulls)
    rows = X.shape[0]
    cols = X.shape[1]
    X = X.reshape( 1, rows * cols)
    null_loc = rng.integers(low = 1, high = (rows-3) * (cols - 3), size = (1,NoOfNulls))
    X[0,null_loc] = np.nan
    X = X.reshape(rows, cols)
    X = np.hstack([X, y.reshape(-1,1)])
    # Transform to pandas DataFrame
    # Create column names
    col_names_cat = [ "cat_" + str(i) for i in range(noOfDiscreteCols) ]
    col_names_num = [ "num_" + str(i) for i in range(X.shape[1] - noOfDiscreteCols) ]
    # Rename last column as target column
    col_names_num[X.shape[1]-noOfDiscreteCols-1] = 'target'

    col_names_cat.extend(col_names_num)
    col_names = col_names_cat


    # FInally return generated data
    return pd.DataFrame(X, columns = col_names)


# Check if above function works
test  = generate_data()
test.head()
# And where are nulls?
np.sum(np.isnan(test))



# 3.0 Delete all existing files from pathToFolder
# 3.1 Delete existing files in the folder
if not os.path.exists(pathToFolder):
    os.makedirs(pathToFolder)
else:
    for f in os.listdir(pathToFolder):
        os.remove(os.path.join(pathToFolder, f))

os.chdir(pathToFolder)

# Clear console of all clutter
os.system("cls")
print("")
print("")
print("--Generating classification datasets--")
print("")
print("")


# 3.0 Generate and save files to pathToFolder
for i in range(noOfDatasetsToGen):
    # Give some name to the dataset
    name = "bda" +  str(rollNumberSeries + i + 1)+str(".csv")   # Roll numbers are from 25001 to 250060
    r = generate_data()
    r.to_csv(name, index = False)

print("")
print("##########################################")
print("Folder where your {} generated csv files are: {} ".format(noOfDatasetsToGen, pathToFolder))
print("##########################################")
print("")
########################################################
