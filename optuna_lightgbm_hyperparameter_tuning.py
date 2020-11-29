# Last amended: 28th Nov, 2020
# Myfolder: C:\Users\Administrator\OneDrive\Documents\breast_cancer
#
# Ref: https://medium.com/optuna/lightgbm-tuner-new-optuna-integration-for-hyperparameter-optimization-8b7095e99258
#      https://github.com/optuna/optuna/blob/master/examples/lightgbm_tuner_simple.py
#
# Objective:
#           i) Learn to use automated tuning of lightgbm
#          ii) Using optuna
#
# See also 'h2o_talkingData.ipynb' in folder:
#   C:\Users\Administrator\OneDrive\Documents\talkingdata
"""
Optuna example that optimizes a classifier configuration
for cancer dataset using LightGBM tuner.
In this example, we optimize the validation log loss of
cancer detection.

"""

# 1.0 Call libraries
import numpy as np
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 1.1 Import optuna integration with lightgbm
# Install as: conda install -c conda-forge optuna
import optuna.integration.lightgbm as lgb

# 2.0 Get data and split it
data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
train_x, val_x, train_y, val_y = train_test_split(data, target, test_size=0.25)

# 2.1 Transform train_x and val_x to lightgbm data-matricies
dtrain = lgb.Dataset(train_x, label=train_y)
dval = lgb.Dataset(val_x, label=val_y)

# 3.0 Set fixed hyper-params
params = {                           # Specify params that are fixed
           "objective": "binary",
           "metric": "binary_logloss",
           "verbosity": -1,
           "boosting_type": "gbdt",
          }


# 3.1 Note that unlike in sklearn, here there is
#     no instantiation of LightGBM model
#     Start modeling as also tuning hyperparameters

model = lgb.train(
                   params,                     # Just fixed params only
                   dtrain,                     # Dataset
                   valid_sets=[dtrain, dval],  # Evaluate performance on these datasets
                   verbose_eval=100,
                   early_stopping_rounds=100
                  )

### Model is ready

# 4.0 Make prediction
prediction = np.rint(
                     model.predict(
                                    val_x,    # Note that it is not lightgbm dataset
                                    num_iteration = model.best_iteration
                                    )
                    )
# 4.1 Determine accuracy
accuracy = accuracy_score(val_y, prediction)

# 4.2 Get best params
best_params = model.params
best_params
accuracy

# 4.3
for key, value in best_params.items():
    print("    {}: {}".format(key, value))

######################
