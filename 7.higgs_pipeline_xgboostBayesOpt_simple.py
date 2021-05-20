# -*- coding: utf-8 -*-
"""
Last amended: 09/10/2020
Ref:
 1. https://dataplatform.ibm.com/analytics/notebooks/20c1c2d6-6a51-4bdc-9b2c-0e3f2bef7376/view?access_token=52b727bd6515bd687cfd88f929cc7869b0ea420e668b2730c6870e72e029f0d1
 2. http://krasserm.github.io/2018/03/21/bayesian-optimization/

Objectives:
    1. Reading from hard-disk random samples of big-data
    2. Using PCA
    3. Pipelining with StandardScaler, PCA and xgboost
    4. Grid tuning of PCA and xgboost--Avoid data leakage
    5. Randomized search of parameters
    6. Bayes optimization
    7. Feature importance
    8. Genetic algorithm for tuning of parameters
    9. Find feature importance of any Black box estimator
       using eli5 API


# IMPT NOTE:
# For a complete example that uses both Pipelining and ColumnTransformer
# Please see this website: http://dalex.drwhy.ai/python-dalex-titanic.html
#  And also Folder 24. Pipelining with columntransformer. See also:
#   https://github.com/ModelOriented/DALEX



"""

################### AA. Call libraries #################
# 1.0 Clear ipython memory
%reset -f

# 1.1 Data manipulation and plotting modules
import numpy as np
import pandas as pd


# 1.2 Data pre-processing
#     z = (x-mean)/stdev
from sklearn.preprocessing import StandardScaler as ss

# 1.3 Dimensionality reduction
from sklearn.decomposition import PCA

# 1.4 Data splitting and model parameter search
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


# 1.5 Modeling modules
#     Call sklearn wrapper of xgboost
# """Scikit-Learn Wrapper interface for XGBoost."""
#  """Implementation of the Scikit-Learn API for XGBoost.
# https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
# Stackoverflow:
# https://stackoverflow.com/a/34696477
# https://stackoverflow.com/a/46947191
# Install as: conda install -c anaconda py-xgboost
from xgboost.sklearn import XGBClassifier


# 1.6 Model pipelining
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


# 1.7 Model evaluation metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix

# 1.8
import matplotlib.pyplot as plt
from xgboost import plot_importance

# 1.9 Needed for Bayes optimization
#     Takes an estimator, performs cross-validation
#     and gives out average score
from sklearn.model_selection import cross_val_score

# 1.10 Install as: pip install bayesian-optimization
#     Refer: https://github.com/fmfn/BayesianOptimization
#     conda install -c conda-forge bayesian-optimization
from bayes_opt import BayesianOptimization


# 1.11 Find feature importance of ANY BLACK BOX estimator
#      See note at the end of this code for explanation
#      Refer: https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
#      Install as:
#      conda install -c conda-forge eli5
import eli5
from eli5.sklearn import PermutationImportance


# 1.12 Misc
import time
import os
import gc
import random
# 1.12.1 Used in Randomized parameter search
from scipy.stats import uniform


# 1.13 Set option to dislay many rows
pd.set_option('display.max_columns', 100)

################# BB. Read data randomly #################
# 2.0 Read random chunks of 10% of data


# 2.1 Set working directory
#os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\higgsBoson")
#os.chdir("D:\\data\\OneDrive\\Documents\\higgsBoson")
path = "/home/ashok/Documents/10.higgsBoson"
path="C:\\Users\\ashok\\Desktop\\cbi\\10.higgsBoson"
os.chdir(path)
os.listdir()


# 2.2 Count number of lines in the file
#     Data has 250001 rows including header also
tr_f = "training.csv.zip"


# 2.3 Total number of lines
#     But we will read 40% of data randomly
total_lines = 250000
num_lines = 0.4 * total_lines    # 40% of data


# 2.4 Read randomly 'p' fraction of files
#     Ref: https://stackoverflow.com/a/48589768

p = num_lines/total_lines  # fraction of lines to read (40%)

# 2.4.1 How to pick up random rows from hard-disk
#       without first loading the complete file in RAM
#       Toss a coin:
#           At each row, toss a biased-coin: 60%->Head, 40%->tail
#           If tail comes, select the row else not.
#           Toss a coin: random.random()
#           Head occurs if value > 0.6 else it is tail
#
#       We do not toss the coin for header row. Keep the header

data = pd.read_csv(
         tr_f,
         header=0,   # First row is header-row
         # 'and' operator returns True if both values are True
         #  random.random() returns values between (0,1)
         #  No of rows skipped will be around 60% of total
         skiprows=lambda i: (i >0 ) and (random.random() > p)    # (i>0) implies skip first header row
         )


# 3.0 Explore data
data.shape                # 100039, 33)
data.columns.values       # Label column is the last one
data.dtypes.value_counts()  # Label column is of object type

# 3.1
data.head(3)
data.describe()
data.Label.value_counts()  # Classes are not unbalanced
                           # Binary data
                           #  b: 65558 , s: 34242

# 3.2 We do not need Id column and Weight column
data.drop(columns = ['EventId','Weight'],inplace = True  )
data.shape                    # (100039, 31); 31 Remining columns



# 3.3 Divide data into predictors and target
#     First 30 columns are predictors
X = data.iloc[ :, 0:30]
X.head(2)


# 3.3.1 30th index or 31st column is target
y = data.iloc[ : , 30]
y.head()


# 3.3.2 Can we change datatype to float32 ?
X.min().min()           # -999.0
X.max().max()           # 4543.913

# 3.3.3  Save memory
X = X.astype('float32')


# 3.4 Transform label data to '1' and '0'
#    'map' works element-wise on a Series.
y = y.map({'b':1, 's' : 0})
y.dtype           # int64


# 3.5 Store column names somewhere
#     for use in feature importance

colnames = X.columns.tolist()


# 4. Split dataset into train and validation parts
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.35,
                                                    shuffle = True,
                                                    stratify = y
                                                    )

# 4.1
X_train.shape        # (65025, 30)
X_test.shape         # (35014, 30)
y_train.shape        # (65025,)
y_test.shape         # (35014,)


################# CC. Create pipeline #################
#### Pipe using XGBoost


# 5 Pipeline steps
# steps: List of (name, transform) tuples
#       (implementing fit/transform) that are
#       chained, in the order in which they
#       are chained, with the last object an
#       estimator.
#      Format: [(name, transformer), (name, transformer)..(name, estimator)]
steps_xg = [('sts', ss() ),
            ('pca', PCA()),
            ('xg',  XGBClassifier(silent = False,
                                  n_jobs=3)        # Specify other parameters here
            )
            ]

# 5.1  Instantiate Pipeline object
pipe_xg = Pipeline(steps_xg)

# 5.2 Another way to create pipeline:
#     Not used below
pipe_xg1 = make_pipeline (ss(),
                          PCA(),
                          XGBClassifier(silent = False,
                                        n_jobs=2)
                          )


##################$$$$$$$$$$$#####################
## Jump now to
##   Either:   Grid Search (DD)             para 6
##       Or:   Random Search (EE)
##       Or:   Bayesian Optimization (GG)
##       Or:   Evolutionary Algorithm (HH)
##################$$$$$$$$$$$#####################


##################### DD. Grid Search #################

# 6.  Specify xgboost parameter-range
# 6.1 Dictionary of parameters (16 combinations)
#     Syntax: {
#              'transformerName__parameterName' : [ <listOfValues> ]
#              }
#
# 6.2 What parameters in the pipe are available for tuning
pipe_xg.get_params()

# 6.3
parameters = {'xg__learning_rate':  [0.03, 0.05], # learning rate decides what percentage
                                                  #  of error is to be fitted by
                                                  #   by next boosted tree.
                                                  # See this answer in stackoverflow:
                                                  # https://stats.stackexchange.com/questions/354484/why-does-xgboost-have-a-learning-rate
                                                  # Coefficients of boosted trees decide,
                                                  #  in the overall model or scheme, how much importance
                                                  #   each boosted tree shall have. Values of these
                                                  #    Coefficients are calculated by modeling
                                                  #     algorithm and unlike learning rate are
                                                  #      not hyperparameters. These Coefficients
                                                  #       get adjusted by l1 and l2 parameters
              'xg__n_estimators':   [200,  300],  # Number of boosted trees to fit
                                                  # l1 and l2 specifications will change
                                                  # the values of coeff of boosted trees
                                                  # but not their numbers

              'xg__max_depth':      [4,6],
              'pca__n_components' : [25,30]
              }                               # Total: 2 * 2 * 2 * 2


# 7  Grid Search (16 * 2) iterations
#    Create Grid Search object first with all necessary
#    specifications. Note that data, X, as yet is not specified

#    Data Leakage and pipelining:
#    Pipeline avoids data leakage during GridSearch
#    See this: https://towardsdatascience.com/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976

clf = GridSearchCV(pipe_xg,            # pipeline object
                   parameters,         # possible parameters
                   n_jobs = 2,         # USe parallel cpu threads
                   cv =3 ,             # No of folds
                   verbose =2,         # Higher the value, more the verbosity
                   scoring = ['accuracy', 'roc_auc'],  # Metrics for performance
                   refit = 'roc_auc'   # Refitting final model on what parameters?
                                       # Those which maximise auc
                   )

## 7.1 Delete objects not needed
#      We need X_train, y_train, X_test, y_test
del X
del data
del y
gc.collect()

######
#### @@@@@@@@@@@@@@@@@@@ #################
## REBOOT lubuntu MACHINE HERE
#### @@@@ AND NOW WORK IN sublime @@@@@#####


# 7.2. Start fitting data to pipeline
start = time.time()
clf.fit(X_train, y_train)
end = time.time()
(end - start)/60               # 25 minutes



# 7.3
f"Best score: {clf.best_score_} "            # 'Best score: 0.8804992694908675 '
f"Best parameter set {clf.best_params_}"

# 7.4. Make predictions using the best returned model
y_pred = clf.predict(X_test)
y_pred

# 7.5 Accuracy
accuracy = accuracy_score(y_test, y_pred)
f"Accuracy: {accuracy * 100.0}"             # 'Accuracy: 82.11165535218126'

# 7.6 Confusion matrix
confusion_matrix( y_test,y_pred)

# 7.7 F1 score
f1_score(y_test,y_pred, pos_label = 1)      # 0.8664199696263183
f1_score(y_test,y_pred, pos_label = 0)      # 0.729313857223354

# 7.8 ROC curve
plot_roc_curve(clf, X_test, y_test)


# 7.9 Get feature importances from GridSearchCV best fitted 'xg' model
#     See stackoverflow: https://stackoverflow.com/q/48377296
clf.best_estimator_.named_steps["xg"].feature_importances_
clf.best_estimator_.named_steps["xg"].feature_importances_.shape

################# Using Feature Importance #############

## The following is a quick calculations to show
## what happens if we drop the least important columns

# 7.10 Create a dataframe of feature importances
fe_values = clf.best_estimator_.named_steps["xg"].feature_importances_
df_fe = pd.DataFrame(data = fe_values,index = colnames, columns = ["fe"]).sort_values(by = 'fe')

# 7.11 First five columns with least feature importance are:
list(df_fe.index.values[:5])

# 7.12 Let us drop these from X_train and X_test
Xtrain = X_train.drop(columns = list(df_fe.index.values[:5]))
Xtest = X_test.drop(columns = list(df_fe.index.values[:5]))

# 7.13 Build model again with reduced dataset
clf_dr = GridSearchCV(pipe_xg,            # pipeline object
                   parameters,         # possible parameters
                   n_jobs = 2,         # USe parallel cpu threads
                   cv =3 ,             # No of folds
                   verbose =2,         # Higher the value, more the verbosity
                   scoring = ['accuracy', 'roc_auc'],  # Metrics for performance
                   refit = 'roc_auc'   # Refitting final model on what parameters?
                                       # Those which maximise auc
                   )

# 7.14
start = time.time()
clf_dr.fit(Xtrain, y_train)
end = time.time()
(end - start)/60               # 25 minutes

# 7.15 Make predictions
y_pred_dr = clf_dr.predict(Xtest)

# 7.16 Compare results.
#      Results may be marginally better
f1_score(y_test,y_pred_dr, pos_label = 1)      # 0.8664199696263183
f1_score(y_test,y_pred_dr, pos_label = 0)      # 0.729313857223354
f1_score(y_test,y_pred, pos_label = 1)      # 0.8664199696263183
f1_score(y_test,y_pred, pos_label = 0)      # 0.729313857223354
##################################


##################### EE. Randomized Search #################

# Tune parameters using randomized search
# 8. Hyperparameters to tune and their ranges
parameters = {'xg__learning_rate':  uniform(0, 1),
              'xg__n_estimators':   range(50,300),
              'xg__max_depth':      range(3,10),
              'pca__n_components' : range(20,30)}



# 8.1 Tune parameters using random search
#     Create the object first
rs = RandomizedSearchCV(pipe_xg,
                        param_distributions=parameters,
                        scoring= ['roc_auc', 'accuracy'],
                        n_iter=15,          # Max combination of
                                            # parameter to try. Default = 10
                        verbose = 3,
                        refit = 'roc_auc',
                        n_jobs = 2,          # Use parallel cpu threads
                        cv = 2               # No of folds.
                                             # So n_iter * cv combinations
                        )


# 8.2 Run random search for 25 iterations. 21 minutes
start = time.time()
rs.fit(X_train, y_train)
end = time.time()
(end - start)/60


# 8.3 Evaluate
f"Best score: {rs.best_score_} "            # 'Best score: 0.8780097831252602 '
f"Best parameter set: {rs.best_params_} "


# 8.4 Make predictions from the best returned model
y_pred = rs.predict(X_test)


# 8.5 Accuracy and f1_score
accuracy = accuracy_score(y_test, y_pred)
f"Accuracy: {accuracy * 100.0}"         # 'Accuracy: 82.0142648448913'
f1_score(y_test,y_pred, pos_label = 1)      # 0.8655661892221722

############### FF. Fitting parameters in our model ##############
###############    Model Importance   #################

# 9. Model with parameters of grid search
model_gs = XGBClassifier(
                    learning_rate = clf.best_params_['xg__learning_rate'],
                    max_depth = clf.best_params_['xg__max_depth'],
                    n_estimators=clf.best_params_['xg__max_depth']
                    )

# 9.1 Model with parameters of random search
model_rs = XGBClassifier(
                    learning_rate = rs.best_params_['xg__learning_rate'],
                    max_depth = rs.best_params_['xg__max_depth'],
                    n_estimators=rs.best_params_['xg__max_depth']
                    )


# 9.2 Modeling with both parameters
start = time.time()
model_gs.fit(X_train, y_train)
model_rs.fit(X_train, y_train)
end = time.time()
(end - start)/60


# 9.3 Predictions with both models
y_pred_gs = model_gs.predict(X_test)
y_pred_rs = model_rs.predict(X_test)


# 9.4 Accuracy from both models
accuracy_gs = accuracy_score(y_test, y_pred_gs)
accuracy_rs = accuracy_score(y_test, y_pred_rs)
accuracy_gs
accuracy_rs




# 10 Get feature importances from both models
help(plot_importance)

# 10.1 Plt now

%matplotlib qt5
model_gs.feature_importances_
model_rs.feature_importances_
# 10.1.1 Importance type: 'weight'
plot_importance(
                model_gs,
                importance_type = 'weight'   # default
                )
# 10.1.2 Importance type: 'gain'
#        # Normally use this
plot_importance(
                model_rs,
                importance_type = 'gain', 
                title = "Feature impt by gain"
                )
plt.show()

# 10.1 Print feature importance
#      https://stackoverflow.com/a/52777909
#      https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
"""
importance_type

    ‘weight’ -      the number of times a feature is used to split the data across all trees.
    ‘gain’ -        the average gain across all splits the feature is used in.
    ‘cover’ -       the average coverage across all splits the feature is used in.
    ‘total_gain’ -  the total gain across all splits the feature is used in.
    ‘total_cover’ - the total coverage across all splits the feature is used in.

"""
# 11.0 Get results in a sorted DataFrame
feature_important = model_gs.get_booster().get_score(importance_type='weight')
feature_important
keys = list(feature_important.keys())
values = list(feature_important.values())

data = pd.DataFrame(data=values,
                    index=keys,
                    columns=["score"]).            \
                        sort_values(               \
                                     by = "score", \
                                    ascending=False)

# 11.1 Compare the results in the following DataFrame
#      with that obtained using PermutationImportance
#      of eli5 below.
                            
data

############### GG. Tuning using Bayes Optimization ############
"""
11. Step 1: Define BayesianOptimization function.
            It broadly acts as follows"
            s1. Gets a dictionary of parameters that specifies
                possible range of values for each one of
                the parameters. [Our set: para_set ]
            s2. Picks one value for each one of the parameters
                (from the specified ranges as in (s1)) evaluate,
                a loss-function that is given to it, say,
                accuracy after cross-validation.
                [Our function: xg_eval() ]
            s3. Depending upon the value of accuracy returned
                by the evaluator and also past values of accuracy
                returned, this function, creates gaussian
                processes and picks up another set of parameters
                from the given dictionary of parameters
            s4. The parameter set is then fed back to (s2) above
                for evaluation
            s5. (s2) t0 (s4) are repeated for given number of
                iterations and then final set of parameters
                that optimizes objective is returned

"""
# 11.1 Which parameters to consider and what is each one's range
para_set = {
           'learning_rate':  (0, 1),                 # any value between 0 and 1
           'n_estimators':   (50,300),               # any number between 50 to 300
           'max_depth':      (3,10),                 # any depth between 3 to 10
           'n_components' :  (20,30)                 # any number between 20 to 30
            }


# 11.2 This is the main workhorse
#      Instantiate BayesianOptimization() object
#      This object  can be considered as performing an internal-loop
#      i)  Given parameters, xg_eval() evaluates performance
#      ii) Based on the performance, set of parameters are selected
#          from para_set and fed back to xg_eval()
#      (i) and (ii) are repeated for given number of iterations
#
xgBO = BayesianOptimization(
                             xg_eval,     # Function to evaluate performance.
                             para_set     # Parameter set from where parameters will be selected
                             )



# 12 Create a function that when passed some parameters
#    evaluates results using cross-validation
#    This function is used by BayesianOptimization() object

def xg_eval(learning_rate,n_estimators, max_depth,n_components):
    # 12.1 Make pipeline. Pass parameters directly here
    pipe_xg1 = make_pipeline (ss(),                        # Why repeat this here for each evaluation?
                              PCA(n_components=int(round(n_components))),
                              XGBClassifier(
                                           silent = False,
                                           n_jobs=2,
                                           learning_rate=learning_rate,
                                           max_depth=int(round(max_depth)),
                                           n_estimators=int(round(n_estimators))
                                           )
                             )

    # 12.2 Now fit the pipeline and evaluate
    cv_result = cross_val_score(estimator = pipe_xg1,
                                X= X_train,
                                y = y_train,
                                cv = 2,
                                n_jobs = 2,
                                scoring = 'f1'
                                ).mean()             # take the average of all results


    # 12.3 Finally return maximum/average value of result
    return cv_result


# 13. Gaussian process parameters
#     Modulate intelligence of Bayesian Optimization process
#     This parameters controls how much noise the GP can handle,
#     so increase it whenever you think that 'target' is very noisy
#gp_params = {"alpha": 1e-5}      # Initialization parameter for gaussian
                                  # process.

# 14. Fit/train (so-to-say) the BayesianOptimization() object
#     Start optimization. 25minutes
#     Our objective is to maximize performance (results)
start = time.time()
xgBO.maximize(init_points=5,    # Number of randomly chosen points to
                                 # sample the target function before
                                 #  fitting the gaussian Process (gp)
                                 #  or gaussian graph
               n_iter=25,        # Total number of times the
               #acq="ucb",       # ucb: upper confidence bound
                                 #   process is to be repeated
                                 # ei: Expected improvement
               # kappa = 1.0     # kappa=1 : prefer exploitation; kappa=10, prefer exploration
#              **gp_params
               )
end = time.time()
(end-start)/60


# 15. Get values of parameters that maximise the objective
xgBO.res        # It is a list of dictionaries
                # Each dictionary records what happened with a set or parameters
xgBO.max        # Parameters that gave best results


################### HH. Tuning using genetic algorithm ##################
## Using genetic algorithm to find best parameters
#  See at the end of ths code: How evolutionary algorithm work?
#  Ref: https://github.com/rsteca/sklearn-deap
#       https://github.com/rsteca/sklearn-deap/blob/master/test.ipynb

# Install as:
# pip install sklearn-deap
from evolutionary_search import EvolutionaryAlgorithmSearchCV


steps_xg = [('sts', ss()),
            ('pca', PCA()),
            ('xg',  XGBClassifier(silent = False,
                                  n_jobs=2)        # Specify other parameters here
            )
            ]

# Instantiate Pipeline object
pipe_xg = Pipeline(steps_xg)

# Specify a grid of parameters. Unlike in Bayes Opitmization,
#  where a range is specified, a grid is specified here.
param_grid = {'xg__learning_rate':  [0.03, 0.04, 0.05],
              'xg__n_estimators':   [200,  300],
              'xg__max_depth':      [4,6],
              'pca__n_components' : [25,30],
              'pca__svd_solver'    : [ 'full', 'randomized']

              }

clf2 = EvolutionaryAlgorithmSearchCV(
                                   estimator=pipe_xg,  # How will objective be evaluated
                                   params=param_grid,  # Parameters range
                                   scoring="accuracy", # Criteria
                                   cv=2,               # No of folds
                                   verbose=True,
                                   population_size=10,  # Should generally be large
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   #tournament_size=3,
                                   generations_number=10
                                   )


start = time.time()
clf2.fit(X_train, y_train)   # 1hr 2 minute
end = time.time()
(end-start)/60


clf2.best_params_

# Our cvresults table (note, includes all individuals
#   with their mean, max, min, and std test score).
out = pd.DataFrame(
                  clf2.cv_results_
                   )

out = out.sort_values(
                     "mean_test_score",
                      ascending=False
                      )

out.head()


y_pred_gen = clf2.predict(X_test)
accuracy_gen = accuracy_score(y_test, y_pred_gen)
accuracy_gen    # 81.88 %

####################### I am done ######################

"""
How PermutationImportance works?
Ref: https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html#algorithm
    Remove a feature only from the test part of the dataset,
    and compute score without using this feature. It doesn’t
    work as-is, because estimators expect feature to be present.
    So instead of removing a feature we can replace it with
    random noise - feature column is still there, but it no
    longer contains useful information. This method works if
    noise is drawn from the same distribution as original
    feature values (as otherwise estimator may fail).
    The simplest way to get such noise is to shuffle values
    for a feature, i.e. use other examples’ feature values -
    this is how permutation importance is computed.
"""

"""
How evolutionary algorithm work?
Refer: https://en.wikipedia.org/wiki/Evolutionary_algorithm#Implementation
       Page 238, Chapter 7 of Book Artificial Intelligence on Moodle

    Step One: Generate the initial population of parameter sets.
              Let us say there are only two parameters: (max_depth,n_estimators)
              Then our population of, say, fifty chromosomes would be like:
              C1 = (3, 10), C2 = (3,15), C3 = (4,10), C4 = (5,16), C5 = (4,16)....

    Step Two: Evaluate the fitness-level of each chromosome in that population.

    Step Three: Depending upon, fitness levels, assign probbaility of selection
                to each pair of parameters. More the fit, higher the probability
                of selection in the next step

    Step Four:  Select a pair for mating as per assigned probabilities.
                Perform cross-over between the selected pair with probability,say, 0.7.
                (that is prob that given pair is crossed over is 0.7)

                Cross-over is done as:
                For example let max_depth for C2 and C4 be represented, as:
                   3      5
                 0011    0101

                Cross-over at breakpoint two-bits from left may produce:
                 0001    0111

   Step Five: Perform mutation of each chromosome with probability, say, 0.001

              Mutation may change  0001, to say, 1001 by flipping any bit

   Step Six :  Perform the three steps: i) Select, ii) perform cross-over and
               iii) mutate till you get another population of 50 pairs

   Step Seven: Goto Step 2 and iterate till number of generations are exhausted.


"""
"""

parameters = {'learning_rate':  [0.03, 0.05],
              'n_estimators':   [200,  300],
              'max_depth':      [4,6]
              }                               # Total: 2 * 2 * 2 * 2


# 7  Grid Search (16 * 2) iterations
#    Create Grid Search object first with all necessary
#    specifications. Note that data, X, as yet is not specified

xgb = XGBClassifier(silent = False, n_jobs=2)

clfz = GridSearchCV(xgb,
                   parameters,         # possible parameters
                   n_jobs = 2,         # USe parallel cpu threads
                   cv =2 ,             # No of folds
                   verbose =2,         # Higher the value, more the verbosity
                   scoring = ['accuracy', 'roc_auc'],  # Metrics for performance
                   refit = 'roc_auc'   # Refitting final model on what parameters?
                                       # Those which maximise auc
                   )


# 7.2. Start fitting data to pipeline
start = time.time()
clfz.fit(X_train, y_train)
end = time.time()
(end - start)/60               # 25 minutes


"""




# 7.11
#  Find feature importance of any BLACK Box model
#  Refer: https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html
#  See note at the end:  How PermutationImportance works?

# Pl see this xgboost example on Titanic dataset:
#   https://eli5.readthedocs.io/en/latest/tutorials/xgboost-titanic.html#explaining-xgboost-predictions-on-the-titanic-dataset

# 7.11.1 Instantiate the importance object
perm = PermutationImportance(
                            clf,
                            random_state=1
                            )

# 7.11.2 fit data & learn
#        Takes sometime

start = time.time()
perm.fit(X_test, y_test)
end = time.time()
(end - start)/60


# 7.11.3 Conclude: Get feature weights

"""
# If you are using jupyter notebook, use:

eli5.show_weights(
                  perm,
                  feature_names = colnames      # X_test.columns.tolist()
                  )


"""

fw = eli5.explain_weights_df(
                  perm,
                  feature_names = colnames      # X_test.columns.tolist()
                  )

# 7.11.4 Print importance
fw
