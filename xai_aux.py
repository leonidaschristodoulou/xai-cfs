import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import expit
import pandas as pd
import seaborn as sns
from collections import defaultdict

import random
import string

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report,accuracy_score,precision_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import os
import tempfile
import tensorflow as tf
from tensorflow.keras.metrics import Precision # TF version <=2.15

import dice_ml
from dice_ml.utils import helpers  # For generating data metadata
from nice import NICE
from alibi.explainers import CounterfactualProto # doesn't work with TF2, so not used. 
from alibi.explainers import CounterfactualRLTabular # this does work with TF2

import warnings
import sys

from scipy.stats import mode
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer

import gurobipy as gp
from gurobipy import GRB

# Function to add feature noise
def add_feature_noise(X, noise_level):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

# Function to flip labels
def flip_labels(y, flip_fraction):
    n_flips = int(len(y) * flip_fraction)
    flip_indices = np.random.choice(len(y), n_flips, replace=False)
    y_noisy = y.copy()
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]  # Flip binary labels
    return y_noisy

# Function to add noise
def add_noise(df, num_feat, cat_feat, noise_level):
    noisy_df = df.copy()
    
    # Add noise to numerical features
    for col in num_feat:
        std_dev = np.std(df[col].astype(float))
        noise = np.random.normal(0, noise_level * std_dev, size=len(df))
        noisy_df[col] = df[col].astype(float) + noise
    
    # Add noise to categorical features
    for col in cat_feat:
        mask = np.random.rand(len(df)) < noise_level
        shuffled = np.random.permutation(df[col])
        noisy_df.loc[mask, col] = shuffled[mask]
    
    return noisy_df

def simulate_noise_effect(df, num_feat, cat_feat, y, max_noise=1.5, step=0.05, min_frequency=None):
    results = []
    noise_levels = np.arange(0, max_noise + step, step)
    label_flip_fractions = np.linspace(0, 0.2, 11) 
    
    datasets = []
    for ii, noise_level in enumerate(noise_levels):
        df_noisy = add_noise(df, num_feat, cat_feat, noise_level)
        y_noisy = flip_labels(y, label_flip_fractions[ii])
        
        pp = Pipeline([
                ('PP', ColumnTransformer([
                    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', min_frequency=min_frequency), cat_feat), 
                    ('num', StandardScaler(), num_feat)],
                    #('num', 'passthrough', num_feat)], 
                    remainder='passthrough'
                    )),
                ])

        X_train, X_test, y_train, y_test = train_test_split(df_noisy, y_noisy, test_size=0.3, random_state=42)
        clf = Pipeline([('PP', pp), ('NN', MLPClassifier(max_iter=1000))])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((noise_level, accuracy))
        datasets.append([df_noisy,y_noisy])
    return results, datasets

def get_preprocessed_numpy_array(clf, df):
    df_const = pd.DataFrame(np.insert(clf['PP'].transform(df).astype(float), 0, 1, axis=1))
    df_const.columns = df_const.columns.astype(str)
    return df_const

def get_tf_input(clf_lr, df_train, y_train, df_test):
    df_train_tf = pd.DataFrame(clf_lr['PP'].transform(df_train).astype('float32'))
    df_train_tf.columns = df_train_tf.columns.astype(str)
    df_train_tf['target'] = y_train
    df_test_tf = pd.DataFrame(clf_lr['PP'].transform(df_test).astype('float32'))
    df_test_tf.columns = df_test_tf.columns.astype(str)
    #tf_cfs = get_dice_clf_tf_cf(model, df_train_tf, df_test_tf, 'target')
    transformed_features = clf_lr[0]['PP'].get_feature_names_out()
    transformed_cat_indices = [i for i, col in enumerate(transformed_features) if col.startswith('cat')]
    transformed_num_indices = [i for i, col in enumerate(transformed_features) if col.startswith('num')]
    return df_train_tf, df_test_tf, [str(i) for i in transformed_num_indices], [str(i) for i in transformed_cat_indices]

def get_cm_results(y_test, y_pred):
    index_array = np.full(len(y_test), None, dtype=object)
    index_array[(y_test == 1) & (y_pred == 1)] = "TP"
    index_array[(y_test == 0) & (y_pred == 0)] = "TN"
    index_array[(y_test == 0) & (y_pred == 1)] = "FP"
    index_array[(y_test == 1) & (y_pred == 0)] = "FN"
    return index_array

from cfrl import generate_rl_counterfactuals

from joblib import dump, load

def fit_pymc(X,y):
    coords = {"coeffs": ['w'+str(i) for i in range(X.shape[1])]}
    with pm.Model(coords=coords) as logistic_model:

        # data containers
        X = pm.Data("X", X)
        y = pm.Data("y", y)

        # priors
        b = pm.Normal("b", mu=0, sigma=1, dims="coeffs")
        
        # linear model
        log_odds_mc = pm.math.dot(X, b)
        
        # link function
        p = pm.Deterministic("p", pm.math.invlogit(log_odds_mc))

        # likelihood
        pm.Bernoulli("obs", p=p, observed=y)

        # Sample from posterior
        idata = pm.sample(progressbar=False, draws=2000, tune=2000, chains=4, target_accept=0.95)
    return np.vstack([idata.posterior['b'][:,:,i].values.flatten() for i in range(idata.posterior['b'].shape[-1])])

class FindCF():
        
    def __init__(self,weights, X, 
                 cont_cf_vars, # 1
                 cat_cf_vars,  # 2
                 cont_no_cf_vars, # 3
                 cat_no_cf_vars,  # 4
                 polytopes=None, 
                 onehot_drop_first = False
                 ):
         
         self.X = X
         self.cont_cf_vars = cont_cf_vars #X.columns[cont_cf_ind].to_list()
         self.cat_cf_vars = cat_cf_vars #X.columns[cat_cf_ind].to_list()
         self.cont_cf_ind = X.columns.get_indexer(cont_cf_vars)
         self.cat_cf_ind = X.columns.get_indexer(cat_cf_vars)

         self.cont_no_cf_vars = cont_no_cf_vars #X.columns[cont_no_cf_ind].to_list()
         self.cat_no_cf_vars = cat_no_cf_vars #X.columns[cat_no_cf_ind].to_list()
         self.cont_no_cf_ind = X.columns.get_indexer(cont_no_cf_vars)
         self.cat_no_cf_ind = X.columns.get_indexer(cat_no_cf_vars)

         self.polytopes = polytopes
         #self.log_odds_epsilon = log_odds_epsilon
         self.w = weights
         self.onehot_drop_first = onehot_drop_first
         self.n = weights.shape[1]
    
    # Calculate the inverse MAD for continuous variables
    def inverse_mad(self,series):
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        return 1 / mad if mad != 0 else 0

    # Function to calculate weights and F values
    def calculate_parameters(self):
        self.weights_continuous = {col: self.inverse_mad(self.X[col]) for col in self.cont_cf_vars}
        k = 1.48  # Normalizing constant
        self.weights_discrete = {}
        for col in self.cat_cf_vars:
            std = np.std(self.X[col])
            self.weights_discrete[col] = k / std if std != 0 else 0
        return

    def get_min(self, x):
        w0_eff = np.mean(np.dot(x[self.cont_no_cf_vars],self.w[self.cont_no_cf_ind]) + np.dot(x[self.cat_no_cf_vars],self.w[self.cat_no_cf_ind]))
        #print(self.cont_cf_vars)
        # Initialize the Gurobi model
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.start()
            with gp.Model(env=env) as m:
                
                # Add variables
                c = m.addVars(len(self.cont_cf_vars), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="c") # lb=x[self.cont_cf_vars]
                if len(self.cat_cf_vars)!=0:
                    d = m.addVars(len(self.cat_cf_vars), vtype=GRB.BINARY, name="d")
            
                #m.addConstr(gp.quicksum(self.w[self.cont_cf_ind][i]*c[j] for i,j in enumerate(self.cont_cf_vars)) + 
                #            gp.quicksum(self.w[self.cat_cf_ind][i]*d[j] for i,j in enumerate(self.cat_cf_vars)) >= -w0_eff, 'lin_model') #cross decision boundary
                
                # Add the marginalized constraint (average of the constraints) across all columns of w
                
                if np.mean(self.w, axis=1)@x <= 0:
                    m.addConstr(
                        (
                            gp.quicksum(
                                gp.quicksum(c[i] * self.w[self.cont_cf_ind][i, j] for i in range(len(self.cont_cf_vars))) for j in range(self.n)
                            ) + 
                            gp.quicksum(
                                gp.quicksum(d[i] * self.w[self.cat_cf_ind][i, j] for i in range(len(self.cat_cf_vars))) for j in range(self.n)
                            )
                        ) / self.n + w0_eff>= 1e-3, name="marginalized_constraint"
                    )
                else:
                    m.addConstr(
                        (
                            gp.quicksum(
                                gp.quicksum(c[i] * self.w[self.cont_cf_ind][i, j] for i in range(len(self.cont_cf_vars))) for j in range(self.n)
                            ) + 
                            gp.quicksum(
                                gp.quicksum(d[i] * self.w[self.cat_cf_ind][i, j] for i in range(len(self.cat_cf_vars))) for j in range(self.n)
                            )
                        ) / self.n + w0_eff<= -1e-3, name="marginalized_constraint"
                    )

                if len(self.polytopes)!=0:
                    if self.onehot_drop_first:
                        for p in self.polytopes: # all indices pointing to binary cols
                            m.addConstr(gp.quicksum(x.iloc[p]) <= 1)
                    else:
                        for p in self.polytopes:
                            m.addConstr(gp.quicksum(x.iloc[p]) == 1)

                abs_cont = m.addVars(len(self.cont_cf_vars), lb=0, name="abs_cont") 
                abs_cat = m.addVars(len(self.cat_cf_vars), lb=0, name="abs_cat")
                
                for i,j in enumerate(self.cont_cf_ind):     
                        m.addConstr(abs_cont[i] >= c[i] - x.iloc[j]) 
                        m.addConstr(abs_cont[i] >= x.iloc[j] - c[i])
                
                if len(self.cat_cf_vars)!=0:
                    for i,j in enumerate(self.cat_cf_ind): 
                            m.addConstr(abs_cat[i] >= d[i] - x.iloc[j]) 
                            m.addConstr(abs_cat[i] >= x.iloc[j] - d[i])
                
                obj = gp.quicksum(self.weights_continuous[j] * abs_cont[i] for i,j in enumerate(self.cont_cf_vars)) + \
                      gp.quicksum(self.weights_discrete[j] * abs_cat[i] for i,j in enumerate(self.cat_cf_vars))

                m.setObjective(obj, GRB.MINIMIZE)

                # Optimize the model
                m.optimize()

                # Check if the model found an optimal solution
                if m.status == GRB.OPTIMAL:
                    return  [d[i].X for i,j in enumerate(self.cat_cf_vars)], [c[i].X for i,j in enumerate(self.cont_cf_vars)]
                else:
                    print('optimization error - returning original array')
                    #print(x)
                    #print('weights:', self.w[self.cont_cf_ind], self.w[self.cat_cf_ind],self.w[self.cont_no_cf_ind])
                    #print('w0_eff', w0_eff)
                    #sys.exit()
                    return list(x[self.cat_cf_vars].values), list(x[self.cont_cf_vars].values) #np.full([len(self.cont_cf_vars)+len(self.cat_cf_vars)], np.nan)

def get_clf_col(df_out, test_col, clf_col):
    df_out[clf_col] = 'TP'
    df_out.loc[(df_out['y_test']==1) & (df_out[test_col]==0), clf_col] = 'FN'
    df_out.loc[(df_out['y_test']==0) & (df_out[test_col]==0), clf_col] = 'TN'
    df_out.loc[(df_out['y_test']==0) & (df_out[test_col]==1), clf_col] = 'FP'
    return df_out

def get_nice_cf(clf,x_train_in, y_train, x_test, cat_feat, num_feat):
    
    def predict_fn(x):
        return clf.predict_proba(x)
        
    NICE_explainer = NICE(
        X_train=x_train_in,
        predict_fn=predict_fn,
        y_train=y_train,
        cat_feat=cat_feat,
        num_feat=num_feat,
        #distance_metric='HEOM',
        #num_normalization='minmax',
        optimization='proximity',
        #justified_cf=True
    )

    # explain an instance
    lr_nice_cf = []
    for i in range(len(x_test)):
        lr_nice_cf.append(NICE_explainer.explain(x_test[i].reshape(1,-1))[0])
    return lr_nice_cf

def get_nice_cf_tf(model,x_train, y_train, x_test, cat_feat, num_feat):
    
    def predict_fn(x):
        pred_prob_class_1 = model.predict(x,verbose=0)
        return np.hstack((1 - pred_prob_class_1, pred_prob_class_1))
            
    NICE_explainer = NICE(
        X_train=x_train,
        predict_fn=predict_fn,
        y_train=y_train,
        cat_feat=cat_feat,
        num_feat=num_feat, #np.arange(x_train_in.shape[1]).tolist(),
        distance_metric='HEOM',
        num_normalization='minmax',
        optimization='proximity',
        justified_cf=True
        )

    # explain an instance
    tf_nice_cf = []
    for i in range(len(x_test)):
        tf_nice_cf.append(NICE_explainer.explain(x_test[i:i+1]))
    return np.array(tf_nice_cf)[:,0,:]

def get_dice_clf_sklearn_cf(clf, df_train, df_test, continuous_features, outcome_name, n_dice_cfs=1):
    df_train[continuous_features] = df_train[continuous_features].astype(float) # somehow otherwise dice fails in this step 
    d_dice = dice_ml.Data(dataframe=df_train, continuous_features=continuous_features, outcome_name=outcome_name)
    m_dice = dice_ml.Model(model=clf, backend='sklearn', model_type='classifier')
    exp = dice_ml.Dice(d_dice, m_dice, method="random")
    e = exp.generate_counterfactuals(df_test, 
                                     total_CFs=n_dice_cfs, 
                                     desired_class="opposite",
                                     #features_to_vary=features_to_vary,
                                     proximity_weight=1, 
                                     diversity_weight=0.0,
                                     )
    x_test_dice_cf_base = []
    for i in range(len(df_test)):
        x_test_dice_cf_base.append(e.cf_examples_list[i].final_cfs_df.values[0])
    return np.array(x_test_dice_cf_base)[:,:-1] # we dont need to keep the last columnm which is the opposite outcome
    
def get_dice_clf_tf_cf(model, x_train, x_test, cont_feat, cat_feat, target):
    # my undertanding is that backend TF2 uses the weights gradient. Why it crushes when I pass
    # method="gradient" is a mystery. 
    # Also I'm not sure how the polytopes are respected so we need to chekc them manually
    
    data_metadata = dice_ml.Data(
        dataframe=x_train,
        continuous_features=cont_feat,
        categorical_features=cat_feat,
        outcome_name=target
    )
    data_metadata.feature_ranges = {i:[0,1] for i in cat_feat}

    exp = dice_ml.Dice(
        data_interface=data_metadata,
        model_interface=dice_ml.Model(model=model, backend="TF2", model_type='classifier')
    )
    e = exp.generate_counterfactuals(x_test, total_CFs=1, desired_class="opposite",proximity_weight=1, diversity_weight=0.0,)
    
    x_test_tf_dice_cf_base = []
    for i in range(len(x_test)):
        x_test_tf_dice_cf_base.append(e.cf_examples_list[i].final_cfs_df.values[0])
    return np.array(x_test_tf_dice_cf_base)[:,:-1] # we don't need the target column

def fun_get_false_cf_indices(y_pred, y_cf_pred, model_desc):
    mismatch = (1 - y_pred) != y_cf_pred
    mismatch_indices = np.where(mismatch)[0]  # Get the indices of mismatched elements
    good_cf_indices = ~mismatch
    print("CFs for", model_desc, "not ok")
    print("Number or wrong CFs:", len(mismatch_indices),"(", 100*len(mismatch_indices)/len(y_pred),"%)")
    return good_cf_indices # NB make sure you return both true and false

def fun_check_cf_lin_model(coeffs, x_test, cfs, model_desc):
    # returns indices
    if coeffs.ndim == 1: 
        y_pred2 = (expit(np.dot(x_test, coeffs))>0.5).astype(int)  # Shape will be (1250,)
        y_cf = (expit(np.dot(cfs, coeffs))>0.5).astype(int)
    else: 
        y_pred2 = (expit(np.dot(x_test, coeffs)).mean(axis=1)>0.5).astype(int)  # Take mean over the last axis
        y_cf = (expit(np.dot(cfs, coeffs)).mean(axis=1)>0.5).astype(int)
    if np.array_equal(1-y_pred2, y_cf):
        print("CFs for", model_desc, "ok")
        return np.full(len(y_cf), True) #np.arange(len(y_cf))
    else:
        good_cf_indices = fun_get_false_cf_indices(y_pred2, y_cf, model_desc)
        return good_cf_indices

def fun_check_cf_pred_model(model, cfs, x_test, model_desc):
    y_cf_pred = model.predict(cfs)
    y_pred = model.predict(x_test)

    if "tf" in model_desc:
        y_cf_pred = (y_cf_pred>0.5).astype(int)
        y_pred = (y_pred>0.5).astype(int)

    if np.array_equal(1-y_pred, y_cf_pred):
        print("CFs for", model_desc, "ok")
        return np.full(len(y_cf_pred), True) #np.arange(len(y_cf_pred))
    else:
        good_cf_indices = fun_get_false_cf_indices(y_pred, y_cf_pred, model_desc)
        return good_cf_indices

class ModelEvaluator:
    def __init__(self):
        # Initialize an empty dictionary to store metrics
        self.model_stats = {}

    def calculate_metrics(self, model_name, y_true, y_pred):
        """
        Calculate accuracy and precision for the given model and predictions.
        Store results in the model_stats dictionary.
        """
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')  # Adjust `average` for multiclass

        # Store metrics in the dictionary
        self.model_stats[(model_name, 'accuracy')] = accuracy
        self.model_stats[(model_name, 'precision')] = precision

    def get_metrics(self, model_name, y_true, y_pred):
        """
        Retrieve all metrics for a specific model.
        """
        # Calculate metrics and store them
        self.calculate_metrics(model_name, y_true, y_pred)

        # Retrieve metrics for the model
        return {
            'accuracy': self.model_stats.get((model_name, 'accuracy')),
            'precision': self.model_stats.get((model_name, 'precision')),
        }

class CFFunctionHandler:
    def __init__(self, df_train, y_train, df_test, df_test_const, cols, polytopes, outcome_name='target'):
        """Class defining specific functions."""
        # Initialize attributes
        self.functions = {
            ('lr', 'skl'): self.func_lr_skl, # 1
            ('lr', 'blr_marg'): self.func_lr_blr_marg, # 2
            ('lr', 'blr_mean'): self.func_lr_blr_mean, # 3
            ('lr', 'nice'): self.func_lr_nice, # 4
            ('lr', 'dice'): self.func_lr_dice, # 5
            ('lr', 'RL'): self.func_lr_rl,
            ('rf', 'nice'): self.func_rf_nice, # 6
            ('rf', 'dice'): self.func_rf_dice, # 7
            ('rf', 'RL'): self.func_rf_rl,
            ('tf', 'nice'): self.func_tf_nice, # 8
            ('tf', 'dice'): self.func_tf_dice, # 9
            #('tf', 'RL'): self.func_tf_rl,
        }
        self.df_train = df_train
        self.df_test = df_test
        self.df_test_const = df_test_const
        self.y_train = y_train
        self.cols = cols
        self.polytopes = polytopes
        self.outcome_name=outcome_name
        self.df_train_in = self.df_train.copy() # used in DiCE with sklearn models
        self.df_train_in[self.outcome_name] = y_train
        self.results = {}  # Dictionary to store results

    def func_lr_rl(self, clf, df_in, categorical_names, numerical_names, X_train, y_train, X_test, y_test):
        print('Working on lr RL')
        self.category_map = defaultdict(list)
        if len(categorical_names)>0:
            features = clf[0].named_steps['PP'].named_transformers_['cat'].get_feature_names_out()
            for feature in features:
                index, category = feature.split("_", 1)  # Split at first underscore
                self.category_map[int(index[1:])].append(category)  # Convert index to int and store
                #category_map[index].append(category)  # Convert index to int and store

            # Convert defaultdict to regular dict
            self.category_map = dict(self.category_map)
        out = generate_rl_counterfactuals(X_train, y_train, X_test, y_test, categorical_names, numerical_names, self.category_map, clf,'sk')
        return out.data['cf']['X']
    
    def func_rf_rl(self, clf, df_in, categorical_names, numerical_names, X_train, y_train, X_test, y_test):
        print('Working on rf RL')
        #category_map = {i:list(df_in[str(i)].unique()) for i in range(len(categorical_names))}
        out = generate_rl_counterfactuals(X_train, y_train, X_test, y_test, categorical_names, numerical_names, self.category_map, clf,'sk')
        return out.data['cf']['X']
    
    def func_tf_rl(self, clf, df_in, categorical_names, numerical_names, X_train, y_train, X_test, y_test):
        print('Working on tf RL')
        category_map = {i:list(df_in[str(i)].unique()) for i in range(len(categorical_names))}
        out = generate_rl_counterfactuals(X_train, y_train, X_test, y_test, categorical_names, numerical_names, category_map, clf,'tf')
        return out.data['cf']['X']

    def func_lr_skl(self, clf):
        print('Working on lr skl')
        #transformed_features = cf_calc.clf_lr[0]['PP'].get_feature_names_out()
        transformed_features = clf[0]['PP'].get_feature_names_out()
        transformed_cat_indices = [i for i, col in enumerate(transformed_features) if col.startswith('cat')]
        transformed_num_indices = [i for i, col in enumerate(transformed_features) if col.startswith('num')]
        self.cat_cols = [str(x + 1) for x in transformed_cat_indices]
        self.num_cols = [str(x + 1) for x in transformed_num_indices]
        lr_coeffs = np.insert(clf['LR'].coef_, 0, clf['LR'].intercept_[0]) # adding the constant here for CF evaluation
        fcf = FindCF(
            lr_coeffs.reshape(-1, 1), 
            self.df_test_const, 
            self.num_cols, 
            self.cat_cols, 
            ['0'], 
            [], 
            polytopes=self.polytopes, #[self.cols[1:4], self.cols[4:7]], 
            onehot_drop_first=True
        )
        fcf.calculate_parameters()
        x_test_lr_cf = []
        for j in range(len(self.df_test_const)):  # Assuming linear access to rows
            cf = fcf.get_min(self.df_test_const.iloc[j])
            x_test_lr_cf.append(np.array(cf[0] + cf[1])) # Our CF function returns the results like that
        x_test_lr_cf = np.array(x_test_lr_cf)
        return x_test_lr_cf

    def func_lr_blr_marg(self, samples):
        print('Working on lr blr_marg')
        fcf = FindCF(
            samples, 
            self.df_test_const, 
            self.num_cols, 
            self.cat_cols, 
            ['0'], 
            [], 
            polytopes=self.polytopes, #[self.cols[1:4],self.cols[4:7]], 
            onehot_drop_first=True)
        fcf.calculate_parameters()
        x_test_mcmc_marg_cf = []
        for j in range(len(self.df_test_const)):
            #x_test_mcmc_marg_cf_base.append(optimize_x_prime(x_test_in[j][1:], samples))
            cf = fcf.get_min(self.df_test_const.iloc[j])
            x_test_mcmc_marg_cf.append(np.array(cf[0] + cf[1]))
        x_test_mcmc_marg_cf = np.array(x_test_mcmc_marg_cf)
        return x_test_mcmc_marg_cf  # Replace with actual logic
    
    def func_lr_blr_mean(self, samples):
        print('Working on lr blr_mean')
        fcf = FindCF(
            np.mean(samples,axis=1).reshape(-1,1), 
            self.df_test_const, 
            self.num_cols, 
            self.cat_cols, 
            ['0'], 
            [], 
            polytopes=self.polytopes, #[self.cols[1:4], self.cols[4:7]], 
            onehot_drop_first=True)
        fcf.calculate_parameters()
        x_test_mcmc_mean_cf = []
        for j in range(len(self.df_test_const)):
            #x_test_mcmc_mean_cf_base.append(optimize_x_prime(x_test_in[j][1:], np.mean(samples,axis=1)))
            cf = fcf.get_min(self.df_test_const.iloc[j])
            x_test_mcmc_mean_cf.append(np.array(cf[0] + cf[1]))
        x_test_mcmc_mean_cf = np.array(x_test_mcmc_mean_cf)
        return x_test_mcmc_mean_cf  # Replace with actual logic

    def func_lr_nice(self, clf, x_train, y_train, x_test, cat_cols, num_cols):
        print('Working on lr NICE')
        cfs = get_nice_cf(clf, x_train, y_train, x_test, cat_cols, num_cols)
        #fun_check_nice(clf, cfs, y_test, "lr nice")
        return cfs
    
    def func_lr_dice(self, clf, num_cols, n_dice_cfs=1):
        print('Working on lr DiCE')
        return get_dice_clf_sklearn_cf(clf,self.df_train_in, self.df_test, num_cols, self.outcome_name, n_dice_cfs=1)

    def func_rf_nice(self, clf, x_train, y_train, x_test, cat_cols, num_cols):
        print('Working on rf NICE')
        return get_nice_cf(clf, x_train, y_train, x_test, cat_cols, num_cols)

    def func_rf_dice(self, clf, num_cols, n_dice_cfs=1):
        print('Working on rf DiCE')
        return get_dice_clf_sklearn_cf(clf, self.df_train_in, self.df_test, num_cols, self.outcome_name, n_dice_cfs=1)
    
    def func_tf_nice(self, model, x_train, y_train, x_test, cat_cols, num_cols):
        print('Working on tf NICE')
        return get_nice_cf_tf(model, x_train, y_train, x_test, cat_cols, num_cols)

    def func_tf_dice(self, model, x_train, x_test, num_cols, cat_cols):
        print('Working on tf DiCE')
        return get_dice_clf_tf_cf(model, x_train, x_test, num_cols, cat_cols, self.outcome_name)

    def execute(self, key, value, **kwargs):
        """Call the appropriate function based on (key, value)."""
        func = self.functions.get((key, value))
        if func is not None:
            result = func(**kwargs)
            self.results[(key, value)] = result  # Store the result
            return result
        else:
            raise ValueError(f"No function defined for ({key}, {value})")

class CLFModels():
    def __init__(self, df_train, y_train, df_test, y_test, cat_feat, num_feat, min_frequency=None):
        self.df_train = df_train
        self.y_train = y_train
        self.df_test = df_test
        self.y_test = y_test
        self.cat_feat = cat_feat #[0, 1, 2, 3, 4]
        self.num_feat = num_feat #[5, 6, 7, 8, 9]
        self.cm_data = {}
        self.min_frequency = min_frequency
        
    def get_models(self):
        pp = Pipeline([
            ('PP', ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False, min_frequency=self.min_frequency), self.cat_feat), 
                ('num', StandardScaler(), self.num_feat)], 
                #('num', 'passthrough', self.num_feat)], 
                remainder='passthrough'
                )),
            ])
        evaluator = ModelEvaluator()

        # LR
        self.clf_lr = Pipeline([
            ('PP', pp),  # Add preprocessing as the first step
            ('LR', LogisticRegression(max_iter=1000))  # Add the model as the second step
        ])
        self.clf_lr.fit(self.df_train, self.y_train)
        y_pred = self.clf_lr.predict(self.df_test) 
        evaluator.get_metrics('lr', self.y_test, y_pred)
        self.cm_data['lr'] = get_cm_results(self.y_test, y_pred)
        self.df_test_const = get_preprocessed_numpy_array(self.clf_lr, self.df_test)

        pp_np = Pipeline([
            ('PP', ColumnTransformer([
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=self.min_frequency), [int(i) for i in self.cat_feat]),  #, drop='first'
                ('num', StandardScaler(), [int(i) for i in self.num_feat])], 
                #('num', 'passthrough', [int(i) for i in self.num_feat])], 
                remainder='passthrough'
                )),
            ])
            
        self.clf_lr_np = Pipeline([('PP', pp_np), ('clf', LogisticRegression(max_iter=1000))]) # numpy pipeline
        self.clf_lr_np.fit(self.df_train.values, self.y_train)

        # PYMC
        self.samples = fit_pymc(get_preprocessed_numpy_array(self.clf_lr, self.df_train).values,self.y_train)
        predicted_probabilities = expit(np.dot(self.df_test_const.values, self.samples))
        p_test_pred = predicted_probabilities.mean(axis=1)
        y_pred = (p_test_pred >= 0.5).astype("int")
        self.cm_data['blr'] = get_cm_results(self.y_test, y_pred)
        evaluator.get_metrics('blr', self.y_test, y_pred)

        # RF
        self.clf_rf = Pipeline([
            ('PP', pp),  # Add preprocessing as the first step
            ('RF', RandomForestClassifier())  # Add the model as the second step
        ])
        self.clf_rf.fit(self.df_train, self.y_train)
        y_pred = self.clf_rf.predict(self.df_test)
        self.cm_data['rf'] = get_cm_results(self.y_test, y_pred)
        evaluator.get_metrics('rf', self.y_test, y_pred)

        self.clf_rf_np = Pipeline([('PP', pp_np), ('clf', RandomForestClassifier())])
        self.clf_rf_np.fit(self.df_train.values, self.y_train)

        # TF (we couldn't get gradient method to produce CFs)
        df_train_tf, df_test_tf, num_cols_tf, cat_cols_tf = get_tf_input(self.clf_lr, self.df_train, self.y_train, self.df_test) # For DiCE later
        train_tf = self.clf_lr['PP'].transform(self.df_train).astype(float)
        train_tf, val_tf, y_train_tf, y_val_tf = train_test_split(train_tf, self.y_train, test_size=0.75)
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(12, input_dim=train_tf.shape[1], activation='relu'))
        self.model.add(tf.keras.layers.Dense(8, activation='relu'))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',Precision()])
        self.model.fit(train_tf, y_train_tf, epochs=50, batch_size=10, validation_data=(val_tf, y_val_tf), verbose=0)
        loss, accuracy, precision = self.model.evaluate(self.clf_lr['PP'].transform(self.df_test).astype(float), self.y_test)
        print(f"TF Test Accuracy: {accuracy * 100:.2f}%")
        y_pred = (self.model.predict(self.clf_lr['PP'].transform(self.df_test).astype(float))>0.5).astype(int).reshape(1,-1)[0]
        self.cm_data['tf'] = get_cm_results(self.y_test, y_pred)
        evaluator.get_metrics('tf', self.y_test, y_pred)
        return evaluator.model_stats, (self.clf_lr, self.samples, self.clf_rf, self.model), self.cm_data


class CFCalculations(CLFModels):
    def __init__(self, df_train, y_train, df_test, y_test, cat_feat, num_feat, polytopes, do_marg=True, do_dice=True, min_frequency=None):
        super().__init__(df_train, y_train, df_test, y_test, cat_feat, num_feat, min_frequency=None)  # Initialize the base class
        self.model_stats, self.models, self.cms = self.get_models()  # Train and retrieve models and associated data
        self.cat_feat = cat_feat
        self.num_feat = num_feat
        self.polytopes = polytopes
        self.do_marg = do_marg
        self.do_dice = do_dice
        self.min_frequency = min_frequency

    def get_cfs(self):
        # Initialize the counterfactual handler
        handler = CFFunctionHandler(
            self.df_train,
            self.y_train,
            self.df_test,
            self.df_test_const,
            self.df_test_const.columns.to_list(),
            self.polytopes,
        )

        self.df_train_tf, self.df_test_tf, self.num_cols_tf, self.cat_cols_tf = get_tf_input(self.clf_lr, self.df_train, self.y_train, self.df_test)
        
        # Execute counterfactual methods for logistic regression
        handler.execute('lr', 'skl', clf=self.clf_lr)

        if self.do_marg:
            handler.execute('lr', 'blr_marg', samples=self.samples)

        handler.execute('lr', 'blr_mean', samples=self.samples)

        handler.execute(
            'lr', 'nice',
            clf=self.clf_lr,
            x_train=self.df_train.values,
            y_train=self.y_train,
            x_test=self.df_test.values,
            cat_cols=self.cat_feat, #[i for i in range(5)], #[i for i in range(len(self.cat_feat))],
            num_cols=self.num_feat #[i for i in range(5,10)] #[i for i in range(len(self.num_feat))]
        )

        if self.do_dice:
            handler.execute(
                'lr', 'dice',
                clf=self.clf_lr,
                num_cols=[str(i) for i in self.num_feat] #[str(i) for i in range(5,10)]
            )

        # , clf, df_in, categorical_names, numerical_names, X_train, y_train, X_test, y_test
        handler.execute(
            'lr', 'RL',
            clf=self.clf_lr_np,
            df_in=self.df_train, # the category map..
            categorical_names=self.cat_feat,
            numerical_names=self.num_feat,
            X_train=self.df_train.values, 
            y_train=self.y_train, 
            X_test=self.df_test.values, 
            y_test=self.y_test
            #num_cols=[str(i) for i in self.num_feat] #[str(i) for i in range(5,10)]
        )

        # Execute counterfactual methods for random forest
        handler.execute(
            'rf', 'nice',
            clf=self.clf_rf,
            x_train=self.df_train.values,
            y_train=self.y_train,
            x_test=self.df_test.values,
            cat_cols = self.cat_feat, #[i for i in range(5)],
            num_cols = self.num_feat #[i for i in range(5,10)]
        )
        
        if self.do_dice:
            handler.execute(
                'rf', 'dice',
                clf=self.clf_rf,
                num_cols=[str(i) for i in self.num_feat] #[str(i) for i in range(5,10)]
            )

        handler.execute(
            'rf', 'RL',
            clf=self.clf_rf_np,
            df_in=self.df_train, # the category map..
            categorical_names=self.cat_feat,
            numerical_names=self.num_feat,
            X_train=self.df_train.values, 
            y_train=self.y_train, 
            X_test=self.df_test.values, 
            y_test=self.y_test
            #num_cols=[str(i) for i in self.num_feat] #[str(i) for i in range(5,10)]
        )

        # Execute counterfactual methods for TensorFlow
        handler.execute(
            'tf', 'nice',
            model=self.model,
            x_train=self.clf_lr['PP'].transform(self.df_train).astype('float32'),
            y_train=self.y_train,
            x_test=self.clf_lr['PP'].transform(self.df_test).astype('float32'),
            cat_cols = self.cat_feat, #[i for i in range(5)],
            num_cols = self.num_feat #[i for i in range(5,10)]
        )

        if self.do_dice:
            handler.execute(
                'tf', 'dice',
                model=self.model,
                x_train=self.df_train_tf,
                x_test=self.df_test_tf,
                num_cols=self.num_cols_tf,
                cat_cols=self.cat_cols_tf
            )

        # handler.execute(
        #     'tf', 'RL',
        #     clf=self.model,
        #     df_in=self.df_train, # the category map..
        #     categorical_names=self.cat_feat,
        #     numerical_names=self.num_feat,
        #     X_train=self.df_train.values, 
        #     y_train=self.df_test_tf, 
        #     X_test=self.df_test.values,  
        #     y_test=self.y_test
        # )

        self.counterfactual_results = handler.results
        print("Counterfactuals calculated")
        return handler.results

    def check_cfs(self):
        def process_linear_model(key, model_data, results_key, description):

            # Generalized processing for linear models
            ns = self.counterfactual_results[results_key].shape[0]

            self.good_cf_indices[key] = fun_check_cf_lin_model(
                model_data, 
                self.df_test_const, 
                np.hstack((np.ones((ns, 1)), self.counterfactual_results[results_key])),
                description
            )
            
            self.groups = [[j-1 for j in i] for i in self.polytopes] #[[0, 1, 2], [3, 4, 5]]
            
            if len(self.groups)==0:
                polytope_conditions = [np.ones(self.counterfactual_results[results_key].shape[0], dtype=bool)]
            else:
                polytope_conditions = [
                    self.counterfactual_results[results_key][:, group[0]:group[-1] + 1].sum(axis=1) <= 1
                    for group in self.groups
                    ]
            polytope_conditions = np.array(polytope_conditions)
            self.good_cf_indices[key] = np.logical_and.reduce([self.good_cf_indices[key], np.all(polytope_conditions, axis=0)]) #np.logical_and(self.good_cf_indices[key], polytope_conditions)
            
        def process_pred_model(key, clf, results_key, df, description):
            # Generalized processing for predictive models
            self.good_cf_indices[key] = fun_check_cf_pred_model(
                clf, 
                self.counterfactual_results[results_key], 
                df, 
                description
            )

            ar_cf = clf['PP'].transform(pd.DataFrame(self.counterfactual_results[results_key], columns=df.columns))
            
            if len(self.groups)==0:
                polytope_conditions = [np.ones(ar_cf.shape[0], dtype=bool)]
            else:
                polytope_conditions = [
                    ar_cf[:, group[0]:group[-1] + 1].sum(axis=1) <= 1
                    for group in self.groups
                ]
            polytope_conditions = np.array(polytope_conditions)
            # Combine the initial condition with the generated conditions
            self.good_cf_indices[key] = np.logical_and.reduce([self.good_cf_indices[key], np.all(polytope_conditions, axis=0)]) #np.logical_and([self.good_cf_indices[key]], polytope_conditions)

        def process_pred_model_tfd(key, tf_model, skl_model, results_key, df, description):
            # Generalized processing for TensorFlow models with sklearn transform pipeline
            self.good_cf_indices[key] = fun_check_cf_pred_model(
                tf_model, 
                self.counterfactual_results[results_key], 
                df, 
                description
            )

            if len(self.groups)==0:
                polytope_conditions = [np.ones(df.values.shape[0], dtype=bool)]
            else:
                polytope_conditions = [
                    df.values[:, group[0]:group[-1] + 1].sum(axis=1) <= 1
                    for group in self.groups
                ]
            polytope_conditions = np.array(polytope_conditions)
            self.good_cf_indices[key] = np.logical_and.reduce([self.good_cf_indices[key], np.all(polytope_conditions, axis=0)]) #np.logical_and(self.good_cf_indices[key], polytope_conditions)

        print("Checking CFs...")
        self.good_cf_indices = {}
        # Linear models
        process_linear_model(('lr', 'skl'), 
                            np.insert(self.clf_lr[1].coef_, 0, self.clf_lr[1].intercept_[0]), 
                            ('lr', 'skl'), 
                            'linear model')

        if self.do_marg:
            process_linear_model(('lr', 'blr_marg'),
                                self.samples, 
                                ('lr', 'blr_marg'), 
                                'bayesian linear model')

        process_linear_model(('lr', 'blr_mean'), 
                             self.samples.mean(axis=1), 
                             ('lr', 'blr_mean'), 
                             'bayesian mean linear model')

        # Predictive models
        process_pred_model(('lr', 'nice'), 
                           self.clf_lr, 
                           ('lr', 'nice'), 
                           self.df_test, 
                           'nice linear model')

        if self.do_dice:
            process_pred_model(('lr', 'dice'), 
                            self.clf_lr, 
                            ('lr', 'dice'), 
                            self.df_test, 
                            'dice linear model')

        process_pred_model(('lr', 'RL'), 
                        self.clf_lr_np, 
                        ('lr', 'RL'), 
                        self.df_test, 
                        'RL linear model')

        process_pred_model(('rf', 'nice'), 
                        self.clf_rf, 
                        ('rf', 'nice'), 
                        self.df_test, 
                        'nice rf model')

        if self.do_dice:
            process_pred_model(('rf', 'dice'), 
                            self.clf_rf, 
                            ('rf', 'dice'), 
                            self.df_test, 
                            'dice rf model')
        
        process_pred_model(('rf', 'RL'), 
                        self.clf_rf_np, 
                        ('rf', 'RL'), 
                        self.df_test, 
                        'RL rf model')

        # TensorFlow models with sklearn transform pipeline
        process_pred_model_tfd(('tf', 'nice'), 
                            self.model, 
                            self.clf_lr, 
                            ('tf', 'nice'), 
                            self.df_test_tf, 
                            'nice tf model')

        if self.do_dice:
            process_pred_model_tfd(('tf', 'dice'), 
                                self.model, 
                                self.clf_lr, 
                                ('tf', 'dice'), 
                                self.df_test_tf, 
                                'dice tf model')

        print("CFs checked")
        return self.good_cf_indices

