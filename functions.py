import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_auc_score, mean_squared_error, mean_absolute_error
from sklearn.utils import resample
from sklearn.tree._tree import TREE_LEAF
import math
import sklearn.tree as tree
import pydotplus
import os
import matplotlib.pyplot as pyplot






columns_reg = {"Prog1_beste": ["Abinote","SKMat_1","SKMat_2","SKMat_3","SKMat_4","mean_SKMat","SKInf_1", "mean_SKInf",
                               "Ktyp_exp", "BFI_K_3", "mean_BFI_K_G", "BM_Inf_17","Prog1_beste"],
               "Prog1_scaled": ["Abinote","SKMat_1","SKMat_2","SKMat_3","SKMat_4","mean_SKMat","SKInf_1",
                               "Ktyp_exp", "BFI_K_3", "mean_BFI_K_G", "BM_Inf_17","Prog1_scaled"],
               "MfI1_beste": ["Abinote","SKMat_1","SKMat_2","SKMat_3","SKMat_4","mean_SKMat","SKInf_1", "SKInf_4",
                              "mean_SKInf", "Kurs_Inf", "Ktyp_exp", "BFI_K_7", "BM_Inf_14","LMI_3","LMI_6","MfI1_beste"],
               "beste": ["Abinote","SKMat_1","SKMat_2","SKMat_3","SKMat_4","mean_SKMat","SKInf_1","mean_SKInf",
                         "Ktyp_exp", "BFI_K_3","mean_BFI_K_G","BM_Inf_17","LMI_3","beste"]}

def load_data_bestanden(LABEL,other1,other2,other3):
    """Loads data for scenarios 4. - 7.
    
    Parameters
    ----------
    LABEL : str
        Name of the column that shall be considered as label, e.g. "Prog1_beste"
    other1 : str
        Name of label of first other scenario, e.g. "MfI1_beste"
    other2: str
        Name of label of second other scenario, e.g. "beste"
    other3: str
        Name of label of third other scenario, e.g. "beide"
    Returns
    -------
    (DataFrame,DataFrame,Series,DataFrame,DataFrame,Series,Series,Index,Index,int,int)
        Whole dataset, Whole dataset without labels, All labels, Train data without labels, Test data without labels, Train labels, Test labels, Names of all columns, Names of all attributes, Number of columns, Number of rows
    """
    
    data = pd.read_csv("fragebogen/melted_bestanden.csv").query('%s != 600'%(LABEL)).query('%s == %s'%(LABEL,LABEL))

    del data["Note_MfI1_20182_T01"]
    del data["Note_MfI1_20182_T02"]
    del data["Note_Prog1_20182_T01"]
    del data["Note_Prog1_20182_T02"]
    del data["%s"%(other1)]
    del data["%s"%(other2)]
    del data["%s"%(other3)]
    if LABEL != "MfI1_beste" and LABEL != "beide":
        del data["Std_Inf"]

    y = copy.deepcopy(data["%s"%(LABEL)])
    X = copy.deepcopy(data)
    del X["%s"%(LABEL)]
    names = data.columns
    feature_names = data.columns[:-1]
    numcols = len(data.columns)
    numrows = len(data.index)
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=42)
    return data,X,y,train_X,test_X,train_y,test_y,names,feature_names,numcols,numrows

def load_data_noten(label,other1,other2,other3=""):
    """Loads data for scenarios 1. - 3.
    
    Parameters
    ----------
    LABEL : str
        Name of the column that shall be considered as label, e.g. "Prog1_beste"
    other1 : str
        Name of label of first other scenario, e.g. "MfI1_beste"
    other2: str
        Name of label of second other scenario, e.g. "beste"
    other3: str, optional
        Name of label of third other scenario, e.g. "Prog1_scaled", 
        only used when dealing with normalized Prog1-scenario, (default "")
    Returns
    -------
    (DataFrame,DataFrame,Series,DataFrame,DataFrame,Series,Series,Index,Index,int,int)
        Whole dataset, Whole dataset without labels, All labels, Train data without labels, Test data without labels, Train labels, Test labels, Names of all columns, Names of all attributes, Number of columns, Number of rows
    """
    if label == "Prog1_scaled":
        data = pd.read_csv("fragebogen/melted_scaled.csv").query('%s != 600'%(label)).query('Prog1_beste != 600').query('%s == %s'%(label,label))
        del data["%s"%(other3)]
        del data["Prog1_normal"]       
    else:
        data = pd.read_csv("fragebogen/melted.csv").query('%s != 600'%(label)).query('%s == %s'%(label,label))
    del data["Note_MfI1_20182_T01"]
    del data["Note_MfI1_20182_T02"]
    del data["Note_Prog1_20182_T01"]
    del data["Note_Prog1_20182_T02"]
    del data["%s"%(other1)]
    del data["%s"%(other2)]
    if label != "MfI1_beste":
        del data["Std_Inf"]
    y = copy.deepcopy(data["%s"%(label)])
    X = copy.deepcopy(data)
    del X["%s"%(label)]
    names = data.columns
    feature_names = data.columns[:-1]
    numcols = len(data.columns)
    numrows = len(data.index)
    train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.3,random_state=42)
  
    return data,X,y,train_X,test_X,train_y,test_y,names,feature_names,numcols,numrows


def load_data_bestanden_raw(LABEL,other1,other2,other3, correlation=False):
    """Loads data for scenarios 4. - 7. when post-pruning is used.
    
    Returns NumPy arrays instead of DataFrames. Used for compatibility issues.
    
    Parameters
    ----------
    LABEL : str
        Name of the column that shall be considered as label, e.g. "Prog1_beste"
    other1 : str
        Name of label of first other scenario, e.g. "MfI1_beste"
    other2: str
        Name of label of second other scenario, e.g. "beste"
    other3: str
        Name of label of third other scenario, e.g. "beide"
    correlation: bool, optional
        Flag to indicate whether attributes with highest correlations shall be used (default False)
    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, Index, Index)
        Whole dataset, Train dataset, Test dataset, Validation dataset, Names of all columns, Names of all attributes
    """

    data_raw = pd.read_csv("fragebogen/melted_bestanden.csv").query('%s != 600'%(LABEL)).query('%s == %s'%(LABEL,LABEL))

    del data_raw["Note_MfI1_20182_T01"]
    del data_raw["Note_MfI1_20182_T02"]
    del data_raw["Note_Prog1_20182_T01"]
    del data_raw["Note_Prog1_20182_T02"]
    del data_raw["%s"%(other1)]
    del data_raw["%s"%(other2)]
    del data_raw["%s"%(other3)]
    y = data_raw["%s"%(LABEL)]
    if LABEL != "MfI1_beste" and LABEL != "beide":
        del data_raw["Std_Inf"]

    names = data_raw.columns
    feature_names = data_raw.columns[:-1]
    
    if correlation:
        columns_corr = [data_raw.columns.get_loc(c) for c in columns_reg["%s"%(LABEL)] if c in data_raw]
        data_raw = data_raw.values
        data_raw = data_raw[:,[columns_corr]][:,0]
        names = columns_reg["%s"%(LABEL)]
        feature_names = columns_reg["%s"%(LABEL)][:-1]
    else:
        data_raw = data_raw.to_numpy()
        
    n_size = int(len(data_raw) * 0.70)
    train_raw = resample(data_raw, n_samples=n_size, random_state=42)
    test_raw = np.array([x for x in data_raw if x.tolist() not in train_raw.tolist()])
    
    n_size = int(len(train_raw) * 0.30)
    validation_raw = resample(train_raw, n_samples=n_size, random_state=42)
    train_raw = np.array([x for x in train_raw if x.tolist() not in validation_raw.tolist()])
    return data_raw,train_raw,test_raw,validation_raw,names,feature_names

def load_data_noten_raw(LABEL,other1,other2,other3="",correlation=False):
    """Loads data for scenarios 1. - 3. when post-pruning is used.
    
    Returns NumPy arrays instead of DataFrames. Used for compatibility issues.
    
    Parameters
    ----------
    LABEL : str
        Name of the column that shall be considered as label, e.g. "Prog1_beste"
    other1 : str
        Name of label of first other scenario, e.g. "MfI1_beste"
    other2: str
        Name of label of second other scenario, e.g. "beste"
    other3: str, optional
        Name of label of third other scenario, e.g. "Prog1_scaled", 
        only used when dealing with normalized Prog1-scenario, (default "")
    correlation: bool, optional
        Flag to indicate whether attributes with highest correlations shall be used (default False)
    Returns
    -------
    (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, Index, Index)
        Whole dataset, Train dataset, Test dataset, Validation dataset, Names of all columns, Names of all attributes
    """
    if LABEL == "Prog1_scaled":
        data_raw = pd.read_csv("fragebogen/melted_scaled.csv").query('%s != 600'%(LABEL)).query('Prog1_beste != 600').query('%s == %s'%(LABEL,LABEL))
        del data_raw["%s"%(other3)]
        del data_raw["Prog1_normal"]       
    else:
        data_raw = pd.read_csv("fragebogen/melted.csv").query('%s != 600'%(LABEL)).query('%s == %s'%(LABEL,LABEL))

    del data_raw["Note_MfI1_20182_T01"]
    del data_raw["Note_MfI1_20182_T02"]
    del data_raw["Note_Prog1_20182_T01"]
    del data_raw["Note_Prog1_20182_T02"]
    del data_raw["%s"%(other1)]
    del data_raw["%s"%(other2)]
    y = data_raw["%s"%(LABEL)]
    if LABEL != "MfI1_beste":
        del data_raw["Std_Inf"]
    names = data_raw.columns
    feature_names = data_raw.columns[:-1]
    if correlation:
        columns_corr = [data_raw.columns.get_loc(c) for c in columns_reg["%s"%(LABEL)] if c in data_raw]
        data_raw = data_raw.values
        data_raw = data_raw[:,[columns_corr]][:,0]
        names = columns_reg["%s"%(LABEL)]
        feature_names = columns_reg["%s"%(LABEL)][:-1]
    else:
        data_raw = data_raw.to_numpy()
        
    n_size = int(len(data_raw) * 0.70)
    train_raw = resample(data_raw, n_samples=n_size, random_state=42)
    test_raw = np.array([x for x in data_raw if x.tolist() not in train_raw.tolist()])
    
    n_size = int(len(train_raw) * 0.30)
    validation_raw = resample(train_raw, n_samples=n_size, random_state=42)
    train_raw = np.array([x for x in train_raw if x.tolist() not in validation_raw.tolist()])
    
    
    return data_raw,train_raw,test_raw,validation_raw,names,feature_names



def display_scores(scores):
    """Shows scores, mean of scores and standard deviations.
        
    Parameters
    ----------
    scores : numpy.ndarray
        Array of achieved scores
    """
    
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
def cross_val(model, X, y, scoring, cv=5):
    """Calculates Cross-Validations scores and displays them.
    
    Parameters
    ----------
    model : DecisionTreeClassifier or DecisionTreeRegressor
        The model to evaluate
    X : DataFrame
        The dataset on which Cross-Validation shall be performed
    y: Series
        The corresponding labels
    scoring: str
        Name of performance measure that shall be used
    cv: int, optional
        Number of folds (default 5)
    """
    
    reg_scores = cross_val_score(model, X, y, scoring=scoring, cv=cv)
    if(scoring=="neg_mean_squared_error"):
        reg_scores = np.sqrt(-reg_scores)
    display_scores(reg_scores)
    
    
    
    
##### Hyperparameter Tuning #####


def bootstrap_score(data_raw, model,scoring, n_iterations=1000, train_size=0.7):
    """Calculates bootstrapped 95% confidence intervals for given performance measure.

    Parameters
    ----------
    data_raw : numpy.ndarray
        The dataset
    model: DecisionTreeClassifier or DecisionTreeRegressor
        The model
    scoring: str
        Performance measure to be used
    n_iterations: int, optional
        Number of bootstrap iterations (default 1000)
    train_size: float
        Percentage of data that shall be used as train data
    """
    
    size = int(len(data_raw) * train_size)

    stats = list()
    for i in range(n_iterations):
        if i % 100 == 0:
            print(i)
        #Split data
        train = resample(data_raw, n_samples=size)
        test = np.array([x for x in data_raw if x.tolist() not in train.tolist()])

        model.fit(train[:,:-1], train[:,-1])

        predictions = model.predict(test[:,:-1])
        if scoring=="mean_squared_error":
            score = math.sqrt(mean_squared_error(test[:,-1], predictions))
        elif scoring=="mean_absolute_error":
            score = mean_absolute_error(test[:,-1], predictions)
        elif scoring=="accuracy":
            score = accuracy_score(test[:,-1], predictions)
        elif scoring=="f1":
            score = f1_score(test[:,-1], predictions)
        elif scoring=="roc_auc":
            score = roc_auc_score(test[:,-1], predictions)
        else: 
            print("invalid scoring parameter")
        #print(score)
        stats.append(score)
    
    pyplot.hist(stats)
    pyplot.show()
    #Calculate confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    if (scoring == "mean_squared_error" or scoring == "mean_absolute_error"):
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(500.0, np.percentile(stats, p))
        print('%.1f%% confidence interval from %.4f to %.4f' % (alpha*100, lower, upper))
    elif (scoring == "accuracy" or scoring == "f1"):
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        print('%.1f%% confidence interval from %.2f%% to %.2f%%' % (alpha*100, lower*100, upper*100))
    else:    
        lower = max(0.0, np.percentile(stats, p))
        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = min(1.0, np.percentile(stats, p))
        print('%.1f%% confidence interval from %.4f to %.4f' % (alpha*100, lower, upper))
    
##### Confidence #####

def plot_decision_path(dt, sample, train_X):
    """Renders the decision tree and highlights the decision path of given sample.
        
    Parameters
    ----------
    dt : DecisionTreeClassifier or DecisionTreeRegressor
        The model of the decision tree
    sample : numpy.ndarray
        Array of all attribute values representing the sample
    train_X: DataFrame
        Train data without labels
        
    Returns
    -------
    pydotplus.graphviz.Dot
        Dot representation of the rendered graph
    """
    dot = tree.export_graphviz(dt, out_file=None,
                                feature_names=train_X.columns,
                                filled=True, rounded=True,
                                special_characters=True, node_ids = True)
    graph = pydotplus.graph_from_dot_data(dot)

    for node in graph.get_node_list():
        node.set_fillcolor('white')

    decision_path = dt.decision_path(sample)
    
    for n, node_value in enumerate(decision_path.toarray()[0]):
        if node_value == 0:
            continue
        node = graph.get_node(str(n))[0]        
        node.set_fillcolor('green')

    return graph

#######
##### Pruning #####
#######

def is_leaf(inner_tree, index):
    """Checks if given inner tree is a leaf node.
    Parameters
    ----------
    inner_tree : sklearn.tree._tree.Tree
        (Root node of) inner tree that should be checked
    index : int
        Index of root node of the inner tree in the whole tree
        
    Returns
    -------
    bool
        True iff inner_tree is leaf node
    """
    
    return (inner_tree.children_left[index] == TREE_LEAF and 
            inner_tree.children_right[index] == TREE_LEAF)

def print_results_class(dt,train_raw, test_raw, validation_raw, label,when="vor",noten=False):
    """Prints results of Classfication on train, test and validation data.
    Parameters
    ----------
    dt: DecisionTreeClassifier or DecisionTreeRegressor
        The model to evaluate
    train_raw: numpy.ndarray
        Train data
    test_raw: numpy.ndarray
        Test data
    validation_raw: numpy.ndarray
        Validation data
    label: str
        Name of label
    when : String, optional
        Flag to indicate if model is already pruned (default 'vor')
    noten: bool, optional
        Flag to indicate if marks or passed/not passed shall be considered (default False)
    """
    
    preds_train = dt.predict(train_raw[:,:-1])
    print(confusion_matrix(train_raw[:,-1], preds_train))
    accuracy = accuracy_score(train_raw[:,-1], preds_train)
    print("ACC %s Pruning Training: "%(when), accuracy)
    if label != "beide" and not noten:
        probs_train = dt.predict_proba(train_raw[:,:-1])
        probs_train  = probs_train[:, 1]
        auc = roc_auc_score(train_raw[:,-1],probs_train)
        f1 = f1_score(train_raw[:,-1], preds_train)
        print("AUC %s Pruning Training: "%(when),auc)
        print("F1 %s Pruning Training: "%(when),f1)
    
    preds_test = dt.predict(test_raw[:,:-1])
    print(confusion_matrix(test_raw[:,-1], preds_test))
    accuracy = accuracy_score(test_raw[:,-1], preds_test)
    print("ACC %s Pruning Test: "%(when), accuracy)
    if label != "beide" and not noten:
        probs_test = dt.predict_proba(test_raw[:,:-1])
        probs_test  = probs_test[:, 1]
        auc = roc_auc_score(test_raw[:,-1],probs_test)
        f1 = f1_score(test_raw[:,-1], preds_test)
        print("AUC %s Pruning Test: "%(when),auc)
        print("F1 %s Pruning Test: "%(when),f1)
    
    preds_val = dt.predict(validation_raw[:,:-1])
    print(confusion_matrix(validation_raw[:,-1], preds_val))
    accuracy = accuracy_score(validation_raw[:,-1], preds_val)
    print("ACC %s Pruning Validation: "%(when), accuracy)
    if label != "beide" and not noten:
        probs_val = dt.predict_proba(validation_raw[:,:-1])
        probs_val  = probs_val[:, 1]
        auc = roc_auc_score(validation_raw[:,-1],probs_val)    
        f1 = f1_score(validation_raw[:,-1], preds_val)
        print("AUC %s Pruning Validation: "%(when),auc)
        print("F1 %s Pruning Validation: "%(when),f1)
        
        
def print_results_reg(dt,train_raw, test_raw, validation_raw,when="vor"):
    """Prints results of Regression on train, test and validation data.
    Parameters
    ----------
    dt: DecisionTreeClassifier or DecisionTreeRegressor
        The model to evaluate
    train_raw: numpy.ndarray
        Train data
    test_raw: numpy.ndarray
        Test data
    validation_raw: numpy.ndarray
        Validation data
    when : str, optional
        Flag to indicate if model is already pruned (default 'vor')
    """
    
    preds_train = dt.predict(train_raw[:,:-1])
    rmse = math.sqrt(mean_squared_error(train_raw[:,-1], preds_train))
    print("RMSE %s Pruning Training: "%(when), rmse)
    mae = mean_absolute_error(train_raw[:,-1], preds_train)
    print("MAE %s Pruning Training: "%(when), mae)
    
    preds_test = dt.predict(test_raw[:,:-1])
    rmse = math.sqrt(mean_squared_error(test_raw[:,-1], preds_test))
    print("RMSE %s Pruning Test: "%(when), rmse)
    mae = mean_absolute_error(test_raw[:,-1], preds_test)
    print("MAE %s Pruning Test: "%(when), mae)
    
    preds_val = dt.predict(validation_raw[:,:-1])
    rmse = math.sqrt(mean_squared_error(validation_raw[:,-1], preds_val))
    print("RMSE %s Pruning Validation: "%(when), rmse)
    mae = mean_absolute_error(validation_raw[:,-1], preds_val)
    print("MAE %s Pruning Validation: "%(when), mae)

##### CVP #####

def cvp(dt,inner_tree, index,threshold):
    """Performs pruning of last predecessor of leaves using Critical Value Pruning.
    Parameters
    ----------
    dt: DecisionTreeClassifier or DecisionTreeRegressor
        The model that shall be pruned
    inner_tree : sklearn.tree._tree.Tree
        (Root node of) current inner tree that shall be pruned
    index : int
        Index of root node of the inner tree in the whole tree
    threshold: float
        Threshold used for deciding if pruning is performed
    """
        
    inner_left = copy.deepcopy(inner_tree.children_left[index])
    inner_right = copy.deepcopy(inner_tree.children_right[index])
    value = copy.deepcopy(inner_tree.value[index])
    
    if inner_tree.impurity[index] >= threshold:
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF

def prune_leaf(dt,inner_tree,threshold, index=0):
    """Performs tree traversal for Critical Value Pruning.
    Parameters
    ----------
    dt: DecisionTreeClassifier or DecisionTreeRegressor
        The model that shall be pruned
    inner_tree : sklearn.tree._tree.Tree
        (Root node of) current inner tree that shall be pruned
    threshold: float
        Threshold used for deciding if pruning is performed
    index : int, optional
        Index of root node of the inner tree in the whole tree (default 0)
    """
    
    #traverse tree to leaves
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_leaf(dt,inner_tree, threshold, inner_tree.children_left[index])
        
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_leaf(dt, inner_tree,threshold, inner_tree.children_right[index])
        
    # Try pruning children if both children are leaves    
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
        is_leaf(inner_tree, inner_tree.children_right[index])) and index != 0:
        cvp(dt,inner_tree, index,threshold)
        
def prune_classifier(tree,measure, validation_raw):
    """Starts Critical Value Pruning routine for Classification Trees.
    
    Parameters
    ----------
    tree: DecisionTreeClassifier
        The tree that shall be pruned
    measure : str
        Name of the performance measure
    validation_raw: numpy.ndarray
        Validation data used for pruning
    
    Returns
    ----------
    (DecisionTreeClassifier, float, float)
        The pruned tree, its score, its threshold
    """
        
    original_tree = copy.deepcopy(tree)
    num_nodes = original_tree.tree_.node_count
    pruned_trees = {}
    best_score = 0
    score = 0
    best_t = 0
    for t in np.arange(1.0,0.05,-0.05):
        prune_leaf(tree,tree.tree_,t)
        pruned_trees[t] = tree
        tree = copy.deepcopy(original_tree)
    for t in np.arange(1.0,0.05,-0.05):
        if measure == "accuracy" or measure == "reg_accuracy":
            score = accuracy_score(pruned_trees[t].predict(validation_raw[:,:-1]),validation_raw[:,-1])
        elif measure == "f1":
            score = f1_score(pruned_trees[t].predict(validation_raw[:,:-1]),validation_raw[:,-1])
        elif measure == "roc_auc":
            probs = pruned_trees[t].predict_proba(validation_raw[:,:-1])
            probs  = probs[:, 1]
            score = roc_auc_score(validation_raw[:,-1],probs)    

        if best_score <= score:
            best_score = score
            best_t = t
    best_tree = pruned_trees[best_t]
    return best_tree, best_score, best_t
    
def prune_regressor(tree,measure, validation_raw):
    """Starts Critical Value Pruning routine for Regression Trees.
    
    Parameters
    ----------
    tree: DecisionTreeRegressor
        The tree that shall be pruned
    measure : str
        Name of the performance measure
    validation_raw: numpy.ndarray
        Validation data used for pruning
    
    Returns
    ----------
    (DecisionTreeRegressor, float, float)
        The pruned tree, its score, its threshold
    """
       
    original_tree = copy.deepcopy(tree)
    num_nodes = original_tree.tree_.node_count
    pruned_trees = {}
    best_score = 500
    score = 0
    best_t = 0
    for t in np.arange(250000,100,-100):
        prune_leaf(tree,tree.tree_,t)
        pruned_trees[t] = tree
        tree = copy.deepcopy(original_tree)
    for t in np.arange(250000,100,-100):
        if measure == "mse":
            score = np.sqrt(mean_squared_error(pruned_trees[t].predict(validation_raw[:,:-1]),validation_raw[:,-1]))
        elif measure == "mae":
            score = mean_absolute_error(pruned_trees[t].predict(validation_raw[:,:-1]),validation_raw[:,-1])
            
        if best_score >= score:
            best_score = score
            best_t = t
    best_tree = pruned_trees[best_t]
    
    return best_tree, best_score, best_t
    

##### REP #####
def rep_last(dt,inner_tree, index, validation_raw, measure, label, names, select=False):
    """Prunes the given inner tree at the root node if performance is better with pruned tree. 
    Parameters
    ----------
    dt: DecisionTreeClassifier or DecisionTreeRegressor
        The model to prune
    inner_tree : sklearn.tree._tree.Tree
        (Root node of) inner tree that should be pruned. 
        Must be a direct predecessor of leaf nodes.
    index : int
        Index of root node of the inner tree in the whole tree
    validation_raw: numpy.ndarray
        Validation data
    measure: str
        Performance measure to be used (i.e. RMSE or Accuracy)
    label: str
        Name of label
    names: pandas.core.indexes.base.Index
        Attribute names
    select: bool, optional
        Flag to indicate wether only selected attributes were used (default False)
    """
    
    #Store old tree if pruning doesn't lead to better performance
    inner_left = copy.deepcopy(inner_tree.children_left[index])
    inner_right = copy.deepcopy(inner_tree.children_right[index])
    value = copy.deepcopy(inner_tree.value[index])
    
    #Calculate performance of unpruned tree on validation data
    preds_old = []
    if measure != "roc_auc":
        preds_old = dt.predict(validation_raw[:,:-1])
    else:
        preds_old = dt.predict_proba(validation_raw[:,:-1])
        preds_old = preds_old[:,1]
        
    
    #Prune children
    inner_tree.children_left[index] = TREE_LEAF
    inner_tree.children_right[index] = TREE_LEAF
    
    #Calculate performance of pruned tree on validation data
    preds_new = []
    if measure != "roc_auc":
        preds_new = dt.predict(validation_raw[:,:-1])
    else:
        preds_new = dt.predict_proba(validation_raw[:,:-1])
        preds_new = preds_new[:,1]
        
    #Render pruned tree
    if select: 
        os.makedirs(os.path.dirname("graphs/dot/%s/REP/select/%s/prune_dt_error_%d.dot"%(label,measure,index)), exist_ok = True)
        os.makedirs(os.path.dirname("graphs/%s/REP/select/%s/prune_dt_error_%d.png"%(label,measure,index)), exist_ok = True)
        tree.export_graphviz(dt, out_file='graphs/dot/%s/REP/select/%s/prune_dt_error_%d.dot'%(label,measure,index), feature_names= names,
                     filled=True, rounded=True,special_characters=True)
        os.system("dot -T png graphs/dot/%s/REP/select/%s/prune_dt_error_%d.dot -o graphs/%s/REP/select/%s/prune_dt_error_%d.png"%(label,measure,index,label,measure,index))
    
    else: 
        os.makedirs(os.path.dirname("graphs/dot/%s/REP/%s/prune_dt_error_%d.dot"%(label,measure,index)), exist_ok = True)
        os.makedirs(os.path.dirname("graphs/%s/REP/%s/prune_dt_error_%d.png"%(label,measure,index)), exist_ok = True)
        tree.export_graphviz(dt, out_file='graphs/dot/%s/REP/%s/prune_dt_error_%d.dot'%(label,measure,index), feature_names= names,
                     filled=True, rounded=True,special_characters=True)
        os.system("dot -T png graphs/dot/%s/REP/%s/prune_dt_error_%d.dot -o graphs/%s/REP/%s/prune_dt_error_%d.png"%(label,measure,index,label,measure,index))
    

    #Check results
    if measure == "accuracy" or measure =="reg_accuracy":
        score_old = accuracy_score(validation_raw[:,-1], preds_old)
        score_new = accuracy_score(validation_raw[:,-1], preds_new)
    elif measure == "f1":
        score_old = f1_score(validation_raw[:,-1], preds_old)
        score_new = f1_score(validation_raw[:,-1], preds_new)
    elif measure == "roc_auc":
        score_old = roc_auc_score(validation_raw[:,-1], preds_old)
        score_new = roc_auc_score(validation_raw[:,-1], preds_new)
    elif measure == "mse":
        score_old = np.sqrt(mean_squared_error(validation_raw[:,-1], preds_old))
        score_new = np.sqrt(mean_squared_error(validation_raw[:,-1], preds_new))
    elif measure == "mae":
        score_old = mean_absolute_error(validation_raw[:,-1],preds_old)
        score_new = mean_absolute_error(validation_raw[:,-1],preds_new)

    #Restore old unpruned tree if no performance gain achieved
    if (measure == "accuracy" or measure == "f1" or measure == "roc_auc" or measure == "reg_accuracy") and score_old > score_new:
        inner_tree.children_left[index] = inner_left
        inner_tree.children_right[index] = inner_right 
    elif (measure == "mse" or measure == "mae") and score_old < score_new:
        inner_tree.children_left[index] = inner_left
        inner_tree.children_right[index] = inner_right 


def prune_error(dt,inner_tree, validation_raw, measure, label, names, select=False, index=0):
    """Prunes the given (inner) tree using Reduced Error Pruning.
    Parameters
    ----------
    dt: DecisionTreeClassifier or DecisionTreeRegressor
        The model to prune
    inner_tree : sklearn.tree._tree.Tree
        (Root node of) inner tree that should be pruned
    validation_raw: numpy.ndarray
        Validation data
    measure: String
        Performance measure to be used (i.e. RMSE or Accuracy)
    label: String
        Name of label
    names: pandas.core.indexes.base.Index
        Attribute names
    select: Boolean, optional
        Flag to indicate wether only selected attributes were used (default False)
    index : int, optional
        Index of root node of the inner tree in the whole tree (default 0)
    """
    
    #Traverse tree to leaves in postorder and prune inner trees recursively

    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_error(dt, inner_tree, validation_raw, measure, label, names, select, inner_tree.children_left[index])
        
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_error(dt, inner_tree, validation_raw, measure, label, names, select, inner_tree.children_right[index])
        if index !=0 and not (is_leaf(inner_tree, inner_tree.children_left[index]) or
        is_leaf(inner_tree, inner_tree.children_right[index])):
            rep_last(dt,inner_tree, index, validation_raw, measure, label, names, select)
         
    if (is_leaf(inner_tree, inner_tree.children_left[index]) or
        is_leaf(inner_tree, inner_tree.children_right[index])):
        if index != 0:
            rep_last(dt,inner_tree, index, validation_raw, measure, label, names, select)
            
def rep(dt,validation_raw, measure, label, names, select=False):
    """Prunes the given tree using Reduced Error Pruning according to given performance measure.
    Parameters
    ----------
    dt: DecisionTreeClassifier or DecisionTreeRegressor
        The model to prune
    validation_raw: numpy.ndarray
        Validation data
    inner_tree : sklearn.tree._tree.Tree
        Tree that should be pruned
    measure: String
        Performance measure to be used (i.e. RMSE or Accuracy)
    label: String
        Name of label
    names: pandas.core.indexes.base.Index
        Attribute names
    select: Boolean, optional
        Flag to indicate wether only selected attributes were used (default False)
    """
    
    prune_error(dt,dt.tree_,validation_raw,measure,label,names,select)
    
    
    
    
    
    
