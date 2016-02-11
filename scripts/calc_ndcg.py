#customized evalerror function for xgboost to pass to feval when doing cross validation
#example: xgb.cv(param, xg_train, num_boost_round = nround, nfold = 10, feval=evalerror)

from __future__ import division
import pandas as pd
import numpy as np

def ndcg(preds, labels):
    """Calculate sum of normalzied discounted cumulative gain for the predictions
    The correct prediction will have a relevance of 1, incorrect predictions will have a relevance of 0.
    Weight the relevance values such that it is reduced logarithmically proportional to it its position     
    
    Args:
        preds: n*5 array of predictor targets
        labels: n*1 array targets
    Returns:
        sum of normalized discounted cumulative gain for the predictions of all users
    """
    
    #find positions where the prediction matches the label
    relv_pos = np.where(np.equal(preds, labels))[1]
    #weight func: log2(i+1), add one more to adjust for 0-based indexing
    total_ndcg = np.sum(1./np.log2(relv_pos+2))
    return total_ndcg
    
def eval_all(cls_prob, dtrain):
    """find top k predictions from probability matrix and call ndcg to find accuracy of predictions
    
    Args:
        cls_prob: 2D array, probability of each class for each person (n persons by m classes),
                  the column index corresponds with class
        labels: labels for the n persons
    returns:
        prediction accuracy using ndcg to evaluate predictions of each AirBNB user
    """
    #determine the top k predictions
    labels = dtrain.get_label()
    top_k = cls_prob.argsort(axis = 1)[:,::-1][:,:5]
#    top_k = cls_prob.argsort(axis = 1)[:,:k:-1]
    #convert true values and compared with predictions to check for equality
    labels = labels[:, None]
    return 1-ndcg(top_k, labels)/len(labels)
    
def eval_ndfUs(cls_prob, dtrain):
    """Calculate NDCG for users who chose US or NDF, NDF is encoded as 7 and US is encoded as 10."""
    
    labels = dtrain.get_label()
    pred = cls_prob.argsort(axis = 1)[:,::-1][:,:5]
#only retain the labels for users who chose US or NDF
    users_idx = np.logical_or(labels == 7, labels == 10)
    labels = labels[users_idx]
    pred = pred[users_idx,:]
    labels = labels[:, None]
    return 1-ndcg(pred, labels)/len(labels)

def eval_foreign(cls_prop, dtrain):
    """Calculate ndcg error for the users who chose destinations outside the US."""
       
    labels = dtrain.get_label()
    pred = cls_prob.argsort(axis = 1)[:,::-1][:,:5]
    users_idx = np.logical_or(labels != 7, labels != 10)
#only retain the labels for users who chose foreign destinations
    labels = labels[users_idx]
    pred = pred[users_idx,:]
    labels = labels[:, None]
    return 1-ndcg(pred, labels)/len(labels)
    
def eval_error3(cls_prob, dtrain):
    f_err = eval_foreign(cls_prob, dtrain)
    ndfUs_err = eval_ndfUs(cls_prob, dtrain)
    all_err = eval_all(cls_prob, dtrain)
    return 'Error3', {'ndfUs': ndfUs_err, 'foreign': f_err, 'all': all_err}  
    
# def us_misclf(cls_prob, dtrain):
#     """Find percent of misclassification at the first position for users who chose US.
#     US is encoded as 10."""
    
#     labels = dtrain.get_label()
#     #sort the probabilities and retain the index with the highest probability
#     pred = cls_prob.argsort(axis = 1)[:,::-1][:,:5]
#     #determine which users have labels as US
#     users_idx = np.where(labels == 10)[0]
#     return 'error', 1 - np.sum(np.equal(pred[users_idx], 10))/len(users_idx)
