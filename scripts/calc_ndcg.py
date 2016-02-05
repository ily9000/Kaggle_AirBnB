#customized evalerror function for xgboost to pass to feval when doing cross validation
#example: xgb.cv(param, xg_train, num_boost_round = nround, nfold = 10, feval=evalerror)

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
        sum of normalized discounted cumulative gain for all the predictions
    """
    
    #find positions where the prediction matches the label
    relv_pos = np.where(np.equal(preds, labels))[1]
    #weight func: log2(i+1), add one more to adjust for 0-based indexing
    total_ndcg = np.sum(1./np.log2(relv_pos+2))
    return total_ndcg
    
def evalerror(cls_prob, dtrain):
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
    k = 5
    top_k = cls_prob.argsort(axis = 1)[:,:k:-1]
    #convert true values and compared with predictions to check for equality
    labels = labels[:, None]
    return 'error', 1-ndcg(top_k, labels)/len(labels)