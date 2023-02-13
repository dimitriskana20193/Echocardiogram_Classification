import numpy as np 
from collections import Counter 
from tree import DecisionTree as DT 
""""This implementation creates 
a number of decision trees and 
fits each one on a random subsample 
of the training data.
When making predictions,
it averages the predictions of all the trees 
to produce a final prediction for each sample."""
def most_common_label(y):
        #calculate the number of occurences of y 
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
def bootstrap(x,y):
    n_samples = x.shape[0]
    #randomly select data from each tree
    idx = np.random.choice(n_samples, size = n_samples, replace =True )
    return x[idx],y[idx]
class R_Forrest:
    def __init__(self, min_samples_split=None,n_trees = 100,  max_depth=None, n_feats=None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.n_trees = n_trees
        self.trees = []
    def get_params(self, deep=True):
        return{'min_samples_split': self.min_samples_split,
        'max_depth': self.max_depth, 
        'n_feats': self.n_feats,
        'n_trees': self.n_trees}
    def set_params(self, **params): 
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
    def __sklearn_clone__(self):
        return self
    def fit(self,X,y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DT(min_samples_split = self.min_samples_split, 
            max_depth = self.max_depth, n_feats = self.n_feats)
            X_sample, y_sample = bootstrap(X,y)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)
        return self
    def predict(self,X): 
        predictions = np.zeros((len(X),len(self.trees)))
        for i, tree in enumerate(self.trees):
            predictions [:, i] = tree.predict(X)
        return np.round(np.mean(predictions, axis=1))
