import numpy as np 
from collections import Counter


# 2 classes, Node and tree
# define entropy 
#change pandas to  numpy for the code to work 
def entropy(y):
    e = np.bincount(y)
    en = e/len(y)
    return -np.sum([p * np.log2(p) for p in en if p>0 ])
class Node:
    #store info for our node
    def __init__(self, feature=None, threshold=None,
    left=None,right=None,*,value=None) -> None:
        #value is a keyword only parameter
        self.feature = feature
        self.threshold = threshold 
        self.left = left
        self.right = right
        self.value = value
    def get_params(self, deep=True):
        return{'feature': self.feature,
        'threshold': self.threshold, 
        'left': self.left,
        'right': self.right,
        'value' : self.value }
    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
    def __sklearn_clone__(self):
        return self
    def is_leaf(self):
        # if value then we are at a leaf node
        return self.value is not None
class DecisionTree:
    def __init__(self, min_samples_split=None,root = None,  max_depth=None, n_feats=None) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = root
    def get_params(self, deep=True):
        return{'min_samples_split': self.min_samples_split,
        'max_depth': self.max_depth, 
        'n_feats': self.n_feats}
    def set_params(self, **params): 
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self
    def __sklearn_clone__(self):
        return self
    def fit(self,X,y):
        #grow tree
        #safety check first if n_feats isnt specified we get the max n_feats 
        # otherwise the min we see below 
        #check to work only with numpy 
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X,y)
        return self
    
    def _grow_tree(self,X,y,depth=0):
        n_samples,n_features = X.shape 
        n_labels = len(np.unique(y))
        # check if parameters are None
        if (self.min_samples_split is None or 
            self.max_depth is None):
                raise ValueError("Please set min_samples_split and max_depth before fitting the model.")
        #stopping criteria
        if (depth >= self.max_depth 
            or n_labels == 1
            or n_samples< self.min_samples_split ):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)
        #replace =False, we dont want the same indices 
        #multiple times
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)
        #greedy search
        best_feat, best_thresh = self.best_criteria(X,y,feat_idxs)
        left_idxs, right_idxs = self._split(X[:,best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs,:],y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs,:],y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right )
    def most_common_label(self,y):
        #calculate the number of occurences of y 
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    def best_criteria(self,X,y,feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
        
            X_col = X[: , feat_idx]
            thresh = np.unique(X_col)
            for t in thresh:
            #sample_thresh = np.random.choice(thresh,size = min(len(thresh),70),
             #replace= False)
                gain = self.info_gain(y,X_col,t)
                if gain> best_gain :
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = t
        return split_idx, split_thresh
    def info_gain(self,y,X_col,split_thresh):
        Parent = entropy(y)
        left_indices ,right_indices = self._split(X_col,split_thresh)
        if (len(left_indices) == 0 
        or len(right_indices) == 0):
            return 0
        n = len(y)
        n_l, n_r =  len(left_indices), len(right_indices)
        e_l, e_r = entropy(y[left_indices]), entropy(y[right_indices])
        ig = Parent - e_l * n_l/n - e_r * n_r/n
        return ig

    def _split(self,X_col,split_thresh):
        left_idxs = np.argwhere(X_col <= split_thresh).flatten()
        right_idxs = np.argwhere(X_col >= split_thresh).flatten()
        return left_idxs, right_idxs
    def traverse_tree(self,x ,node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x,node.left)
        return self.traverse_tree(x, node.right)
    
    def predict(self,X):
        #traverse tree 
        return np.array([self.traverse_tree(x,self.root) for x in X])
    