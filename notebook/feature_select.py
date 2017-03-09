import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RandomizedLasso
from sklearn.preprocessing import MinMaxScaler
#from sklearn.linear_model import Lasso
from scipy.stats import uniform as sp_rand 

class Feature_Filter:
    def __init__(self,model,X_train,y,
                 par_grid=dict(alpha=np.array([1,0.1,0.01,0.001,0.0001,0]))):
        self.model = model
        #self.par_dict = par_dict
        self.x = X_train
        self.y = y
        self.names = X_train.columns
        self.par_grid = par_grid
        #self.rankdict = {}
        self.ranks_df = pd.DataFrame()
        
    def tune_par_grid(self):
        grid = GridSearchCV(
                estimator=self.model,
                param_grid=self.par_grid)
        grid.fit(self.x, self.y)
        #print(grid)
        #print(grid.best_score_)
        #print(grid.best_estimator_.alpha)
        self.SearchCV = grid
        
    def tune_par_rand(self,par_dict={'alpha': sp_rand()},
                      N_search = 100):
        rsearch = RandomizedSearchCV(
                estimator=self.model, 
                param_distributions=par_dict, n_iter=N_search)
        rsearch.fit(self.x, self.y)
        #print(rsearch)
        #print(rsearch.best_score_)
        #print(rsearch.best_estimator_.alpha)
        self.SearchCV = rsearch    
    
    def rank_to_dict(self, rank, names, order=1):
        minmax = MinMaxScaler()
        rank = minmax.fit_transform(order*np.array([rank]).T).T[0]
        rank = map(lambda x: round(x, 2), rank)
        rank_dict = dict(zip(names, rank ))
        rank_dict = [(k, rank_dict[k]) for k in 
                     sorted(rank_dict, key=rank_dict.get, reverse=True)]
        return rank_dict

    def rand_lasso(self,max_it=50000, n_resample = 500):
        par = self.SearchCV.best_estimator_.alpha
        rlasso = RandomizedLasso(
                alpha=par, max_iter=max_it,
                random_state=1,
                n_resampling=n_resample)
        rlasso.fit(self.x, self.y)
        # create a dict to store the ranks 
        #rankdict={}
        #rankdict["Rn_La"] = self.rank_to_dict(np.abs(rlasso.scores_), self.names)
        #self.ranks_df = pd.DataFrame.from_dict(rankdict)
        #self.ranks_df = self.ranks_df.append(pd.DataFrame(self.rankdict))
        return rlasso.scores_
        
    def auto_select(self,N_searchCV=100):
        np.random.seed(1)
        self.tune_par_rand(N_search=N_searchCV)
        scores = self.rand_lasso()
        return scores
        