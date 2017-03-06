#import numpy as np
from sklearn.model_selection import GridSearchCV

class Para_Filter:
    def __init__(self,model,par_dict,X_train,y):
        self.model = model
        self.par_dict = par_dict
        self.x = X_train
        self.y = y
        
    def tune_par_grid(self):
        grid = GridSearchCV(
                estimator=self.model,
                param_grid=self.par_dict)
        grid.fit(self.x, self.y)
        print(grid)
        print(grid.best_score_)
        print(grid.best_estimator_.alpha)