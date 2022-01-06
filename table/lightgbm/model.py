import lightgbm as lgb
import pickle
import joblib
import numpy as np
from sklearn.model_selection import KFold
from loss import rmspe,feval_rmspe
seed=2021

class LightGBM:

    def __init__(self,rawdata,test_data,target_feature,loss):
        self.loss=loss
        self.target_feature=target_feature
        self.rawdata=rawdata
        self.params = {
                    'learning_rate': 0.1,        
                    'lambda_l1': 1,
                    'lambda_l2': 1,
                    'num_leaves': 100,
                    'min_sum_hessian_in_leaf': 20,
                    'feature_fraction': 0.8,
                    'feature_fraction_bynode': 0.1,
                    'bagging_fraction': 0.90,
                    'bagging_freq': 42,
                    'min_data_in_leaf': 690,
                    'max_depth': 3,
                    'seed': seed,
                    'feature_fraction_seed': seed,
                    'bagging_seed': seed,
                    'drop_seed': seed,
                    'data_random_seed': seed,
                    'objective': 'rmse',
                    'boosting': 'gbdt',
                    'verbosity': -1,
                    'n_jobs': -1,
                }   

    def makeFeatures(self):
        print("make features")
        newdata=self.rawdata
        return newdata

    def tunning(self,kfold_n=5):
        pass
    
    def featureSelection(self,model,features):
        pass

    def train(self,kfold_n):
        model_list=[]
        dataset=self.makeFeatures()
        x = dataset.drop([self.target_feature], axis = 1)
        y = dataset[self.target_feature]
        oof_predictions = np.zeros(x.shape[0])
        # Create a KFold object
        kfold = KFold(n_splits = kfold_n, random_state = 66, shuffle = True)
        # Train
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
            print(f'Training fold {fold + 1}')
            x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
            y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
            train_dataset = lgb.Dataset(x_train, y_train)
            val_dataset = lgb.Dataset(x_val, y_val)
            locals()["models_%s"%fold] = lgb.train(params = self.params, 
                            train_set = train_dataset, 
                            valid_sets = [train_dataset, val_dataset],
                            feval = self.loss)
            model_list.append(locals()["models_%s"%fold])
            joblib.dump(locals()["models_%s"%fold], './models/model_%s.pkl'%fold)
            rmspe_score = rmspe(y, oof_predictions)
            # Add predictions to the out of folds array
            oof_predictions[val_ind] = locals()["models_%s"%fold].predict(x_val)

        rmspe_score = rmspe(y, oof_predictions)
        return model_list,rmspe_score
            
    def pipeline(self):
        features=self.makeFeatures()
        self.tunning()
        self.train(5)
        
        
        

        




            
