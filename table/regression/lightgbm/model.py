import lightgbm as lgb
import pickle
import joblib
import numpy as np
from sklearn.model_selection import KFold
import mlflow
import optuna.integration.lightgbm as lgb_o
from .loss import rmspe,feval_rmspe,rmse
from datetime import datetime
seed=2022

class LightGBM:
    """
        step1[Feature Engineering]:
            Create New Features
        step2[Feature Selection]:
            - Importance
            - Boruta
        step3[Tuning]:
            - Optuna
    """

    def __init__(self,rawdata,target_feature,loss):
        """Initialize the model.

        Args:
            rawdata (dataframe): Raw data.
            target_feature (string): Feature you want to predict.
            loss (function): you can choose [rmspe,feval_rmspe]
        """
        self.loss=loss
        self.target_feature=target_feature
        self.features=list(rawdata.columns.drop(self.target_feature))
        self.data=rawdata
        self.params ={
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

    def _featuresEngineering(self):
        """You need to custmize this by your self.

        Returns:
            dataframe: _description_
        """
        print("make features")
        self.data
        

    def _tunning(self,kfold_n=5):
        params = {'objective': 'regression',
          'metric': 'rmse',
          'random_seed':seed}
        best_params=[] 
        
        self._featuresEngineering()
        x = self.data.drop([self.target_feature], axis = 1)
        y = self.data[self.target_feature]
        # Create a KFold object
        kfold = KFold(n_splits = kfold_n, random_state = 66, shuffle = True)
        # Train
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
            print(f'Training fold {fold + 1}')
            x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
            y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
            train_dataset = lgb_o.Dataset(x_train, y_train)
            val_dataset = lgb_o.Dataset(x_val, y_val)
            
            locals()["gbm_o%s"%fold] =lgb_o.train(params = params, 
                            train_set = train_dataset, 
                            valid_sets = [train_dataset, val_dataset],
                            early_stopping_rounds=400,
                            verbose_eval=-1)
            best_params.append(locals()["gbm_o%s"%fold].params)
        return best_params

    def _featureSelection(self,model,features):
        pass

    def _train(self,kfold_n,discription):
        mlflow.lightgbm.autolog()
        mlflow.set_tag('mlflow.note.content', discription)
        model_list=[]
        self._featuresEngineering()
        x = self.data.drop([self.target_feature], axis = 1)
        y = self.data[self.target_feature]
        oof_predictions = np.zeros(x.shape[0])
        # Create a KFold object
        kfold = KFold(n_splits = kfold_n, random_state = 66, shuffle = True)
        # Train
        mlflow.end_run()
        with mlflow.start_run(run_name="lightGBM"):
            if type(self.params) is list:
                pass
            else:
                for key in self.params:
                    mlflow.log_param(key, self.params[key])

            for fold, (trn_ind, val_ind) in enumerate(kfold.split(x)):
                print(f'Training fold {fold + 1}')
                x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
                y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
                train_dataset = lgb.Dataset(x_train, y_train)
                val_dataset = lgb.Dataset(x_val, y_val)
                if type(self.params) is list:
                    model_tmp=lgb.train(params = self.params[fold], 
                                    train_set = train_dataset, 
                                    valid_sets = [train_dataset, val_dataset],
                                    early_stopping_rounds=400,
                                    verbose_eval=-1,
                                    )
                else:
                    model_tmp=lgb.train(params = self.params, 
                                    train_set = train_dataset, 
                                    valid_sets = [train_dataset, val_dataset],
                                    early_stopping_rounds=400,
                                    verbose_eval=-1,
                                    )
                model_list.append(model_tmp)
                joblib.dump(model_tmp, 'regression/lightgbm/models/model_%s.pkl'%fold)
                
                # Add predictions to the out of folds array
                oof_predictions[val_ind] = model_tmp.predict(x_val)
            rmspe_score=rmspe(y, oof_predictions)
            rmse_score=rmse(y, oof_predictions)
            print(f"rmpse is {rmspe_score} , rmse is {rmse_score}")
            mlflow.log_metric("mean rmse", rmspe_score)
            mlflow.log_metric("mean rmspe_score", np.mean(rmse_score))
            mlflow.end_run()
            
        
            
    def pipeline(self):
        self.data=self._featuresEngineering()
        self.tunning()
        self.train(kfold_n=5)
        


        

        




            
