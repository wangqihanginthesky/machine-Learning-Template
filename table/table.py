from abc import ABCMeta
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

class TableModel(metaclass=ABCMeta):

    def __init__(self,rawdata):
        """Initialize the model.
        """
        self.data=rawdata

    def _featuresEngineering(self):
        """You need to custmize this by your self.

        Returns:
            dataframe: _description_
        """
        print("make features")
        newdata=self.data
        return newdata

    def _tunning(self,kfold_n=5):
        pass

    def _featureSelection(self,model,features):
        pass

    def _train(self,kfold_n):
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
            rmspe_score = mean_squared_error(y, oof_predictions)
            # Add predictions to the out of folds array
            oof_predictions[val_ind] = locals()["models_%s"%fold].predict(x_val)

        rmspe_score = rmspe(y, oof_predictions)

        return model_list,rmspe_score
            
    def pipeline(self):
        self.data=self._featuresEngineering()
        
        self.tunning()
        self.train(5)
        
