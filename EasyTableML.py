#Load model
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
import lightgbm as lgb_model

#Data Process
import numpy as np
import pandas as pd
import math

#Automatic parameter adjustment
from tune_sklearn import TuneGridSearchCV

#Estimate model
import sklearn.metrics as sm

#OS
import joblib
import os

#Ensemble learn
from sklearn.ensemble import StackingRegressor

#Visualization
from matplotlib import pyplot as plt
import seaborn as sns

class EasyTableMLRegression():
    def get_base_models(self):  
        linear = LinearRegression()
        knn = KNeighborsRegressor()
        svr=SVR()
        ridge=Ridge()
        etree=ExtraTreeRegressor()
        rf=RandomForestRegressor()
        gb=GradientBoostingRegressor()
        bag=BaggingRegressor()
        lgbm=lgb_model.sklearn.LGBMRegressor()

        models={
            'liner':linear,
            'knn':knn,
            'svm':svr,
            'ridge':ridge,
            'etree':etree,
            'rf':rf,
            'gb':gb,
            'bag':bag,
            'lgbm':lgbm,
        }
        return models
        
    def base_auto_parameter(self, model_list, x_train, y_train, auto_scoring='r2', cv=3,custom_parameters=None ,n_jobs=1, details=1):
        if custom_parameters == None:
            parameters={
                'liner':{
                    'n_jobs':[-1],
                    'fit_intercept':[False,True]
                },
                'knn':{
                    'n_jobs':[-1],
                    'n_neighbors':[5,10,20],
                },
                'svm':{
                    'gamma':['scale','auto']
                },
                'ridge':{
                    'alpha':[0.1,1,10],
                },
                'etree':{
                    'splitter':['best'],
                },
                'rf':{
                    'n_jobs':[-1],
                    'n_estimators':[100,300],
                },
                'gb':{
                    'n_estimators':[100,300],
                },
                'bag':{
                    'n_jobs':[-1],
                    'n_estimators':[10,100,300],
                },
                'lgbm':{
                    'n_jobs':[-1],
                    'learning_rate':[0.001,0.01,0.1,0.5,1,1.5,2],
                    'n_estimators':[i for i in range(1,502,2)]
                }
            }
        else:
            parameters=custom_parameters

        print('Auto Parameter Start!')
        if not os.path.exists(os.path.join('models','AutoML')):
            os.makedirs(os.path.join('models','AutoML'))
        best_models={}
        for i, (name, model) in enumerate(model_list.items()):
            print('Note(Auto Parameter):','We are currently searching the ',i+1," model's beat parameter, ",'model name: ',name)
            print('Plase Wait...')
            grid_search=TuneGridSearchCV(model,parameters[name],refit=True,cv=cv,scoring=auto_scoring,verbose=details,n_jobs=n_jobs)
            grid_search.fit(x_train,y_train)
            joblib.dump(grid_search.best_estimator_, os.path.join('models','AutoML',name+'.pkl'))
            print('Best model will be saved in:',os.path.join('models','AutoML',name+'.pkl'))
            best_models[name]=grid_search.best_estimator_
        print('Auto Parameter Done!')
        return best_models
    
    def train(self, model_list, x_train, y_train, train_type, meta_model=None, cv=3, auto_custom_parameters=None,n_jobs=1,auto_scoring='r2', details=1, auto_parameter=False):
        if train_type == 'base':
            if auto_parameter==False:
                for i, (name, model) in enumerate(model_list.items()):
                    print('Note:','We are currently training the ',i+1,' model,','model name: ',name)
                    print('Plase Wait...')
                    model.fit(x_train,y_train)
                    #save model
                    joblib.dump(model, os.path.join('models',name+'.pkl'))
                print('Train Done!')
            else:
                model_list = self.base_auto_parameter(model_list,x_train,y_train,auto_scoring=auto_scoring,cv=cv,custom_parameters=auto_custom_parameters,n_jobs=n_jobs,details=details)
            return model_list
        if train_type == 'meta':
            if auto_parameter == False:
                if meta_model == None:
                    raise ValueError('When you train META model and not use auto_parameter, "meta_model" cannot be None')
                print('Note:','We are currently training the META model')
                print('Plase Wait...')
                meta_learner=StackingRegressor(estimators=list(model_list.items()),final_estimator=meta_model,n_jobs=-1)
                meta_learner.fit(x_train,y_train)
                joblib.dump(meta_learner, os.path.join('models','meta_learner.pkl'))
            else:
                if auto_custom_parameters == None:
                    parameters={
                        'final_estimator':[
                            MLPRegressor(hidden_layer_sizes=(30,100,10),max_iter=5000,alpha=0.1),
                            MLPRegressor(hidden_layer_sizes=(30,100,10),max_iter=5000,alpha=1),
                            MLPRegressor(hidden_layer_sizes=(30,100,10),max_iter=5000,alpha=10),
                        ],
                        'n_jobs':[-1]
                    }
                else:
                    parameters=auto_custom_parameters
                grid_search=TuneGridSearchCV(StackingRegressor(estimators=list(model_list.items()),cv=3),parameters,refit=True,cv=5,scoring=auto_scoring,verbose=details,n_jobs=1)
                grid_search.fit(x_train,y_train)
                meta_learner=grid_search.best_estimator_
                joblib.dump(meta_learner, os.path.join('models','AutoML','meta_learner.pkl'))
            return meta_learner

    def load_base_model(self,model_list,is_auto_parameter=False):
        if is_auto_parameter==False:
            for i in range(len(model_list)):
                model_ll=list(model_list.items())
                name=model_ll[i][0]
                print('Note:','Load model ',i+1,' ,','model name: ',name)
                model_list[name]=joblib.load(os.path.join('models',name+'.pkl'))
            print('Load Done!')
        else:
            for i in range(len(model_list)):
                model_ll=list(model_list.items())
                name=model_ll[i][0]
                print('Note:','Load model ',i+1,' ,','model name: ',name)
                model_list[name]=joblib.load(os.path.join('models','AutoML',name+'.pkl'))
            print('Load Done!')
        return model_list
    
    def load_meta_learner(self,is_auto_parameter=False):
        if is_auto_parameter == True:
            meta_learner=joblib.load(os.path.join('models','AutoML','meta_learner.pkl'))
        else:
            meta_learner=joblib.load(os.path.join('models','meta_learner.pkl'))
        return meta_learner
    
    def estimate(self,model_list,x_test,y_test,details=0):
        #details:1->limited details;2->all details
        P = np.zeros((y_test.shape[0], len(model_list)))
        P = pd.DataFrame(P)
        cols = []
        result_score=np.zeros((3,len(model_list)))
        result_score=pd.DataFrame(result_score)
        result_score.index=['MAE','RMSE','R2']

        for i, (name, model) in enumerate(model_list.items()):
            if details >= 1:
                print('Valid model ',i+1,' model name : ',name)
            y_test_pred=model.predict(x_test)
            MAE=sm.mean_absolute_error(y_test, y_test_pred)
            RMSE=np.sqrt(sm.mean_squared_error(y_test, y_test_pred))
            R2=sm.r2_score(y_test, y_test_pred)
            if details >= 2:
                print('MAE:',MAE)
                print('RMSE:',RMSE)
                print('R square:',R2)
                print('\n')
            P.iloc[:, i]=y_test_pred
            result_score.iloc[0,i]=MAE
            result_score.iloc[1,i]=RMSE
            result_score.iloc[2,i]=R2
            cols.append(name)
        P.columns = cols
        result_score.columns=cols
        return P,result_score
    
    def estimate_base(self,model_list,x_train,y_train,x_valid,y_valid,details=0):
        print('Estimating Train Set...')
        _,result_score_train=self.estimate(model_list,x_train,y_train,details)
        
        print('Estimating Valid/Test Set...')
        P,result_score_valid=self.estimate(model_list,x_valid,y_valid,details)

        if (details >= 1):
            print('result_score_train')
            print(result_score_train)
            print('result_score_valid/test')
            print(result_score_valid)

        print('\nAuto Suggestion:')
        overfit=(result_score_train - result_score_valid).T
        overfit=overfit[overfit['R2']>0.05]
        low_score=result_score_valid.T
        low_score=low_score[low_score['R2']<0.9]
        if (len(overfit)>0):
            print('The fellowing models may have overfitting, please consider modifying them.')
            print('Train score - Valid score:')
            print(overfit)
        if (len(low_score)>0):
            print('The fellowing models have lower scores, please consider removing them.')
            print(low_score)
        if (len(overfit)==0 and len(low_score)==0):
            print('All models performed well!')

        print('Base Models Correlation Matrix:')
        corrmat = P.corr()
        f, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(corrmat, square=True,annot=True)
        return P
    
    def estimate_meta(self,meta_model,x_train,y_train,x_valid,y_valid,details=0):
        print('Estimating Train Set...')
        _,result_score_train=self.estimate({'meta_model':meta_model},x_train,y_train,details)
        
        print('Estimating Valid/Test Set...')
        _,result_score_valid=self.estimate({'meta_model':meta_model},x_valid,y_valid,details)

        if (details >= 1):
            print('result_score_train')
            print(result_score_train)
            print('result_score_valid/test')
            print(result_score_valid)

        print('\nAuto Suggestion:')
        overfit=(result_score_train - result_score_valid).T
        overfit=overfit[overfit['R2']>0.05]
        low_score=result_score_valid.T
        low_score=low_score[low_score['R2']<0.9]
        if (len(overfit)>0):
            print('The model may have overfitting, please consider modifying it.')
            print('Train score - Valid score:')
            print(overfit)
        if (len(low_score)>0):
            print('Note:The model have low scores')
            print(low_score)
        if (len(overfit)==0 and len(low_score)==0):
            print('All models performed well!')

    def predict(self,model,x_pred):
        y_pred=model.predict(x_pred)
        return y_pred
    
    def fit(self,x_train,y_train,auto_scoring='r2',cv=3 ,n_jobs=1,details=1):
        print('Start Auto Train! Leave it all to me!')
        base_models=self.get_base_models()
        best_models=self.train(base_models,x_train,y_train,'base',cv=cv,auto_scoring=auto_scoring,n_jobs=n_jobs,details=details,auto_parameter=True)
        meta_learner=self.train(best_models,x_train,y_train,'meta',cv=cv,auto_scoring=auto_scoring,n_jobs=n_jobs,details=details,auto_parameter=True)
        print('All Done!')
        return meta_learner