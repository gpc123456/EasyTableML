#Load model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb_model
from catboost import CatBoostRegressor, Pool
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

#Data Process
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Automatic parameter adjustment
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

#Estimate model
import sklearn.metrics as sm

#OS
import joblib
import os

#Ensemble learn
from sklearn.ensemble import StackingRegressor

#Auto features
from openfe import openfe, transform

#Visualization
from matplotlib import pyplot as plt
import seaborn as sns


class EasyTableMLRegression():

    def auto_features(self, x_train, x_test, y_train, features_number=20, n_jobs=1):
        '''
        return : x_train, x_test, y_train
        '''
        ofe = openfe()
        ofe.fit(data=x_train, label=y_train, n_jobs=n_jobs)
        x_train, x_test = transform(x_train, x_test, ofe.new_features_list[:features_number], n_jobs=n_jobs)
        ss = StandardScaler()
        x_train = ss.fit_transform(x_train)
        x_test = ss.transform(x_test)
        y_train = np.squeeze(np.array(y_train))
        return x_train, x_test, y_train

    def get_base_models(self):
        knn = KNeighborsRegressor()
        lgbm = lgb_model.sklearn.LGBMRegressor()
        catboost = CatBoostRegressor(verbose=0)

        models = {'knn': knn, 'lgbm': lgbm, 'catboost': catboost}
        return models

    def auto_parameter_lgbm(self, lbgm_model, x_train, y_train, auto_scoring='r2', cv=5, n_jobs=1, details=1):
        #Stage one:
        print('Stage 1 of 5')
        parameters_stage_one = {
            'learning_rate': np.around(np.arange(0.05, 0.21, 0.01), 2).tolist(),
            'n_estimators': [i for i in range(1, 1002, 2)]
        }
        lbgm_model = HalvingGridSearchCV(lbgm_model,
                                         parameters_stage_one,
                                         refit=True,
                                         cv=cv,
                                         scoring=auto_scoring,
                                         verbose=details,
                                         n_jobs=n_jobs)
        lbgm_model.fit(x_train, y_train)
        lbgm_model = lbgm_model.best_estimator_

        #Stage two
        print('Stage 2 of 5')
        parameters_stage_two = {'min_child_samples': [i for i in range(10, 200, 1)]}
        lbgm_model = HalvingGridSearchCV(lbgm_model,
                                         parameters_stage_two,
                                         refit=True,
                                         cv=cv,
                                         scoring=auto_scoring,
                                         verbose=details,
                                         n_jobs=n_jobs)
        lbgm_model.fit(x_train, y_train)
        lbgm_model = lbgm_model.best_estimator_

        #Stage three:
        print('Stage 3 of 5')
        parameters_stage_three = {'max_depth': [2, 3, 4, 5, 6], 'num_leaves': [i for i in range(3, 64, 1)]}
        lbgm_model = GridSearchCV(lbgm_model,
                                  parameters_stage_three,
                                  refit=True,
                                  cv=cv,
                                  scoring=auto_scoring,
                                  verbose=details,
                                  n_jobs=n_jobs)
        lbgm_model.fit(x_train, y_train)
        lbgm_model = lbgm_model.best_estimator_

        #Stage four:
        print('Stage 4 of 5')
        parameters_stage_four = {
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
        }
        lbgm_model = GridSearchCV(lbgm_model,
                                  parameters_stage_four,
                                  refit=True,
                                  cv=cv,
                                  scoring=auto_scoring,
                                  verbose=details,
                                  n_jobs=n_jobs)
        lbgm_model.fit(x_train, y_train)
        lbgm_model = lbgm_model.best_estimator_

        #Stage five:
        print('Stage 5 of 5')
        parameters_stage_five = {
            'reg_alpha': [0, 1e-5, 1e-3, 1e-1, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 10],
            'reg_lambda': [0, 1e-5, 1e-3, 1e-1, 0.1, 0.4, 0.6, 0.7, 0.9, 1.0, 10]
        }
        lbgm_model = HalvingGridSearchCV(lbgm_model,
                                         parameters_stage_five,
                                         refit=True,
                                         cv=cv,
                                         scoring=auto_scoring,
                                         verbose=details,
                                         n_jobs=n_jobs)
        lbgm_model.fit(x_train, y_train)
        lbgm_model = lbgm_model.best_estimator_
        return lbgm_model

    def base_auto_parameter(self,
                            model_list,
                            x_train,
                            y_train,
                            auto_scoring='r2',
                            cv=5,
                            custom_parameters=None,
                            n_jobs=1,
                            details=1):
        if custom_parameters == None:
            parameters = {
                'knn': {
                    'n_jobs': [-1],
                    'n_neighbors': [i for i in range(2, 103, 1)],
                },
                'catboost': {
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [6, 7, 8, 9, 10],
                    'l2_leaf_reg': [0, 0.1, 1, 3, 5, 10]
                },
            }
        else:
            parameters = custom_parameters

        print('Auto Parameter Start!')
        if not os.path.exists(os.path.join('models', 'AutoML')):
            os.makedirs(os.path.join('models', 'AutoML'))
        best_models = {}
        for i, (name, model) in enumerate(model_list.items()):
            print('Note(Auto Parameter):', 'We are currently searching the ', i + 1, " model's beat parameter, ",
                  'model name: ', name)
            print('Plase Wait...')
            if type(model) == type(lgb_model.sklearn.LGBMRegressor()) and custom_parameters == None:
                grid_search = self.auto_parameter_lgbm(model,
                                                       x_train,
                                                       y_train,
                                                       auto_scoring=auto_scoring,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       details=details)
            else:
                grid_search = HalvingGridSearchCV(model,
                                                  parameters[name],
                                                  refit=True,
                                                  cv=cv,
                                                  scoring=auto_scoring,
                                                  verbose=details,
                                                  n_jobs=n_jobs)
                grid_search.fit(x_train, y_train)
                grid_search = grid_search.best_estimator_
            joblib.dump(grid_search, os.path.join('models', 'AutoML', name + '.pkl'))
            print('Best model will be saved in:', os.path.join('models', 'AutoML', name + '.pkl'))
            best_models[name] = grid_search
        print('Auto Parameter Done!')
        return best_models

    def train(self,
              model_list,
              x_train,
              y_train,
              train_type,
              meta_model=None,
              cv=5,
              auto_custom_parameters=None,
              n_jobs=1,
              auto_scoring='r2',
              details=1,
              auto_parameter=False):
        if train_type == 'base':
            if auto_parameter == False:
                for i, (name, model) in enumerate(model_list.items()):
                    print('Note:', 'We are currently training the ', i + 1, ' model,', 'model name: ', name)
                    print('Plase Wait...')
                    model.fit(x_train, y_train)
                    #save model
                    joblib.dump(model, os.path.join('models', name + '.pkl'))
                print('Train Done!')
            else:
                model_list = self.base_auto_parameter(model_list,
                                                      x_train,
                                                      y_train,
                                                      auto_scoring=auto_scoring,
                                                      cv=cv,
                                                      custom_parameters=auto_custom_parameters,
                                                      n_jobs=n_jobs,
                                                      details=details)
            return model_list
        if train_type == 'meta':
            if auto_parameter == False:
                if meta_model == None:
                    raise ValueError(
                        'When you train META model and not use auto_parameter, "meta_model" cannot be None')
                print('Note:', 'We are currently training the META model')
                print('Plase Wait...')
                meta_learner = StackingRegressor(estimators=list(model_list.items()),
                                                 final_estimator=meta_model,
                                                 cv=cv,
                                                 passthrough=True,
                                                 verbose=details,
                                                 n_jobs=n_jobs)
                meta_learner.fit(x_train, y_train)
                joblib.dump(meta_learner, os.path.join('models', 'meta_learner.pkl'))
            else:
                if auto_custom_parameters == None:
                    meta_learner_L1_1 = StackingRegressor(estimators=list(model_list.items()),
                                                          final_estimator=MLPRegressor(hidden_layer_sizes=(30, 100, 20),
                                                                                       alpha=0.1,
                                                                                       max_iter=5000),
                                                          cv=cv,
                                                          passthrough=True,
                                                          verbose=details,
                                                          n_jobs=n_jobs)
                    meta_learner_L1_2 = StackingRegressor(estimators=list(model_list.items()),
                                                          final_estimator=RandomForestRegressor(n_jobs=-1),
                                                          cv=cv,
                                                          passthrough=True,
                                                          verbose=details,
                                                          n_jobs=n_jobs)
                    meta_learner_L1_3 = StackingRegressor(estimators=list(model_list.items()),
                                                          final_estimator=KNeighborsRegressor(),
                                                          cv=cv,
                                                          passthrough=True,
                                                          verbose=details,
                                                          n_jobs=n_jobs)
                    meta_learner_L1 = [('L1_1', meta_learner_L1_1), ('L1_2', meta_learner_L1_2),
                                       ('L1_3', meta_learner_L1_3)]
                    print("Training final META learner...")
                    meta_learner = StackingRegressor(estimators=meta_learner_L1,
                                                     final_estimator=MLPRegressor(hidden_layer_sizes=(30, 100, 20),
                                                                                  alpha=1,
                                                                                  max_iter=5000),
                                                     cv=cv,
                                                     verbose=details,
                                                     n_jobs=n_jobs).fit(x_train, y_train)
                else:
                    grid_search = GridSearchCV(StackingRegressor(estimators=list(
                        model_list.items(),
                        passthrough=True,
                        n_jobs=-1,
                    ),
                                                                 cv=5),
                                               auto_custom_parameters,
                                               refit=True,
                                               cv=cv,
                                               scoring=auto_scoring,
                                               verbose=details,
                                               n_jobs=n_jobs)
                    grid_search.fit(x_train, y_train)
                    meta_learner = grid_search.best_estimator_
                joblib.dump(meta_learner, os.path.join('models', 'AutoML', 'meta_learner.pkl'))
            return meta_learner

    def load_base_model(self, model_list, is_auto_parameter=False):
        if is_auto_parameter == False:
            for i in range(len(model_list)):
                model_ll = list(model_list.items())
                name = model_ll[i][0]
                print('Note:', 'Load model ', i + 1, ' ,', 'model name: ', name)
                model_list[name] = joblib.load(os.path.join('models', name + '.pkl'))
            print('Load Done!')
        else:
            for i in range(len(model_list)):
                model_ll = list(model_list.items())
                name = model_ll[i][0]
                print('Note:', 'Load model ', i + 1, ' ,', 'model name: ', name)
                model_list[name] = joblib.load(os.path.join('models', 'AutoML', name + '.pkl'))
            print('Load Done!')
        return model_list

    def load_meta_learner(self, is_auto_parameter=False):
        if is_auto_parameter == True:
            meta_learner = joblib.load(os.path.join('models', 'AutoML', 'meta_learner.pkl'))
        else:
            meta_learner = joblib.load(os.path.join('models', 'meta_learner.pkl'))
        return meta_learner

    def estimate(self, model_list, x_test, y_test, details=0):
        #details:1->limited details;2->all details
        P = np.zeros((y_test.shape[0], len(model_list)))
        P = pd.DataFrame(P)
        cols = []
        result_score = np.zeros((3, len(model_list)))
        result_score = pd.DataFrame(result_score)
        result_score.index = ['MAE', 'RMSE', 'R2']

        for i, (name, model) in enumerate(model_list.items()):
            if details >= 1:
                print('Valid model ', i + 1, ' model name : ', name)
            y_test_pred = model.predict(x_test)
            MAE = sm.mean_absolute_error(y_test, y_test_pred)
            RMSE = np.sqrt(sm.mean_squared_error(y_test, y_test_pred))
            R2 = sm.r2_score(y_test, y_test_pred)
            if details >= 2:
                print('MAE:', MAE)
                print('RMSE:', RMSE)
                print('R square:', R2)
                print('\n')
            P.iloc[:, i] = y_test_pred
            result_score.iloc[0, i] = MAE
            result_score.iloc[1, i] = RMSE
            result_score.iloc[2, i] = R2
            cols.append(name)
        P.columns = cols
        result_score.columns = cols
        return P, result_score

    def estimate_base(self, model_list, x_train, y_train, x_valid, y_valid, details=0):
        print('Estimating Train Set...')
        _, result_score_train = self.estimate(model_list, x_train, y_train, details)

        print('Estimating Valid/Test Set...')
        P, result_score_valid = self.estimate(model_list, x_valid, y_valid, details)

        if (details >= 1):
            print('result_score_train')
            print(result_score_train)
            print('result_score_valid/test')
            print(result_score_valid)

        print('\nAuto Suggestion:')
        overfit = (result_score_train - result_score_valid).T
        overfit = overfit[overfit['R2'] > 0.05]
        low_score = result_score_valid.T
        low_score = low_score[low_score['R2'] < 0.9]
        if (len(overfit) > 0):
            print('The fellowing models may have overfitting, please consider modifying them.')
            print('Train score - Valid score:')
            print(overfit)
        if (len(low_score) > 0):
            print('The fellowing models have lower scores, please consider removing them.')
            print(low_score)
        if (len(overfit) == 0 and len(low_score) == 0):
            print('All models performed well!')

        print('Base Models Correlation Matrix:')
        corrmat = P.corr()
        f, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(corrmat, square=True, annot=True)
        return P

    def estimate_meta(self, meta_model, x_train, y_train, x_valid, y_valid, details=0):
        print('Estimating Train Set...')
        _, result_score_train = self.estimate({'meta_model': meta_model}, x_train, y_train, details)

        print('Estimating Valid/Test Set...')
        _, result_score_valid = self.estimate({'meta_model': meta_model}, x_valid, y_valid, details)

        if (details >= 1):
            print('result_score_train')
            print(result_score_train)
            print('result_score_valid/test')
            print(result_score_valid)

        print('\nAuto Suggestion:')
        overfit = (result_score_train - result_score_valid).T
        overfit = overfit[overfit['R2'] > 0.05]
        low_score = result_score_valid.T
        low_score = low_score[low_score['R2'] < 0.9]
        if (len(overfit) > 0):
            print('The model may have overfitting, please consider modifying it.')
            print('Train score - Valid score:')
            print(overfit)
        if (len(low_score) > 0):
            print('Note:The model have low scores')
            print(low_score)
        if (len(overfit) == 0 and len(low_score) == 0):
            print('All models performed well!')

    def predict(self, model, x_pred):
        y_pred = model.predict(x_pred)
        return y_pred

    def fit(self, x_train, y_train, auto_scoring='r2', cv=5, n_jobs=1, details=1):
        print('Start Auto Train! Leave it all to me 😊!')
        base_models = self.get_base_models()
        best_models = self.train(base_models,
                                 x_train,
                                 y_train,
                                 'base',
                                 cv=cv,
                                 auto_scoring=auto_scoring,
                                 n_jobs=n_jobs,
                                 details=details,
                                 auto_parameter=True)
        print("We are training the META model, please wait(It may take a long time)...")
        meta_learner = self.train(best_models,
                                  x_train,
                                  y_train,
                                  'meta',
                                  cv=cv,
                                  auto_scoring=auto_scoring,
                                  n_jobs=n_jobs,
                                  details=details,
                                  auto_parameter=True)
        print('All Done 👌!')
        return meta_learner