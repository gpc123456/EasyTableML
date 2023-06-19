# EasyTableML:An automated machine learning framework for tabular data
  
EasyTableML is an automated machine learning framework suitable for tabular data, which is very convenient to use. You only need 1-2 lines of code to train an excellent machine learning model.  

## Install  
Step one : Clone this repository using `git clone` or download this repository  

Step two : Using `pip install -r requirements.txt` to install dependencies  

Step three : Feel free to use 😊  
  
## QuickStart
### Training model
You can train a machine learning model with your data in just four simple steps: 

Step one : Import module  
`import EasyTableML`  

Step two : Initialize the EasyTableMLRegression object  
`easyml=EasyTableML.EasyTableMLRegression()`

Step three (optional) : Extract automatic features  
`x_train, x_test, y_train=easyml.auto_features(x_train,  x_test, y_train, n_jobs="The number of your cpu cores")`  
**Note:** `x_train` `x_test,`  `y_train` Must be `pandas.DataFrame`

Step Four : Train model  
`model=easyml.fit(x_train,y_train,n_jobs="The number of your cpu cores")`
### Predicting data
Use `easyml.predict(model, x_test)` to make predictions about data
## API reference
#### auto_features( x_train, x_test, y_train, features_number=20, n_jobs=1)
**Function**: Automatic extraction of training features, batch normalization of data, improve the accuracy of subsequent training process  

**Parameters**
`x_train`:Training features. The type must be *pandas.DataFrame*. It is recommended to encode the text and remove null values before input.  

`x_test`:Test features (predicted features).  The type must be *pandas.DataFram*. It is recommended to encode the text and remove null values before input.  

`y_train`:Training labels. The type must be *pandas.DataFrame*.  

`features_number`:The number of features to be expanded in automatic feature extraction. The higher the value, the higher the model accuracy, but the slower the training speed. The default value is *20*.  

`n_jobs`:The number of resources used in automatic feature extraction. It is recommended to set to the number of CPU cores. The default value is *1*.  

**Return**  
Transformed `x_train`, `x_test`, `y_train`, type : `numpy.array`  

---

#### get_base_models()  
**Function**: Gets the default base learning model.  

**Parameters**  
None.  

**Return**  
Model dictionary `{'model_name':model_object}`  

---

#### train(model_list,x_train,y_train,train_type,meta_model=None,cv=5,auto_custom_parameters=None,n_jobs=1,auto_scoring='r2',details=1,auto_parameter=False):  
**Function**: Training model.  

**Parameters**  
`model_list`:Model dictionary *{'model_name':model_object}*.  

`x_train`:Training features. The type must be *numpy.array*.  

`y_train`:Training labels. The type must be *numpy.array*. The shape must be *(labels_number,)*  

`train_type`:*'base'* or *'meta'*. 'base' trains the base model and 'meta' trains the metamodel.  

`meta_model`:If it is not None, the default metamodel is not used for training. Use a custom metamodel. `auto_parameter` must also be set to *False* to take effect. The default value is *None*  

`cv`:The number of splits when cross-verifying. The default value is *5*  

`auto_custom_parameters`:When `auto_parameter` is set to *True*, you can customize the parameter search range by specifying this parameter. The parameter input format is *{'model_name':{'parameter_name',[parameter_range]}}*. The default value is *None*  

`n_jobs`:Resources used during training. It is recommended to set to the number of CPU cores. The default value is *1*  

`auto_scoring`:Model evaluation index in automatic parameter search. This parameter is invalid when `auto_parameter` is set to False. For the value of this parameter, see: https://scikit-learn.org/stable/modules/model_evaluation.html. The default value is *r2*  

`details`:The level of detail of the output information.The larger the value, the more detailed the output information. The default value is *1*  

`auto_parameter`:*True* or *False*. Whether to use automatic parameter search. If it is set to *False*, the automatic parameter search is not performed and the model provided in `model_list` or `meta_model` is used directly for training. If set to *True*, the optimal parameters for the model in `model_list` are searched automatically.  

**Return**  
If `train_type` is *base*, return the trained model dictionary `{'model_name':model_object}`  
If `train_type` is *meta*, return metamodel object.  

---

#### load_base_model(model_list, is_auto_parameter=True)  
**Function**: Load the locally saved base models.  

**Parameters**  
`model_list`:Model dictionary *{'model_name':model_object}*.  

`is_auto_parameter`:*True* or *False*. If *True*, loads the model saved when using automatic parameter search; If *False*, load the model that was saved when automatic parameter search was not used. The default value is *True*.  

**Return**  
Returns model dictionary with loaded data `{'model_name':model_object}`  

---

#### load_meta_learner(is_auto_parameter=True)  
**Function**: Load the locally saved meta model.  

**Parameters**  
`is_auto_parameter`:*True* or *False*. If *True*, loads the model saved when using automatic parameter search; If *False*, load the model that was saved when automatic parameter search was not used. The default value is *True*.  

**Return** 
Returns metamodel with loaded data.  

---

#### estimate_base(model_list, x_train, y_train, x_valid, y_valid, details=0)  
**Function**: Evaluate the training effect of the base models and output MAE, RMSE and R2 scores of the base model.  

**Parameters**  
`model_list`:Model dictionary *{'model_name':model_object}*.  

`x_train`:Training features. The type must be *numpy.array*.  

`y_train`:Training labels. The type must be *numpy.array*. The shape must be *(labels_number,)* 

`x_valid`:Test or validation features. The type must be *numpy.array*.    

`y_valid`:Test or validation labels. The type must be *numpy.array*. The shape must be *(labels_number,)*   

`details`:The level of detail of the output information.The larger the value, the more detailed the output information. The default value is *0*  

**Return**  
None. 

---
#### estimate_meta(meta_model, x_train, y_train, x_valid, y_valid, details=0)  
**Function**: Evaluate the training effect of the metamodel and output MAE, RMSE and R2 scores of the base model.  

**Parameters**  
`meta_model`:Metamodel object.  

`x_train`:Training features. The type must be *numpy.array*.  

`y_train`:Training labels. The type must be *numpy.array*. The shape must be *(labels_number,)* 

`x_valid`:Test or validation features. The type must be *numpy.array*.  

`y_valid`:Test or validation labels. The type must be *numpy.array*. The shape must be *(labels_number,)*   

`details`:The level of detail of the output information.The larger the value, the more detailed the output information. The default value is *0*  

**Return**  
None.  

---  

#### predict(model, x_pred)  
**Function**:Predict the value of the label.  

**Parameters**  
`model`:A model object.  

`x_pred`:Features to be predicted. The type must be *numpy.array*.  

**Return**  
Predicted results. The type is `numpy.array`.  

---  

#### fit(x_train, y_train, auto_scoring='r2', cv=5, n_jobs=1, details=1)  
**Function**:One line of code gets the training model.  

**Parameters**  
`x_train`:Training features. The type must be *numpy.array*.  

`y_train`:Training labels. The type must be *numpy.array*. The shape must be *(labels_number,)*  

`auto_scoring`:Model evaluation index in automatic parameter search. This parameter is invalid when `auto_parameter` is set to False. For the value of this parameter, see: https://scikit-learn.org/stable/modules/model_evaluation.html. The default value is *r2*  

`cv`:The number of splits when cross-verifying. The default value is *5*  

`n_jobs`:Resources used during training. It is recommended to set to the number of CPU cores. The default value is *1*  

`details`:The level of detail of the output information.The larger the value, the more detailed the output information. The default value is *1*  