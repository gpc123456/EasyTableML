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
`x_train`:Training features. must be in the format of `pandas.DataFrame`. It is recommended to encode the text and remove null values before input.  

`x_test`:Test features (predicted features).  must be in the format of `pandas.DataFrame`. It is recommended to encode the text and remove null values before input.  

`y_train`:Training labels. must be in the format of `pandas.DataFrame`.  

`features_number`:The number of features to be expanded in automatic feature extraction. The higher the value, the higher the model accuracy, but the slower the training speed. The default value is 20.  

`n_jobs`:The number of resources used in automatic feature extraction. It is recommended to set to the number of CPU cores. The default value is 1.  

**Return**  
Transformed `x_train`, `x_test`, `y_train`, type : `numpy.array`  

#### get_base_models()  
**Function**: Gets the default base learning model.  

**Parameters**  
None.  

**Return**  
Model dictionary `{'model_name':model object}`