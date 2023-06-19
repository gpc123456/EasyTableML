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
**Note:** `y_train` Must be `pandas.DataFrame`

Step Four : Train model  
`model=easyml.fit(x_train,y_train,n_jobs="The number of your cpu cores")`
### Predicting data
Use `easyml.predict(model, x_test)` to make predictions about data