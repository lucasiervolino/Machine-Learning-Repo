import lightgbm as lgb
import numpy as np

# dataset
train = pd.read_csv(path_train)
test = pd.read_csv(path_test)

from sklearn.model_selection import train_test_split
X_train, val_X, y_train, val_y = train_test_split(Data_final, y_train, test_size = 0.2, random_state = 9)

# general function to create lgbm model with custom parameters
def call_lgbm(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression_l1",
        "metric" : "rmse",
        "num_leaves" : 50,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.6,
        "bagging_frequency" : 2,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42,
#        "max_bin": 50,
#        'subsample_for_bin':300,
#        "reg_alpha":0.01,
#        "reg_lambda":0.06,
#        "max_depth": 5,
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=500, 
                      verbose_eval=150, 
                      evals_result=evals_result)
    
    pred_test_y = (model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result

pred_test, model, evals_result = call_lgbm(X_train, y_train, val_X, val_y, val_X)
print("Training Completed")

#Evaluation
from sklearn.metrics import mean_squared_error
rms_r2 = np.sqrt(mean_squared_error((val_y), (pred_test)))
from sklearn.metrics import r2_score
r2 = r2_score((val_y), (pred_test))

#%%
# Evaluation of feature importance by the lgb model
print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:15])
#%%
flist = [x for x in x_train.columns if not x in ['ID','target']]

model.save_model('mode.txt')
model = lgb.Booster(model_file='mode.txt')
