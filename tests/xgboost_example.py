import pandas as pd
import xgboost as xgb

train = pd.read_csv('binary.train', sep='\t', header=None)
test = pd.read_csv('binary.test', sep='\t', header=None)
train = xgb.DMatrix(train.drop(0, axis=1).values, label=train[0].values)
test = xgb.DMatrix(test.drop(0, axis=1).values, label=test[0].values)

print("binary:logistic - tr: binary.train, va: binary.test")

param = {'max_depth': 5, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': ['auc']}
watchlist  = [(train, 'train'), (test, 'eval')]
num_round = 10
bst = xgb.train(param, train, num_round, evals=watchlist)
bst.dump_model("xgboost.binary.model.dump")

with open("xgboost.binary.pred.txt", "w") as o:
    for pred in bst.predict(test):
        o.write('%f\n' % pred)

