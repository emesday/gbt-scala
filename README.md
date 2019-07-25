# gbt-scala

gbt-scala is a predictor of existing models (of XGBoost, ...) written in Scala

# Demo

```
$ (cd tests && python3 xgboost_example.py)
binary:logistic - tr: binary.train, va: binary.test
[0]     train-auc:0.756447      eval-auc:0.761779
[1]     train-auc:0.780245      eval-auc:0.787498
[2]     train-auc:0.783758      eval-auc:0.779509
[3]     train-auc:0.789038      eval-auc:0.78799
[4]     train-auc:0.793306      eval-auc:0.788538
[5]     train-auc:0.798081      eval-auc:0.790135
[6]     train-auc:0.801384      eval-auc:0.791884
[7]     train-auc:0.804421      eval-auc:0.789353
[8]     train-auc:0.806241      eval-auc:0.7872
[9]     train-auc:0.809945      eval-auc:0.790094
$ head tests/xgboost.binary.model.dump 
booster[0]:
0:[f25<1.06649995] yes=1,no=2,missing=1
        1:[f25<0.661499977] yes=3,no=4,missing=3
                3:[f9<1.13750005] yes=7,no=8,missing=7
                        7:[f13<1.29500008] yes=15,no=16,missing=15
                                15:[f27<0.855499983] yes=29,no=30,missing=29
                                        29:leaf=-0.0276621785
                                        30:leaf=-0.0791666731
                                16:[f21<0.859499991] yes=31,no=32,missing=31
                                        31:leaf=-0.0521739125
$ sbt "testOnly *XGBoostSuite"
...
[info] XGBoostSuite:
[info] - xgboost - data preparation - if fail, run `cd tests && python3 xgboost_example.py` first
[info] - xgboost - AUC = 0.79
[info] - xgboost - predictions are the same
[info] Run completed in 624 milliseconds.
[info] Total number of tests run: 3
[info] Suites: completed 1, aborted 0
[info] Tests: succeeded 3, failed 0, canceled 0, ignored 0, pending 0
[info] All tests passed.
...
```

# Usage

Train a model using XGBoost

```
$ (cd tests && python3 xgboost_example.py)
```

Then,

```
scala> import com.github.mskimm.gbt.xgboost._
import com.github.mskimm.gbt.xgboost._

scala> import com.github.mskimm.gbt._
import com.github.mskimm.gbt._

scala> val model = XGBoostModel.load("tests/xgboost.binary.model.dump", Classification) // or Regression
model: com.github.mskimm.gbt.TreeModel = com.github.mskimm.gbt.ClassificationTreeModel@ccc8b16

scala> val pos = Vectors.dense(Array[Float](0.6f, 0.2f, -0.4f, 0.8f, 0.3f, 0.8f, -1.1f, -0.7f, 2.1f, 1.0f, -0.2f, 1.4f, 0.0f, 1.5f, 1.8f, 1.6f, 0.0f, 1.5f, -0.1f, -0.7f, 3.1f, 0.9f, 1.0f, 0.9f, 0.8f, 0.5f, 0.9f, 0.7f))
pos: com.github.mskimm.gbt.Vector = com.github.mskimm.gbt.DenseVector@25183158

scala> val neg = Vectors.dense(Array[Float](0.3f, 1.8f, 1.0f, 1.0f, 0.3f, 1.5f, -0.9f, 1.7f, 0.0f, 0.9f, -1.9f, -0.2f, 0.0f, 1.5f, 0.4f, -0.3f, 2.5f, 0.8f, -0.4f, 0.6f, 3.1f, 0.6f, 0.9f, 0.9f, 0.8f, 0.8f, 1.1f, 1.1f))
neg: com.github.mskimm.gbt.Vector = com.github.mskimm.gbt.DenseVector@666a4c2f

scala> model.predict(pos)
res2: Float = 0.6129771

scala> model.predict(neg)
res3: Float = 0.47721037
```

# Features

 - Predictor
     - [XGBoost](https://github.com/dmlc/xgboost)
         - dump by `dump_model` (see [`tests/xgboost_example.py`](https://github.com/mskimm/gbt-scala/blob/master/tests/xgboost_example.py) and [`XGBoostSuite`](https://github.com/mskimm/gbt-scala/blob/master/src/test/scala/com/github/mskimm/gbt/xgboost/XGBoostSuite.scala))
     - [TBD] [LightGBM](https://github.com/microsoft/LightGBM)
     - [TBD] [CatBoost](https://github.com/catboost/catboost)
         
 - Trainer (inspired by [TinyGBT](https://github.com/lancifollia/tinygbt))
     - Exact greedy algorithm
     - [TBD] Approximate greedy algorithm using quantile sketch and gradient histogram
     - Regression with L2 loss
     - Classification with log loss
     
# References
 - T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.
 - [XGBoost](https://github.com/dmlc/xgboost)
 - https://github.com/lancifollia/tinygbt
 - https://github.com/dimleve/tinygbt


