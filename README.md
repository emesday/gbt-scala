# gbt-scala

This is a Scala version of [TinyGBT](https://github.com/lancifollia/tinygbt).

# Demo

- dataset:
   - regression: [LightGBM's regression example data](https://github.com/Microsoft/LightGBM/tree/master/examples/regression)
   - classification: [LightGBM's binary classification example data](https://github.com/microsoft/LightGBM/tree/master/examples/binary_classification)

```
$ sbt test
...
test gbt - regression
Start training ... (tr: 7000, va: 500)
round   1, trloss: 0.7619, valoss: 0.7920, elapsed: 0.86 secs
round   2, trloss: 0.2096, valoss: 0.2136, elapsed: 0.68 secs
round   3, trloss: 0.1996, valoss: 0.2059, elapsed: 0.64 secs
round   4, trloss: 0.1967, valoss: 0.2055, elapsed: 0.51 secs
round   5, trloss: 0.1959, valoss: 0.2050, elapsed: 0.45 secs
round   6, trloss: 0.1957, valoss: 0.2049, elapsed: 0.48 secs
round   7, trloss: 0.1957, valoss: 0.2049, elapsed: 0.54 secs
round   8, trloss: 0.1957, valoss: 0.2048, elapsed: 0.52 secs
round   9, trloss: 0.1957, valoss: 0.2048, elapsed: 0.51 secs
round  10, trloss: 0.1957, valoss: 0.2048, elapsed: 0.50 secs
round  11, trloss: 0.1957, valoss: 0.2048, elapsed: 0.54 secs
round  12, trloss: 0.1957, valoss: 0.2048, elapsed: 0.83 secs
round  13, trloss: 0.1957, valoss: 0.2048, elapsed: 0.63 secs
round  14, trloss: 0.1957, valoss: 0.2048, elapsed: 0.72 secs
round  15, trloss: 0.1957, valoss: 0.2048, elapsed: 0.64 secs
round  16, trloss: 0.1957, valoss: 0.2048, elapsed: 0.51 secs
Early stopping, the best round was:round  11, valoss: 0.2048
Training finished. Elapsed: 9.56 secs secs
RMSE of test dataset: 0.45258915
test gbt - classification
Start training ... (tr: 7000, va: 500)
round   1, trloss: 0.8405, valoss: 0.8536, elapsed: 0.59 secs
round   2, trloss: 0.5646, valoss: 0.5751, elapsed: 0.78 secs
round   3, trloss: 0.5541, valoss: 0.5821, elapsed: 0.63 secs
round   4, trloss: 0.5468, valoss: 0.5795, elapsed: 0.55 secs
round   5, trloss: 0.5446, valoss: 0.5785, elapsed: 0.54 secs
round   6, trloss: 0.5441, valoss: 0.5783, elapsed: 0.86 secs
round   7, trloss: 0.5440, valoss: 0.5782, elapsed: 0.90 secs
round   8, trloss: 0.5439, valoss: 0.5782, elapsed: 0.69 secs
round   9, trloss: 0.5439, valoss: 0.5782, elapsed: 0.74 secs
round  10, trloss: 0.5439, valoss: 0.5782, elapsed: 0.52 secs
round  11, trloss: 0.5439, valoss: 0.5782, elapsed: 0.48 secs
round  12, trloss: 0.5439, valoss: 0.5782, elapsed: 0.47 secs
Early stopping, the best round was:round   2, valoss: 0.5751
Training finished. Elapsed: 7.74 secs secs
AUC of test dataset: 0.7631097
...
```

# Usage

```scala
val trainDataset: Array[LabeledPoint] = _
val testDataset: Array[LabeledPoint] = _
val featureSize = 100
val gbt = new GBT(
  numBoostRounds = 20,
  maxDepth = 5,
  lambda = 1,
  featureSize = featureSize,
  minSplitGain = 0.1f,
  learningRate = 0.3f,
  earlyStoppingRounds = 5,
  func = Regression)
  
val model = gbt.train(trainDataset, testDataset)
val predictions = testDataset.map(_.point).map(model.predict)
```

Note: for classification, assign `func` to `Classification` instead of `Regression` 

# Features

 - Trainer
     - Exact greedy algorithm
     - [TBD] Approximate greedy algorithm using quantile sketch and gradient histogram
     - Regression with L2 loss
     - Classification with log loss
 - Predictor
     - XGBoost Model
         - dump by `dump_model` (see [`tests/xgboost_example.py`](https://github.com/mskimm/gbt-scala/blob/master/tests/xgboost_example.py) and [`XGBoostSuite`](https://github.com/mskimm/gbt-scala/blob/master/src/test/scala/com/github/mskimm/gbt/xgboost/XGBoostSuite.scala))
 
# References
 - T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.
 - https://github.com/lancifollia/tinygbt
 - https://github.com/dimleve/tinygbt


