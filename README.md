# gbt-scala

This is a Scala version of [TinyGBT](https://github.com/lancifollia/tinygbt).

# Demo

```bash
$ sbt test

Start training ... (tr: 7000, va: 500)
round   1, trloss: 0.7619, valoss: 0.7920, elapsed: 1.15 secs
round   2, trloss: 0.2096, valoss: 0.2136, elapsed: 0.69 secs
round   3, trloss: 0.1996, valoss: 0.2059, elapsed: 0.60 secs
round   4, trloss: 0.1967, valoss: 0.2055, elapsed: 0.52 secs
round   5, trloss: 0.1959, valoss: 0.2050, elapsed: 0.52 secs
round   6, trloss: 0.1957, valoss: 0.2049, elapsed: 0.48 secs
round   7, trloss: 0.1957, valoss: 0.2049, elapsed: 0.50 secs
round   8, trloss: 0.1957, valoss: 0.2048, elapsed: 0.50 secs
round   9, trloss: 0.1957, valoss: 0.2048, elapsed: 0.50 secs
round  10, trloss: 0.1957, valoss: 0.2048, elapsed: 0.50 secs
round  11, trloss: 0.1957, valoss: 0.2048, elapsed: 0.50 secs
round  12, trloss: 0.1957, valoss: 0.2048, elapsed: 0.50 secs
round  13, trloss: 0.1957, valoss: 0.2048, elapsed: 0.50 secs
round  14, trloss: 0.1957, valoss: 0.2048, elapsed: 0.51 secs
round  15, trloss: 0.1957, valoss: 0.2048, elapsed: 0.51 secs
round  16, trloss: 0.1957, valoss: 0.2048, elapsed: 0.52 secs
Early stopping, the best round was:round  11, valoss: 0.2048
Training finished. Elapsed: 9.01 secs secs
RMSE of test dataset: 0.45258915
Start training ... (tr: 919, va: 453)
round   1, trloss: 0.7868, valoss: 0.7732, elapsed: 0.01 secs
round   2, trloss: 0.1438, valoss: 0.1763, elapsed: 0.02 secs
round   3, trloss: 0.1024, valoss: 0.1342, elapsed: 0.01 secs
round   4, trloss: 0.0948, valoss: 0.1260, elapsed: 0.01 secs
round   5, trloss: 0.0926, valoss: 0.1237, elapsed: 0.01 secs
round   6, trloss: 0.0920, valoss: 0.1230, elapsed: 0.01 secs
round   7, trloss: 0.0918, valoss: 0.1228, elapsed: 0.01 secs
round   8, trloss: 0.0918, valoss: 0.1228, elapsed: 0.01 secs
round   9, trloss: 0.0918, valoss: 0.1228, elapsed: 0.01 secs
round  10, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
round  11, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
round  12, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
round  13, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
round  14, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
round  15, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
round  16, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
round  17, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
round  18, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
round  19, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
round  20, trloss: 0.0917, valoss: 0.1227, elapsed: 0.01 secs
Early stopping, the best round was:round  15, valoss: 0.1227
Training finished. Elapsed: 0.24 secs secs
AUC of test dataset: 0.9981339
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

 - Exact greedy algorithm
 - [TBD] Approximate greedy algorithm using quantile sketch and gradient histogram
 - Regression with L2 loss
 - Classification with log loss
 
# References
 - T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. 2016.


