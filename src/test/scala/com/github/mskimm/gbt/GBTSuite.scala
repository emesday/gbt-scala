package com.github.mskimm.gbt

import org.scalatest.{FunSuite, Matchers}

class GBTSuite extends FunSuite with Matchers {

  def read(train: String, parser: String => LabeledPoint): Array[LabeledPoint] = {
    val source = io.Source.fromFile(train)
    val result = source.getLines().map(parser).toArray
    source.close()
    result
  }

  test("gbt - regression") {
    val featureSize = 28
    val train = "tests/regression.train"
    val test = "tests/regression.test"
    val parser = { line: String =>
      val xs = line.split("\\s+")
      LabeledPoint(xs(0).toFloat, Vectors.dense(xs.drop(1).map(_.toFloat)))
    }

    val trainDataset = read(train, parser)
    val testDataset = read(test, parser)

    val gbt = new GBT(
      numBoostRounds = 20,
      maxDepth = 5,
      lambda = 1,
      featureSize = featureSize,
      minSplitGain = 0.1f,
      learningRate = 0.3f,
      earlyStoppingRounds = 5,
      func = Regression,
      seed = 1234L)

    val model = gbt.train(trainDataset, testDataset)

    val scores = testDataset.map(_.point).map(model.predict)
    val labels = testDataset.map(_.label)
    val rmse = Metrics.rmse(scores, labels)
    println(s"RMSE of test dataset: $rmse")
    rmse shouldBe 0.4525f +- 1e-4f
  }

  test("gbt - classification") {
    val featureSize = 4
    val train = "tests/data_banknote_authentication.txt.train"
    val test = "tests/data_banknote_authentication.txt.test"
    val parser = { line: String =>
      val xs = line.split(",")
      LabeledPoint(xs(0).toFloat, Vectors.dense(xs.drop(1).map(_.toFloat)))
    }

    val trainDataset = read(train, parser)
    val testDataset = read(test, parser)

    val gbt = new GBT(
      numBoostRounds = 20,
      maxDepth = 5,
      lambda = 1,
      featureSize = featureSize,
      minSplitGain = 0.1f,
      learningRate = 0.3f,
      earlyStoppingRounds = 5,
      func = Classification,
      seed = 1235L)

    val model = gbt.train(trainDataset, testDataset)

    val scores = testDataset.map(_.point).map(model.predict)
    val labels = testDataset.map(_.label)
    val metrics = new BinaryClassificationMetrics(scores, labels)
    val auc = metrics.areaUnderROC()

    println(s"AUC of test dataset: $auc")
    auc shouldBe 0.9981f +- 1e-4f
  }

}