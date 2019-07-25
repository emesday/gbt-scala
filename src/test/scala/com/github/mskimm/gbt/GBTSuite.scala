package com.github.mskimm.gbt

import org.scalatest.{FunSuite, Matchers}

object GBTSuite {

  def read(path: String, parser: String => LabeledPoint): Array[LabeledPoint] = {
    val source = io.Source.fromFile(path)
    val result = source.getLines().map(parser).toArray
    source.close()
    result
  }

  val tsvParser: String => LabeledPoint = { line: String =>
    val xs = line.split("\\s+")
    LabeledPoint(xs(0).toFloat, Vectors.dense(xs.drop(1).map(_.toFloat)))
  }

}

class GBTSuite extends FunSuite with Matchers {

  import GBTSuite._

  test("gbt - regression") {
    println("test gbt - regression")
    val featureSize = 28
    val train = "tests/regression.train"
    val test = "tests/regression.test"

    val trainDataset = read(train, tsvParser)
    val testDataset = read(test, tsvParser)

    val gbt = new GBT(
      numBoostRounds = 20,
      maxDepth = 5,
      lambda = 1f,
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
    println("test gbt - classification")
    val featureSize = 28
    val train = "tests/binary.train"
    val test = "tests/binary.test"

    val trainDataset = read(train, tsvParser)
    val testDataset = read(test, tsvParser)

    val gbt = new GBT(
      numBoostRounds = 100,
      maxDepth = 5,
      lambda = 1f,
      featureSize = featureSize,
      minSplitGain = 0.1f,
      learningRate = 0.3f,
      earlyStoppingRounds = 10,
      func = Classification,
      seed = 1235L)

    val model = gbt.train(trainDataset, testDataset)

    val scores = testDataset.map(_.point).map(model.predict)
    val labels = testDataset.map(_.label)
    val metrics = new BinaryClassificationMetrics(scores, labels)
    val auc = metrics.areaUnderROC()

    println(s"AUC of test dataset: $auc")
    auc shouldBe 0.7631f +- 1e-4f
  }

}
