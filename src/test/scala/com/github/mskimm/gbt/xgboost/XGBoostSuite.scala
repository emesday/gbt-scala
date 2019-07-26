package com.github.mskimm.gbt.xgboost

import java.nio.file.{Files, Paths}

import com.github.mskimm.gbt._
import com.github.mskimm.gbt.evaluation.BinaryClassificationMetrics
import org.scalatest.{FunSuite, Inspectors, Matchers}

class XGBoostSuite extends FunSuite with Matchers with Inspectors {

  import com.github.mskimm.gbt.experimental.GBTSuite._

  val xgboostModelPath = "tests/xgboost.binary.model.dump"
  val xgboostPredPath = "tests/xgboost.binary.pred.txt"

  val model = XGBoostModel.load("tests/xgboost.binary.model.dump", GBTModel.Classification)

  val test: String = "tests/binary.test"
  val testDataset: Array[LabeledPoint] = read(test, tsvParser)

  val scores: Array[Float] = testDataset.map(_.point).map(model.predict)
  val labels: Array[Float] = testDataset.map(_.label)

  test("xgboost - data preparation - if fail, run `cd tests && python3 xgboost_example.py` first") {
    Files.exists(Paths.get(xgboostModelPath)) shouldBe true
    Files.exists(Paths.get(xgboostPredPath)) shouldBe true
  }

  test("xgboost - AUC = 0.79") {
    val metrics = new BinaryClassificationMetrics(scores, labels)
    val auc = metrics.areaUnderROC()
    auc shouldBe 0.79f +- 1e-4f
  }

  test("xgboost - predictions are the same") {
    val xgboostPredictions = read(xgboostPredPath,
      line => LabeledPoint(line.toFloat, Vectors.empty)).map(_.label)
    forAll(scores.zip(xgboostPredictions)) { case (actual, expected) => actual shouldBe expected +- 1e-4f }
  }

}
