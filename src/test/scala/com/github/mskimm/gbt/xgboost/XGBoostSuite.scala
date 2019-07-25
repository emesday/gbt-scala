package com.github.mskimm.gbt.xgboost

import java.nio.file.{Files, Paths}

import com.github.mskimm.gbt.{BinaryClassificationMetrics, LabeledPoint, Vectors}
import org.scalatest.{FunSuite, Inspectors, Matchers}

class XGBoostSuite extends FunSuite with Matchers with Inspectors {

  import com.github.mskimm.gbt.GBTSuite._

  val xgboostModelPath = "tests/xgboost.binary.model.dump"
  val xgboostPredPath = "tests/xgboost.binary.pred.txt"

  test("xgboost - data preparation - if fail, run `cd tests && python3 xgboost_example.py` first") {
    Files.exists(Paths.get(xgboostModelPath)) shouldBe true
    Files.exists(Paths.get(xgboostPredPath)) shouldBe true
  }

  test("xgboost - prediction") {
    val is = Files.newInputStream(Paths.get("tests/xgboost.binary.model.dump"))
    val model = XGBoostModel.load(is)
    is.close()

    val test = "tests/binary.test"
    val testDataset = read(test, tsvParser)

    val scores = testDataset.map(_.point).map(model.predict)
    val labels = testDataset.map(_.label)
    val metrics = new BinaryClassificationMetrics(scores, labels)
    val auc = metrics.areaUnderROC()

    println(s"AUC of test dataset: $auc")
    auc shouldBe 0.7900f +- 1e-4f

    val xgboostPredictions = read(xgboostPredPath,
      line => LabeledPoint(line.toFloat, Vectors.empty)).map(_.label)

    forAll(scores.zip(xgboostPredictions)) { case (actual, expected) => actual shouldBe expected +- 1e-4f }
  }

}
