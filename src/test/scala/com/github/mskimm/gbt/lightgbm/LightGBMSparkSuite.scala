package com.github.mskimm.gbt.lightgbm

import com.github.mskimm.testing.{Datasets, SparkSessionProvider}
import com.microsoft.ml.spark.LightGBMClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.scalatest.{FunSuite, Matchers}

class LightGBMSparkSuite extends FunSuite with Matchers with SparkSessionProvider {

  test("mmlspark") {
    val (tr, va) = Datasets.binary

    val model = new LightGBMClassifier()
      .setLearningRate(0.1)
      .setNumIterations(10)
      .setNumLeaves(63)
      .fit(tr)

    val auc = new BinaryClassificationEvaluator().evaluate(model.transform(va))
    println(auc)
  }

}
