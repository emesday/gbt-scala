package com.github.mskimm.gbt.xgboost

import com.github.mskimm.gbt.{GBTModel, Vectors}
import com.github.mskimm.testing.SparkSessionProvider
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.linalg.{DenseVector => MlDenseVector, SparseVector => MlSparseVector, Vector => MlVector, Vectors => MlVectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalatest.{FunSuite, Matchers}

class XGBoostSparkSuite extends FunSuite with Matchers with SparkSessionProvider {

  def toDataset(ds: Dataset[String]): DataFrame = {
    import ds.sparkSession.implicits._
    ds
      .map { line =>
        val xs = line.split("\t")
        val label = xs(0).toDouble
        val features = xs.drop(1).map(_.toDouble)
        (label, MlVectors.dense(features))
      }
      .toDF("label", "features")
  }

  test("xgboost4j-spark") {
    import spark.implicits._
    val tr = toDataset(spark.read.text("tests/binary.train").as[String])
    val va = toDataset(spark.read.text("tests/binary.test").as[String])

    val model = new XGBoostClassifier()
      .setNumWorkers(1)
      .setNumRound(10)
      .setMaxDepth(5)
      .setEta(0.1)
      .fit(tr)

    val dump = model.nativeBooster.getModelDump()
    val dumpIterator = dump.iterator.zipWithIndex.flatMap { case (booster, index) =>
      Iterator(s"booster[$index]:") ++ booster.split("\n")
    }

    val model2 = XGBoostModel.load(dumpIterator, GBTModel.Classification)
    val bcastModel = spark.sparkContext.broadcast(model2)
    val rawPredictUDF = udf { features: Any =>
      val vector = features.asInstanceOf[MlVector] match {
        case d: MlDenseVector => Vectors.dense(d.values.map(_.toFloat))
        case s: MlSparseVector => Vectors.sparse(s.indices, s.values.map(_.toFloat))
      }
      val prediction = bcastModel.value.predict(vector).toDouble
      MlVectors.dense(Array(-prediction, prediction))
    }

    val withPredictions = va
      .withColumn("rawPrediction", rawPredictUDF(col("features")))

    val auc = new BinaryClassificationEvaluator()
      .evaluate(model.transform(va))

    val auc2 = new BinaryClassificationEvaluator()
      .evaluate(withPredictions)

    println(s"AUC of test dataset: $auc")
    println(s"AUC of test dataset: $auc2")

    auc2 shouldBe auc
  }

}
