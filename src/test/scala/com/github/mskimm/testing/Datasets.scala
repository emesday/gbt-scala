package com.github.mskimm.testing

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

object Datasets {

  def load(path: String, format: String)(
    implicit spark: SparkSession): DataFrame = {
    import spark.implicits._
    format match {
      case "tsv" =>
        spark.read.text(path)
          .as[String]
          .map { line =>
            val xs = line.split("\t")
            val label = xs(0).toDouble
            val features = xs.drop(1).map(_.toDouble)
            (label, Vectors.dense(features))
          }
          .toDF("label", "features")
      case "indices" => spark.read.parquet(path)
        spark.read.parquet(path)
          .as[(Double, Seq[Int])]
          .map { case (label, indices) =>
            (label, Vectors.sparse(1000000, indices.toArray, Array.fill(indices.length)(1.0)))
          }
          .toDF("label", "features")
    }
  }

  def load(
    trPath: String, vaPath: String, format: String)(
    implicit spark: SparkSession): (DataFrame, DataFrame) = {
    (load(trPath, format), load(vaPath, format))
  }

  def binary(implicit spark: SparkSession): (DataFrame, DataFrame) = {
    (load("tests/binary.train", "tsv"), load("tests/binary.train", "tsv"))
  }

}
