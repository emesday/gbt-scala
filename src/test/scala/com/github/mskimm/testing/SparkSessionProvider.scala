package com.github.mskimm.testing
import org.apache.spark.sql.SparkSession
import org.scalatest.{Suite, SuiteMixin}

trait SparkSessionProvider extends SuiteMixin { this: Suite =>

  lazy val spark: SparkSession = SparkSession
    .builder
    .master("local[*]")
    .appName("spark-session-provider")
    .getOrCreate()

}
