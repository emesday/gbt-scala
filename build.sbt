name := "gbt-scala"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.4.3" % Test,
  "org.apache.spark" %% "spark-sql" % "2.4.3" % Test,
  "org.apache.spark" %% "spark-mllib" % "2.4.3" % Test,
  "ml.dmlc" % "xgboost4j-spark" % "0.90" % Test,
  "org.scalatest" %% "scalatest" % "3.0.8" % Test
)

