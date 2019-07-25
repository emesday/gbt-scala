package com.github.mskimm.gbt.xgboost

import java.io.InputStream
import java.nio.file.{Files, Paths}

import com.github.mskimm.gbt._

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.matching.Regex

case class XGBoostNode(feature: Int, value: Float, yes: Int, no: Int, missing: Int) extends TreeNode {

  override def traverse(vector: Vector, nodes: Seq[TreeNode]): TreeNode = {
    vector.get(feature) match {
      case None => nodes(missing)
      case Some(v) if v < value => nodes(yes)
      case _ => nodes(no)
    }
  }

}

object XGBoostModel {

  private val defaultCondition: String = "<"
  private val comparator: (Int, Int) => Int = Integer.compare
  private val boosterRegex: Regex = "^booster\\[([0-9]+)\\]:$".r
  private val treeNodeRegex: Regex = "^\t*([0-9]+):\\[f([0-9]+)([<>])(.*)\\] yes=([0-9]+),no=([0-9]+),missing=([0-9]+)$".r
  private val leafNodeRegex: Regex = "^\t*([0-9]+):leaf=(.*)$".r

  def load(lines: Iterator[String], func: ModelFunctions): TreeModel = {
    var boosterId = -1
    var booster = new mutable.HashMap[Int, TreeNode]()
    val boosters = new ArrayBuffer[Tree]()
    val features = new mutable.HashSet[Int]()

    def addBooster() = {
      val size = booster.keys.max + 1
      val booster1 = new Array[TreeNode](size)
      for ((i, v) <- booster) {
        booster1(i) = v
      }
      boosters += new Tree(booster1, 0)
    }

    for (line <- lines if line.trim.nonEmpty) {
      line match {
        case boosterRegex(boosterId0) =>
          if (boosterId != -1) {
            addBooster()
          }
          boosterId = boosterId0.toInt
          require(boosterId == boosters.length)
          booster = new mutable.HashMap[Int, TreeNode]()
        case treeNodeRegex(id, feature, condition, value, yes, no, missing) if condition == defaultCondition =>
          features += feature.toInt
          booster(id.toInt) = XGBoostNode(feature.toInt, value.toFloat, yes.toInt, no.toInt, missing.toInt)
        case leafNodeRegex(id, value) =>
          booster(id.toInt) = Leaf(value.toFloat)
      }
    }
    addBooster()
    func.createModel(boosters)
  }

  def load(is: InputStream, func: ModelFunctions): TreeModel = {
    val lines = scala.io.Source.fromInputStream(is).getLines()
    load(lines, func)
  }

  def load(path: String, func: ModelFunctions): TreeModel = {
    val is = Files.newInputStream(Paths.get(path))
    val model = load(is, func)
    is.close()
    model
  }

}

