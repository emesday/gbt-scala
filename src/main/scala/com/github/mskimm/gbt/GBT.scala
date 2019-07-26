package com.github.mskimm.gbt

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

case class Accumulator(
  g: Float,
  h: Float,
  private var gl: Float = 0,
  private var hl: Float = 0,
  private var gr: Float = 0,
  private var hr: Float = 0,
  var bestGain: Float = 0,
  var bestFeature: Int = 0,
  var bestVal: Float = 0,
  var bestLeftInstanceIds: Array[Int] = Array.emptyIntArray,
  var bestRightInstanceIds: Array[Int] = Array.emptyIntArray) {

  def reset(): Unit = {
    gl = 0
    hl = 0
    gr = 0
    hr = 0
  }

  def update(x: Float, y: Float, lambda: Float): Unit = {
    gl += x
    hl += y
    gr = g - gl
    hr = h - hl
  }

  def updateSplit(lambda: Float, feature: Int, value: Float, sortedInstanceIds: Array[Int], j: Int): Unit = {
    val currentGain = getSplitGain(lambda)
    if (currentGain > bestGain) {
      bestGain = currentGain
      bestFeature = feature
      bestVal = value
      val split = sortedInstanceIds.splitAt(j + 1)
      bestLeftInstanceIds = split._1
      bestRightInstanceIds = split._2
    }
  }

  def getSplitGain(lambda: Float): Float = {
    def calcTerm(g: Float, h: Float) = {
      g * g / (h + lambda)
    }

    calcTerm(gl, hl) + calcTerm(gr, hr) - calcTerm(g, h)
  }

}

case class ActiveDataset(
  private val instances: Array[LabeledPoint],
  private val active: Array[Int] = Array.emptyIntArray) {

  def argSortBy(featureIndex: Int): Array[Int] = {
    val zipWithIndex = if (active.isEmpty) {
      instances.map(_.point(featureIndex)).zipWithIndex
    } else {
      active.map(index => instances(index).point(featureIndex) -> index)
    }
    zipWithIndex.sortBy(_._1).map(_._2)
  }

  def getValue(instanceIndex: Int, featureIndex: Int): Float = {
    instances(instanceIndex).point(featureIndex)
  }

  def getAccumulator(grad: Array[Float], hess: Array[Float]): Accumulator = {
    Accumulator(activeSum(grad), activeSum(hess))
  }

  def activeSum(values: Array[Float]): Float = {
    var sum = 0f
    if (active.isEmpty) {
      var index = 0
      while (index < instances.length) {
        sum += values(index)
        index += 1
      }
    } else {
      for (index <- active) {
        sum += values(index)
      }
    }
    sum
  }

}

class GBT(
  numBoostRounds: Int,
  maxDepth: Int,
  lambda: Float,
  featureSize: Int,
  minSplitGain: Float,
  learningRate: Float,
  earlyStoppingRounds: Int,
  func: ModelFunctions,
  seed: Long = Random.nextLong()) {

  def train(
    instances: Array[LabeledPoint],
    evalInstances: Array[LabeledPoint] = Array.empty[LabeledPoint]): TreeModel = {
    val elapsed = new StopWatch

    println(s"Start training ... (tr: ${instances.length}, va: ${evalInstances.length})")

    val trees = new ArrayBuffer[Tree]()
    var shrinkageRate = 1f

    val activeDataset = ActiveDataset(instances)
    val grad = new Array[Float](instances.length)
    val hess = new Array[Float](instances.length)

    var bestValLoss = Float.MaxValue
    var bestRound = 0
    var round = 0
    while (round < numBoostRounds) {
      val elapsed = new StopWatch
      computeGradient(instances, trees, grad, hess)
      trees += buildTree(activeDataset, grad, hess, shrinkageRate)
      if (round > 0) {
        shrinkageRate *= learningRate
      }
      val trLoss = computeLoss(trees, instances)
      val vaLoss = computeLoss(trees, evalInstances)
      if (evalInstances.nonEmpty) {
        println(f"round ${round + 1}%3s, trloss: $trLoss%.4f, valoss: $vaLoss%.4f, elapsed: $elapsed")
      } else {
        println(f"round ${round + 1}%3s, trloss: $trLoss%.4f, elapsed: $elapsed")
      }

      if (evalInstances.nonEmpty && vaLoss < bestValLoss) {
        bestValLoss = vaLoss
        bestRound = round
      }

      if (round - bestRound >= earlyStoppingRounds) {
        print("Early stopping, the best round was:")
        println(f"round ${bestRound + 1}%3s, valoss: $bestValLoss%.4f")
        round = numBoostRounds // break
      }
      round += 1
    }

    println(s"Training finished. Elapsed: $elapsed secs")
    func.createModel(trees)
  }

  def computeGradient(
    instances: Array[LabeledPoint], trees: Seq[Tree], grad: Array[Float], hess: Array[Float]): Unit = {
    require(instances.length <= grad.length)
    require(instances.length <= hess.length)
    if (trees.isEmpty) {
      implicit val random: Random = new Random(seed)
      for (i <- instances.indices) {
        grad(i) = func.initGrad
        hess(i) = func.initHess
      }
    } else {
      val m = func.createModel(trees)
      var i = 0
      for (LabeledPoint(y, point) <- instances) {
        val yh = m.predict(point)
        grad(i) = func.computeGrad(y, yh)
        hess(i) = func.computeHess(y, yh)
        i += 1
      }
    }
  }

  def computeLoss(trees: Seq[Tree], instances: Array[LabeledPoint]): Float = {
    val m = func.createModel(trees)
    var loss = 0f
    for (LabeledPoint(y, point) <- instances) {
      val yh = m.predict(point)
      loss += func.computeLoss(y, yh) / instances.length
    }
    loss
  }

  def buildTree(dataset: ActiveDataset, grad: Array[Float], hess: Array[Float], shrinkageRate: Float): Tree = {
    def inner(
      treeNodes: ArrayBuffer[TreeNode],
      instances: ActiveDataset,
      depth: Int): TreeNode = {
      if (depth > maxDepth) {
        val value = computeLeafWeight(grad, hess, lambda, instances) * shrinkageRate
        LeafNode(value)
      } else {
        val accum = instances.getAccumulator(grad, hess)
        for (feature <- 0 until featureSize) {
          accum.reset()
          val sortedIds = instances.argSortBy(feature)
          for (j <- sortedIds.indices) {
            accum.update(grad(sortedIds(j)), hess(sortedIds(j)), lambda)
            accum.updateSplit(lambda, feature, instances.getValue(sortedIds(j), feature), sortedIds, j)
          }
        }
        if (accum.bestGain < minSplitGain) {
          val value = computeLeafWeight(grad, hess, lambda, instances) * shrinkageRate
          LeafNode(value)
        } else {
          val left = inner(treeNodes, instances.copy(active = accum.bestLeftInstanceIds), depth + 1)
          val right = inner(treeNodes, instances.copy(active = accum.bestRightInstanceIds), depth + 1)
          treeNodes += left
          treeNodes += right
          InternalNode(accum.bestFeature, accum.bestVal, treeNodes.length - 2, treeNodes.length - 1)
        }
      }
    }

    val treeNodes = new ArrayBuffer[TreeNode]()
    treeNodes += inner(treeNodes, dataset, 0)
    new Tree(treeNodes)
  }

  def computeLeafWeight(grad: Array[Float], hess: Array[Float], lambda: Float, instances: ActiveDataset): Float = {
    -instances.activeSum(grad) / (instances.activeSum(hess) + lambda)
  }
}
