package com.github.mskimm.gbt

import scala.util.Random

trait TreeModel {

  val trees: Seq[Tree]

  def predict(vector: Vector): Float

}

trait TreeNode

sealed case class Node(feature: Int, value: Float, left: Int, right: Int) extends TreeNode

sealed case class Leaf(value: Float) extends TreeNode

class Tree(nodes: Seq[TreeNode]) extends Serializable {

  val root: TreeNode = nodes.last

  def predict(vector: Vector): Float = {
    var node = root
    while (!node.isInstanceOf[Leaf]) {
      val n = node.asInstanceOf[Node]
      val found = vector(n.feature)
      if (found <= n.value) {
        node = nodes(n.left)
      } else {
        node = nodes(n.right)
      }
    }
    node.asInstanceOf[Leaf].value
  }

}

class ClassificationTreeModel(override val trees: Seq[Tree]) extends TreeModel {

  def predict(vector: Vector): Float = {
    var sum = 0f
    for (tree <- trees) {
      sum += tree.predict(vector)
    }
    sigmoid(sum)
  }

  def sigmoid(x: Float): Float = {
    1f / (1f + math.exp(-x).toFloat)
  }

}

class RegressionTreeModel(override val trees: Seq[Tree]) extends TreeModel {

  def predict(vector: Vector): Float = {
    var sum = 0f
    for (tree <- trees) {
      sum += tree.predict(vector)
    }
    sum
  }

}

trait ModelFunctions {

  def createModel(trees: Seq[Tree]): TreeModel

  def initGrad(implicit random: Random): Float

  def initHess(implicit random: Random): Float

  def computeGrad(y: Float, yh: Float): Float

  def computeHess(y: Float, yh: Float): Float

  def computeLoss(y: Float, yh: Float): Float

}

private[gbt] object Regression extends ModelFunctions {

  override def createModel(trees: Seq[Tree]): TreeModel = {
    new RegressionTreeModel(trees)
  }

  override def initGrad(implicit random: Random): Float = random.nextFloat()

  override def initHess(implicit random: Random): Float = 2f

  override def computeGrad(y: Float, yh: Float): Float = 2 * (yh - y)

  override def computeHess(y: Float, yh: Float): Float = 2

  override def computeLoss(y: Float, yh: Float): Float = {
    val d = y - yh
    d * d
  }

}

private[gbt] object Classification extends ModelFunctions {

  override def createModel(trees: Seq[Tree]): TreeModel = {
    new ClassificationTreeModel(trees)
  }

  override def initGrad(implicit random: Random): Float = random.nextFloat()

  override def initHess(implicit random: Random): Float = random.nextFloat()

  override def computeGrad(y: Float, yh: Float): Float = yh - y

  override def computeHess(y: Float, yh: Float): Float = yh * (1f - yh)

  override def computeLoss(y: Float, yh: Float): Float = {
    -(y * math.log(yh) + (1 - y) * math.log(1 - yh)).toFloat
  }

}