package com.github.mskimm.gbt

import scala.util.Random

trait TreeModel extends Serializable {

  val trees: Seq[Tree]

  def predict(vector: Vector): Float

}

trait TreeNode extends Serializable {

  def traverse(vector: Vector, nodes: Seq[TreeNode]): TreeNode

}

case class Node(feature: Int, value: Float, left: Int, right: Int) extends TreeNode {

  override def traverse(vector: Vector, nodes: Seq[TreeNode]): TreeNode = {
    val found = vector(feature)
    if (found <= value) {
      nodes(left)
    } else {
      nodes(right)
    }
  }

}

case class Leaf(value: Float) extends TreeNode {

  override def traverse(vector: Vector, node: Seq[TreeNode]): TreeNode = this

}

class Tree(nodes: Seq[TreeNode], rootIndex: Int = -1) extends Serializable {

  val root: TreeNode = if (rootIndex >= 0) nodes(rootIndex) else nodes.last

  def predict(vector: Vector): Float = {
    var node = root
    while (!node.isInstanceOf[Leaf]) {
      node = node.traverse(vector, nodes)
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

object Regression extends ModelFunctions {

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

object Classification extends ModelFunctions {

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