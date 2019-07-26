package com.github.mskimm.gbt

trait GBTModel extends Serializable {

  val trees: Seq[Tree]

  def predict(vector: Vector): Float

}

object GBTModel extends Enumeration {

  type ModelType = Value
  val Classification, Regression = Value

  def create(trees: Seq[Tree], tpe: GBTModel.Value): GBTModel = {
    tpe match {
      case Classification => new GBTClassificationModel(trees)
      case Regression => new GBTRegressionModel(trees)
    }
  }

}

class GBTClassificationModel(override val trees: Seq[Tree]) extends GBTModel {

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

class GBTRegressionModel(override val trees: Seq[Tree]) extends GBTModel {

  def predict(vector: Vector): Float = {
    var sum = 0f
    for (tree <- trees) {
      sum += tree.predict(vector)
    }
    sum
  }

}

trait TreeNode extends Serializable {

  def traverse(vector: Vector, nodes: Seq[TreeNode]): TreeNode

}

case class InternalNode(feature: Int, value: Float, left: Int, right: Int) extends TreeNode {

  override def traverse(vector: Vector, nodes: Seq[TreeNode]): TreeNode = {
    val found = vector(feature)
    if (found <= value) {
      nodes(left)
    } else {
      nodes(right)
    }
  }

}

case class LeafNode(value: Float) extends TreeNode {

  override def traverse(vector: Vector, node: Seq[TreeNode]): TreeNode = this

}

class Tree(nodes: Seq[TreeNode], rootIndex: Int = -1) extends Serializable {

  val root: TreeNode = if (rootIndex >= 0) nodes(rootIndex) else nodes.last

  def predict(vector: Vector): Float = {
    var node = root
    while (!node.isInstanceOf[LeafNode]) {
      node = node.traverse(vector, nodes)
    }
    node.asInstanceOf[LeafNode].value
  }

}

