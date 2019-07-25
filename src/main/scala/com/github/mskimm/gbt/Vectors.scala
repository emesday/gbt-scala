package com.github.mskimm.gbt

trait Vector {

  def apply(i: Int): Float

  def get(i: Int): Option[Float]

}

object EmptyVector extends Vector {

  override def get(i: Int): Option[Float] = None

  override def apply(i: Int): Float = throw new IllegalArgumentException

}

class DenseVector(values: Array[Float])
  extends Vector with Serializable {

  override def apply(i: Int): Float = values(i)

  override def get(i: Int): Option[Float] = Some(apply(i))
}

class SparseVector(indices: Array[Int], values: Array[Float] = null)
  extends Vector with Serializable {

  def apply(i: Int): Float = {
    java.util.Arrays.binarySearch(indices, i) match {
      case found if found < 0 => 0f
      case found if values != null => values(found)
      case _ => 1f
    }
  }

  override def get(i: Int): Option[Float] = {
    apply(i) match {
      case 0 => None
      case value => Some(value)
    }
  }

}

object Vectors {

  def sparse(indices: Array[Int]): Vector = {
    new SparseVector(indices)
  }

  def sparse(indices: Array[Int], values: Array[Float]): Vector = {
    new SparseVector(indices, values)
  }

  def dense(values: Array[Float]): Vector = {
    new DenseVector(values)
  }

  def empty: Vector = EmptyVector

}

case class LabeledPoint(label: Float, point: Vector)
