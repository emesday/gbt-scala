package com.github.mskimm.gbt

trait Vector {

  def apply(i: Int): Float

}

class DenseVector(values: Array[Float])
  extends Vector with Serializable {

  override def apply(i: Int): Float = values(i)

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

}

case class LabeledPoint(label: Float, point: Vector)
