package com.github.mskimm.gbt

import scala.collection.mutable

// from spark
class BinaryLabelCounter(
  var weightedNumPositives: Float = 0,
  var weightedNumNegatives: Float = 0) extends Serializable {

  def +=(label: Float): BinaryLabelCounter = {
    if (label > 0.5f) weightedNumPositives += 1 else weightedNumNegatives += 1
    this
  }

  def +=(label: Float, weight: Float): BinaryLabelCounter = {
    if (label > 0.5f) weightedNumPositives += weight else weightedNumNegatives += weight
    this
  }

  def +=(other: BinaryLabelCounter): BinaryLabelCounter = {
    weightedNumPositives += other.weightedNumPositives
    weightedNumNegatives += other.weightedNumNegatives
    this
  }

  override def clone: BinaryLabelCounter = {
    new BinaryLabelCounter(weightedNumPositives, weightedNumNegatives)
  }

  override def toString: String = s"{numPos: $weightedNumPositives, numNeg: $weightedNumNegatives}"
}

// from spark
case class BinaryConfusionMatrix(
  count: BinaryLabelCounter,
  totalCount: BinaryLabelCounter) {

  def weightedTruePositives: Float = count.weightedNumPositives

  def weightedFalsePositives: Float = count.weightedNumNegatives

  def weightedFalseNegatives: Float =
    totalCount.weightedNumPositives - count.weightedNumPositives

  def weightedTrueNegatives: Float =
    totalCount.weightedNumNegatives - count.weightedNumNegatives

  def weightedPositives: Float = totalCount.weightedNumPositives

  def weightedNegatives: Float = totalCount.weightedNumNegatives
}

// from spark
trait BinaryClassificationMetricComputer extends Serializable {
  def apply(c: BinaryConfusionMatrix): Float
}

// from spark
object Precision extends BinaryClassificationMetricComputer {
  override def apply(c: BinaryConfusionMatrix): Float = {
    val totalPositives = c.weightedTruePositives + c.weightedFalsePositives
    if (totalPositives == 0f) {
      1f
    } else {
      c.weightedTruePositives / totalPositives
    }
  }
}

// from spark
object FalsePositiveRate extends BinaryClassificationMetricComputer {
  override def apply(c: BinaryConfusionMatrix): Float = {
    if (c.weightedNegatives == 0f) {
      0f
    } else {
      c.weightedFalsePositives / c.weightedNegatives
    }
  }
}

// from spark
object Recall extends BinaryClassificationMetricComputer {
  override def apply(c: BinaryConfusionMatrix): Float = {
    if (c.weightedPositives == 0f) {
      0f
    } else {
      c.weightedTruePositives / c.weightedPositives
    }
  }
}

// from spark
case class FMeasure(beta: Float) extends BinaryClassificationMetricComputer {
  private val beta2 = beta * beta

  override def apply(c: BinaryConfusionMatrix): Float = {
    val precision = Precision(c)
    val recall = Recall(c)
    if (precision + recall == 0f) {
      0f
    } else {
      (1f + beta2) * (precision * recall) / (beta2 * precision + recall)
    }
  }
}

// from spark
object AreaUnderCurve {
  def of(curve: Iterable[(Float, Float)]): Float = {
    curve.toIterator.sliding(2).withPartial(false).aggregate(0f)(
      seqop = (auc: Float, points: Seq[(Float, Float)]) => auc + trapezoid(points),
      combop = _ + _
    )
  }

  private def trapezoid(points: Seq[(Float, Float)]): Float = {
    require(points.length == 2)
    val x = points.head
    val y = points.last
    (y._1 - x._1) * (y._2 + x._2) / 2f
  }
}

class BinaryClassificationMetrics(
  val scores: Array[Float],
  val labels: Array[Float],
  val numBins: Int) {

  require(numBins >= 0, "numBins must be nonnegative")

  private lazy val (
    cumulativeCounts: Array[(Float, BinaryLabelCounter)],
    confusions: Array[(Float, BinaryConfusionMatrix)],
    totalCount0: BinaryLabelCounter) = {
    val counterMap = mutable.HashMap[Float, BinaryLabelCounter]()
    for ((score, label) <- scores.zip(labels)) {
      val counter = counterMap.getOrElseUpdate(score, new BinaryLabelCounter(0L, 0L))
      counter += label
    }
    val counts = counterMap.toArray.sortBy(-_._1)
    val binnedCounts = if (numBins == 0) {
      counts
    } else {
      throw new NotImplementedError("numBins > 0 is not implemented")
    }
    val cumulativeCounts = binnedCounts.map(_._1).zip(
      binnedCounts.map(_._2).scanLeft(new BinaryLabelCounter())((agg, c) => agg.clone() += c).drop(1))
    val totalCount = cumulativeCounts.last._2
    val confusions = cumulativeCounts.map { case (score, cumCount) =>
      (score, BinaryConfusionMatrix(cumCount, totalCount))
    }
    (cumulativeCounts, confusions, totalCount)
  }

  def this(scores: Array[Float], labels: Array[Float]) = this(scores, labels, 0)

  def thresholds(): Seq[Float] = cumulativeCounts.map(_._1)

  def areaUnderROC(): Float = AreaUnderCurve.of(roc())

  def roc(): Seq[(Float, Float)] = {
    val rocCurve = createCurve(FalsePositiveRate, Recall)
    Seq((0f, 0f)) ++ rocCurve ++ Seq((1f, 1f))
  }

  def areaUnderPR(): Float = AreaUnderCurve.of(pr())

  def pr(): Seq[(Float, Float)] = {
    val prCurve = createCurve(Recall, Precision)
    val (_, firstPrecision) = prCurve.head
    Seq((0f, firstPrecision)) ++ prCurve
  }

  private def createCurve(
    x: BinaryClassificationMetricComputer,
    y: BinaryClassificationMetricComputer): Array[(Float, Float)] = {
    confusions.map { case (_, c) =>
      (x(c), y(c))
    }
  }

  def fMeasureByThreshold(): Seq[(Float, Float)] = fMeasureByThreshold(1.0f)

  def fMeasureByThreshold(beta: Float): Seq[(Float, Float)] = createCurve(FMeasure(beta))

  def precisionByThreshold(): Seq[(Float, Float)] = createCurve(Precision)

  def recallByThreshold(): Seq[(Float, Float)] = createCurve(Recall)

  private def createCurve(y: BinaryClassificationMetricComputer): Array[(Float, Float)] = {
    confusions.map { case (s, c) =>
      (s, y(c))
    }
  }

  def totalCount: BinaryLabelCounter = totalCount0.clone()
}

object Metrics {

  def rmse(scores: Array[Float], labels: Array[Float]): Float = {
    var l = 0f
    for ((y, yh) <- labels.zip(scores)) {
      l += (y - yh) * (y - yh) / scores.length
    }
    math.sqrt(l).toFloat
  }

}