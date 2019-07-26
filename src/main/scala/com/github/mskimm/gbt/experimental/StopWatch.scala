package com.github.mskimm.gbt.experimental

class StopWatch {

  val s: Long = System.nanoTime()

  override def toString: String = {
    f"${(System.nanoTime() - s) / 1000000000.0}%.2f secs"
  }

}
