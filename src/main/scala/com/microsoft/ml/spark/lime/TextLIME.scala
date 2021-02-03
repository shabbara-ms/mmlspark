package com.microsoft.ml.spark.lime

import breeze.stats.distributions.Rand
import com.microsoft.ml.spark.core.contracts.Wrappable
import com.microsoft.ml.spark.core.schema.{DatasetExtensions, ImageSchemaUtils}
import org.apache.spark.ml.{ComplexParamsReadable, ComplexParamsWritable, Estimator, Model, Transformer}
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.param.{DoubleArrayParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, explode_outer, lit, monotonically_increasing_id, size, udf}
import org.apache.spark.sql.types.{ArrayType, BinaryType, StringType, StructType}
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import com.microsoft.ml.spark.FluentAPI._

object TextLIME extends ComplexParamsReadable[TextLIME]

/** Distributed implementation of
  * Local Interpretable Model-Agnostic Explanations (LIME)
  *
  * https://arxiv.org/pdf/1602.04938v1.pdf
  */
class TextLIME(val uid: String) extends Model[TextLIME]
  with LIMEBase with Wrappable {
//class TextLIME(val uid: String) extends Transformer with LIMEBase
//  with Wrappable  {
// ***********************************************
// Below code is created by Sharath
  val kernel_width : Int = 25
  import breeze.numerics._
  import org.apache.spark.util.random
  import scala.util.Random
  def kernel : Double = sqrt(exp(-pow(d,2) / pow(kernel_width,2))) // d is the number of strings which are taken as input by classifier to give d,k array of prediction probabilities, k is classes
  val randval = scala.util.Random
  val split_expression : String = "\w+"

  def explain_instance(arr:  List.empty[Char]) : Unit = {
    val num_features: Int = 20 // maximum number of features present in explanation
    val num_samples: Int = 5000 //size of the neighborhood to learn the linear model
    val distance_metric: String = "cosine"

  }// ************************************************


  def this() = this(Identifiable.randomUID("TextLIME"))

  //TODO rename this to masked text column
  val columnMeans = new DoubleArrayParam(this, "columnMeans", "the means of each of the columns for perturbation")

  def getColumnMeans: Array[Double] = $(columnMeans)

  def setColumnMeans(v: Array[Double]): this.type = set(columnMeans, v)

  val columnSTDs = new DoubleArrayParam(this, "columnSTDs",
    "the standard deviations of each of the columns for perturbation")

  def getColumnSTDs: Array[Double] = $(columnSTDs)

  def setColumnSTDs(v: Array[Double]): this.type = set(columnSTDs, v)

  private def perturbedDenseVectors(v: DenseVector): Seq[DenseVector] = {
    Seq.fill(getNSamples) {
      val perturbed = BDV.rand(v.size, Rand.gaussian) * BDV(getColumnSTDs) + BDV(getColumnMeans)
      new DenseVector(perturbed.toArray)
    }
  }

  protected def maskText(tokens: Seq[String]): Seq[String] = {
    // ???
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val df = dataset.toDF
    val idCol = DatasetExtensions.findUnusedColumnName("id", df)
    val statesCol = DatasetExtensions.findUnusedColumnName("states", df)
    val inputCol2 = DatasetExtensions.findUnusedColumnName("inputCol2", df)

    // Data frame with new column containing superpixels (Array[Cluster]) for each row (image)
    val spt = new SuperpixelTransformer()
      .setCellSize(getCellSize)
      .setModifier(getModifier)
      .setInputCol(getInputCol)
      .setOutputCol(getSuperpixelCol)

    val spDF = spt.transform(df)

    // Indices of the columns containing each image and image's superpixels
    val inputType = df.schema(getInputCol).dataType
    val maskUDF = udf(maskText _, ArrayType(StringType))

    val mapped = spDF.withColumn(idCol, monotonically_increasing_id())
      .withColumnRenamed(getInputCol, inputCol2)
      .withColumn(statesCol, explode_outer(getSampleUDF(size(col(getSuperpixelCol).getField("clusters")))))
      .withColumn(getInputCol, maskUDF(col(inputCol2), col(spt.getOutputCol), col(statesCol)))
      .withColumn(statesCol, udf(
        { barr: Seq[Boolean] => new DenseVector(barr.map(b => if (b) 1.0 else 0.0).toArray) },
        VectorType)(col(statesCol)))
      .mlTransform(getModel)
      .drop(getInputCol)

    LIMEUtils.localAggregateBy(mapped, idCol, Seq(statesCol, getPredictionCol))
      .withColumn(statesCol, arrToMatUDF(col(statesCol)))
      .withColumn(getPredictionCol, arrToVectUDF(col(getPredictionCol)))
      .withColumn(getOutputCol, fitLassoUDF(col(statesCol), col(getPredictionCol), lit(getRegularization)))
      .drop(statesCol, getPredictionCol)
      .withColumnRenamed(inputCol2, getInputCol)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {
    schema.add(getSuperpixelCol, SuperpixelData.Schema).add(getOutputCol, VectorType)
  }

}

