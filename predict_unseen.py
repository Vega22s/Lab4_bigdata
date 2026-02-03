from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.master("local[1]").appName("Predict_Unseen").getOrCreate()

df = spark.read.option("header", True).option("inferSchema", True).csv("C:\\Labbigdata\\vehicle_stolen_dataset.csv")

indexers = [
    StringIndexer(inputCol="brand", outputCol="brand_index", handleInvalid="keep"),
    StringIndexer(inputCol="color", outputCol="color_index", handleInvalid="keep"),
    StringIndexer(inputCol="time", outputCol="time_index", handleInvalid="keep"),
    StringIndexer(inputCol="stoled", outputCol="label", handleInvalid="keep")
]

pipeline = Pipeline(stages=indexers)
indexed_df = pipeline.fit(df).transform(df)

vectorAssembler = VectorAssembler(inputCols=["brand_index", "color_index", "time_index"], outputCol="features")
final_df = vectorAssembler.transform(indexed_df)

train_data, test_data = final_df.randomSplit([0.70, 0.30], seed=43)

nb = NaiveBayes(featuresCol="features", labelCol="label", modelType="multinomial")
model = nb.fit(train_data)

predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

unseen_data = spark.createDataFrame([
    ("BMW", "black", "night"),
    ("NISSAN", "red", "day"),
    ("Audi", "blue", "night"),
    ("VEGA", "black", "day")
], ["brand", "color", "time"])

unseen_indexers = [
    StringIndexer(inputCol="brand", outputCol="brand_index", handleInvalid="keep"),
    StringIndexer(inputCol="color", outputCol="color_index", handleInvalid="keep"),
    StringIndexer(inputCol="time", outputCol="time_index", handleInvalid="keep")
]

unseen_pipeline = Pipeline(stages=unseen_indexers)
indexed_unseen = unseen_pipeline.fit(df).transform(unseen_data)
vindexed_unseen = vectorAssembler.transform(indexed_unseen)

unseen_predictions = model.transform(vindexed_unseen)
result = unseen_predictions.select("brand", "color", "time", "prediction")
result.show()
