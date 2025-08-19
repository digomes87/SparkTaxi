"""
NYC Taxi Trip Duration Prediction using Spark MLlib
Machine Learning model to predict trip duration based on various features
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import seaborn as sns

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SparkConfig, DataConfig


class TripDurationPredictor:
    """Machine Learning model for predicting taxi trip duration"""
    
    def __init__(self):
        """Initialize the predictor with Spark session"""
        self.spark = SparkConfig.create_spark_session()
        DataConfig.create_directories()
        self.df = None
        self.model = None
        self.feature_cols = []
        
    def load_and_prepare_data(self, file_path=None):
        """
        Load and prepare data for machine learning
        
        Args:
            file_path (str): Path to the CSV file. If None, uses default path.
        """
        if file_path is None:
            file_path = os.path.join(DataConfig.RAW_DATA_DIR, DataConfig.TAXI_DATA_FILE)
        
        try:
            self.df = self.spark.read.csv(file_path, header=True, inferSchema=True)
            print(f"Data loaded successfully from {file_path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create sample data for demonstration
            self._create_sample_data()
        
        # Prepare features
        self._prepare_features()
        
    def _create_sample_data(self):
        """Create sample data for demonstration purposes"""
        print("Creating sample data for ML demonstration...")
        
        # Sample schema
        schema = StructType([
            StructField("id", StringType(), True),
            StructField("vendor_id", IntegerType(), True),
            StructField("pickup_datetime", TimestampType(), True),
            StructField("dropoff_datetime", TimestampType(), True),
            StructField("passenger_count", IntegerType(), True),
            StructField("pickup_longitude", DoubleType(), True),
            StructField("pickup_latitude", DoubleType(), True),
            StructField("dropoff_longitude", DoubleType(), True),
            StructField("dropoff_latitude", DoubleType(), True),
            StructField("store_and_fwd_flag", StringType(), True),
            StructField("trip_duration", IntegerType(), True),
            StructField("trip_distance", DoubleType(), True),
            StructField("fare_amount", DoubleType(), True)
        ])
        
        # Create more diverse sample data for ML
        sample_data = []
        import random
        from datetime import datetime, timedelta
        
        base_date = datetime(2016, 1, 1)
        for i in range(1000):  # Create 1000 sample records
            pickup_time = base_date + timedelta(
                days=random.randint(0, 30),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            
            # Generate realistic coordinates for NYC
            pickup_lat = 40.7 + random.uniform(-0.1, 0.1)
            pickup_lon = -73.9 + random.uniform(-0.1, 0.1)
            dropoff_lat = 40.7 + random.uniform(-0.1, 0.1)
            dropoff_lon = -73.9 + random.uniform(-0.1, 0.1)
            
            # Calculate distance (simplified)
            import math
            distance = math.sqrt((pickup_lat - dropoff_lat)**2 + (pickup_lon - dropoff_lon)**2)
            distance = distance * 69  # Convert to approximate miles
            
            # Generate duration based on distance with some randomness
            base_duration = distance * 180 + random.randint(300, 1800)  # 3 minutes per mile + random
            
            dropoff_time = pickup_time + timedelta(seconds=base_duration)
            
            sample_data.append((
                f"id{i:04d}",
                random.randint(1, 2),
                pickup_time,
                dropoff_time,
                random.randint(1, 4),
                pickup_lon,
                pickup_lat,
                dropoff_lon,
                dropoff_lat,
                random.choice(["N", "Y"]),
                int(base_duration),
                __builtins__['round'](distance, 2),
                __builtins__['round'](distance * 2.5 + random.uniform(2, 10), 2)  # Base fare calculation
            ))
        
        self.df = self.spark.createDataFrame(sample_data, schema)
        print(f"Sample data created with {self.df.count()} records")
        
    def _prepare_features(self):
        """Prepare features for machine learning"""
        print("Preparing features for machine learning...")
        
        # Add time-based features
        self.df = self.df.withColumn("pickup_hour", hour("pickup_datetime")) \
                        .withColumn("pickup_day_of_week", dayofweek("pickup_datetime")) \
                        .withColumn("pickup_month", month("pickup_datetime"))
        
        # Calculate distance using Haversine formula (simplified)
        self.df = self.df.withColumn(
            "distance_calculated",
            sqrt(
                pow(col("pickup_latitude") - col("dropoff_latitude"), 2) +
                pow(col("pickup_longitude") - col("dropoff_longitude"), 2)
            ) * lit(69)  # Approximate miles conversion
        )
        
        # Convert store_and_fwd_flag to numeric
        self.df = self.df.withColumn(
            "store_and_fwd_numeric",
            when(col("store_and_fwd_flag") == "Y", 1).otherwise(0)
        )
        
        # Filter out invalid data
        self.df = self.df.filter(
            (col("trip_duration") > 60) &  # At least 1 minute
            (col("trip_duration") < 7200) &  # Less than 2 hours
            (col("passenger_count") > 0) &
            (col("passenger_count") <= 6) &
            (col("pickup_latitude").between(40.5, 41.0)) &
            (col("pickup_longitude").between(-74.5, -73.5)) &
            (col("dropoff_latitude").between(40.5, 41.0)) &
            (col("dropoff_longitude").between(-74.5, -73.5))
        )
        
        # Define feature columns
        self.feature_cols = [
            "vendor_id",
            "passenger_count",
            "pickup_hour",
            "pickup_day_of_week",
            "pickup_month",
            "pickup_latitude",
            "pickup_longitude",
            "dropoff_latitude",
            "dropoff_longitude",
            "distance_calculated",
            "store_and_fwd_numeric"
        ]
        
        print(f"Features prepared. Dataset size: {self.df.count()} records")
        print(f"Feature columns: {self.feature_cols}")
        
    def train_models(self):
        """Train multiple regression models and compare performance"""
        print("\n=== TRAINING MACHINE LEARNING MODELS ===")
        
        # Prepare feature vector
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features"
        )
        
        # Scale features
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        # Split data
        train_data, test_data = self.df.randomSplit([0.8, 0.2], seed=42)
        print(f"Training set size: {train_data.count()}")
        print(f"Test set size: {test_data.count()}")
        
        # Define models to train
        models = {
            "Linear Regression": LinearRegression(
                featuresCol="scaled_features",
                labelCol="trip_duration",
                predictionCol="prediction"
            ),
            "Random Forest": RandomForestRegressor(
                featuresCol="scaled_features",
                labelCol="trip_duration",
                predictionCol="prediction",
                numTrees=10
            ),
            "Gradient Boosted Trees": GBTRegressor(
                featuresCol="scaled_features",
                labelCol="trip_duration",
                predictionCol="prediction",
                maxIter=10
            )
        }
        
        # Train and evaluate each model
        results = {}
        evaluator = RegressionEvaluator(
            labelCol="trip_duration",
            predictionCol="prediction",
            metricName="rmse"
        )
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            
            # Create pipeline
            pipeline = Pipeline(stages=[assembler, scaler, model])
            
            # Train model
            fitted_pipeline = pipeline.fit(train_data)
            
            # Make predictions
            predictions = fitted_pipeline.transform(test_data)
            
            # Evaluate model
            rmse = evaluator.evaluate(predictions)
            mae_evaluator = RegressionEvaluator(
                labelCol="trip_duration",
                predictionCol="prediction",
                metricName="mae"
            )
            mae = mae_evaluator.evaluate(predictions)
            
            r2_evaluator = RegressionEvaluator(
                labelCol="trip_duration",
                predictionCol="prediction",
                metricName="r2"
            )
            r2 = r2_evaluator.evaluate(predictions)
            
            results[model_name] = {
                "model": fitted_pipeline,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "predictions": predictions
            }
            
            print(f"{model_name} Results:")
            print(f"  RMSE: {rmse:.2f} seconds")
            print(f"  MAE: {mae:.2f} seconds")
            print(f"  R²: {r2:.4f}")
        
        # Select best model based on RMSE
        best_model_name = __builtins__['min'](results.keys(), key=lambda k: results[k]["rmse"])
        self.model = results[best_model_name]["model"]
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best RMSE: {results[best_model_name]['rmse']:.2f} seconds")
        
        # Save model performance comparison
        self._save_model_comparison(results)
        
        return results
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model"""
        print("\n=== HYPERPARAMETER TUNING ===")
        
        # Prepare data
        assembler = VectorAssembler(
            inputCols=self.feature_cols,
            outputCol="features"
        )
        
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True
        )
        
        # Use Random Forest for tuning
        rf = RandomForestRegressor(
            featuresCol="scaled_features",
            labelCol="trip_duration",
            predictionCol="prediction"
        )
        
        # Create pipeline
        pipeline = Pipeline(stages=[assembler, scaler, rf])
        
        # Parameter grid
        param_grid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [5, 10, 20]) \
            .addGrid(rf.maxDepth, [5, 10, 15]) \
            .build()
        
        # Cross validator
        evaluator = RegressionEvaluator(
            labelCol="trip_duration",
            predictionCol="prediction",
            metricName="rmse"
        )
        
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3
        )
        
        # Split data
        train_data, test_data = self.df.randomSplit([0.8, 0.2], seed=42)
        
        # Fit cross validator
        print("Running cross-validation...")
        cv_model = cv.fit(train_data)
        
        # Get best model
        best_model = cv_model.bestModel
        
        # Evaluate on test data
        predictions = best_model.transform(test_data)
        rmse = evaluator.evaluate(predictions)
        
        print(f"Best model RMSE after tuning: {rmse:.2f} seconds")
        
        # Get best parameters
        best_rf_model = best_model.stages[-1]
        print(f"Best parameters:")
        print(f"  Number of trees: {best_rf_model.getNumTrees()}")
        print(f"  Max depth: {best_rf_model.getMaxDepth()}")
        
        self.model = best_model
        return best_model
    
    def feature_importance_analysis(self):
        """Analyze feature importance from the trained model"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        if self.model is None:
            print("No model trained yet. Please train a model first.")
            return
        
        # Get the Random Forest model from pipeline
        rf_model = None
        for stage in self.model.stages:
            if hasattr(stage, 'featureImportances'):
                rf_model = stage
                break
        
        if rf_model is None:
            print("Feature importance not available for this model type.")
            return
        
        # Get feature importances
        importances = rf_model.featureImportances.toArray()
        
        # Create feature importance DataFrame
        feature_importance_data = list(zip(self.feature_cols, importances))
        feature_importance_df = self.spark.createDataFrame(
            feature_importance_data,
            ["feature", "importance"]
        ).orderBy(desc("importance"))
        
        print("Feature Importance Ranking:")
        feature_importance_df.show()
        
        # Save feature importance
        self._save_results(feature_importance_df, "feature_importance")
        
        return feature_importance_df
    
    def make_predictions(self, new_data=None):
        """Make predictions on new data"""
        print("\n=== MAKING PREDICTIONS ===")
        
        if self.model is None:
            print("No model trained yet. Please train a model first.")
            return
        
        if new_data is None:
            # Use a sample from existing data
            new_data = self.df.limit(10)
        
        # Make predictions
        predictions = self.model.transform(new_data)
        
        # Show predictions vs actual
        result = predictions.select(
            "trip_duration",
            "prediction",
            (col("prediction") - col("trip_duration")).alias("error")
        )
        
        print("Sample Predictions vs Actual:")
        result.show()
        
        # Calculate prediction accuracy
        avg_error = result.select(avg(abs(col("error"))).alias("avg_absolute_error")).collect()[0][0]
        print(f"Average absolute error: {avg_error:.2f} seconds")
        
        return predictions
    
    def _save_model_comparison(self, results):
        """Save model comparison results"""
        comparison_data = []
        for model_name, metrics in results.items():
            comparison_data.append((
                model_name,
                metrics["rmse"],
                metrics["mae"],
                metrics["r2"]
            ))
        
        comparison_df = self.spark.createDataFrame(
            comparison_data,
            ["model", "rmse", "mae", "r2"]
        )
        
        self._save_results(comparison_df, "model_comparison")
    
    def _save_results(self, dataframe, filename):
        """Save analysis results to CSV files"""
        try:
            output_path = os.path.join(DataConfig.RESULTS_DIR, f"{filename}.csv")
            pandas_df = dataframe.toPandas()
            pandas_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def run_complete_ml_pipeline(self):
        """Run the complete machine learning pipeline"""
        print("Starting NYC Taxi Trip Duration Prediction Pipeline...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        if self.df is None:
            print("Failed to load data. Exiting.")
            return
        
        # Train models
        results = self.train_models()
        
        # Hyperparameter tuning
        self.hyperparameter_tuning()
        
        # Feature importance analysis
        self.feature_importance_analysis()
        
        # Make sample predictions
        self.make_predictions()
        
        print("\n=== MACHINE LEARNING PIPELINE COMPLETE ===")
        print(f"Results saved in: {DataConfig.RESULTS_DIR}")
    
    def stop(self):
        """Stop the Spark session"""
        if self.spark:
            self.spark.stop()
            print("Spark session stopped.")


def main():
    """Main function to run the ML pipeline"""
    predictor = TripDurationPredictor()
    
    try:
        predictor.run_complete_ml_pipeline()
    except KeyboardInterrupt:
        print("\nML pipeline interrupted by user.")
    except Exception as e:
        print(f"Error during ML pipeline: {e}")
    finally:
        predictor.stop()


if __name__ == "__main__":
    main()