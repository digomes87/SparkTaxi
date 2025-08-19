"""
NYC Taxi Data Analysis Project Configuration with Spark
"""

import os
from pyspark.sql import SparkSession

class SparkConfig:
    """Spark Configuration Settings"""
    
    APP_NAME = "NYCTaxiAnalysis"
    MASTER = "local[*]"  # Use all available cores
    
    # Memory configurations
    DRIVER_MEMORY = "4g"
    EXECUTOR_MEMORY = "4g"
    
    @staticmethod
    def create_spark_session():
        """Creates and returns a configured Spark session"""
        return SparkSession.builder \
            .appName(SparkConfig.APP_NAME) \
            .master(SparkConfig.MASTER) \
            .config("spark.driver.memory", SparkConfig.DRIVER_MEMORY) \
            .config("spark.executor.memory", SparkConfig.EXECUTOR_MEMORY) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()

class DataConfig:
    """Data Configuration Settings"""
    
    # Directories
    DATA_DIR = "data"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
    RESULTS_DIR = "results"
    
    # Data files
    TAXI_DATA_FILE = "nyc_taxi_data.csv"
    
    # Dataset URLs (Kaggle)
    KAGGLE_DATASET_URL = "https://www.kaggle.com/c/nyc-taxi-trip-duration/data"
    
    @staticmethod
    def create_directories():
        """Creates necessary directories if they don't exist"""
        directories = [
            DataConfig.DATA_DIR,
            DataConfig.RAW_DATA_DIR,
            DataConfig.PROCESSED_DATA_DIR,
            DataConfig.RESULTS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)