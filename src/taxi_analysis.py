"""
NYC Taxi Data Analysis with Apache Spark
Main analysis script for temporal patterns, distance-duration relationships, and popular routes
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SparkConfig, DataConfig


class NYCTaxiAnalyzer:
    """Main class for NYC Taxi data analysis"""
    
    def __init__(self):
        """Initialize the analyzer with Spark session"""
        self.spark = SparkConfig.create_spark_session()
        DataConfig.create_directories()
        self.df = None
        
    def load_data(self, file_path=None):
        """
        Load taxi data from CSV file
        
        Args:
            file_path (str): Path to the CSV file. If None, uses default path.
        """
        if file_path is None:
            file_path = os.path.join(DataConfig.RAW_DATA_DIR, DataConfig.TAXI_DATA_FILE)
        
        try:
            self.df = self.spark.read.csv(file_path, header=True, inferSchema=True)
            print(f"Data loaded successfully from {file_path}")
            print(f"Number of records: {self.df.count()}")
            self.df.printSchema()
        except Exception as e:
            print(f"Error loading data: {e}")
            # Create sample data for demonstration if file doesn't exist
            self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration purposes"""
        print("Creating sample data for demonstration...")
        
        # Sample schema based on NYC Taxi dataset
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
        
        # Create sample data with proper timestamp conversion
        from datetime import datetime
        sample_data = [
            ("id001", 1, datetime(2016, 1, 1, 10, 30, 0), datetime(2016, 1, 1, 10, 45, 0), 1, -73.98, 40.75, -73.97, 40.76, "N", 900, 2.5, 12.5),
            ("id002", 2, datetime(2016, 1, 1, 14, 20, 0), datetime(2016, 1, 1, 14, 35, 0), 2, -73.99, 40.74, -73.96, 40.77, "N", 900, 3.2, 15.0),
            ("id003", 1, datetime(2016, 1, 1, 18, 45, 0), datetime(2016, 1, 1, 19, 10, 0), 1, -73.97, 40.76, -73.95, 40.78, "N", 1500, 4.1, 18.5),
            ("id004", 2, datetime(2016, 1, 2, 8, 15, 0), datetime(2016, 1, 2, 8, 30, 0), 3, -73.96, 40.77, -73.98, 40.75, "N", 900, 2.8, 13.0),
            ("id005", 1, datetime(2016, 1, 2, 22, 30, 0), datetime(2016, 1, 2, 22, 50, 0), 1, -73.95, 40.78, -73.99, 40.74, "N", 1200, 3.5, 16.0),
            ("id006", 1, datetime(2016, 1, 3, 7, 0, 0), datetime(2016, 1, 3, 7, 20, 0), 2, -73.94, 40.79, -73.97, 40.76, "N", 1200, 3.8, 17.0),
            ("id007", 2, datetime(2016, 1, 3, 12, 15, 0), datetime(2016, 1, 3, 12, 35, 0), 1, -73.96, 40.77, -73.93, 40.80, "N", 1200, 4.2, 19.5),
            ("id008", 1, datetime(2016, 1, 3, 20, 45, 0), datetime(2016, 1, 3, 21, 5, 0), 4, -73.98, 40.75, -73.95, 40.78, "N", 1200, 3.1, 14.5),
            ("id009", 2, datetime(2016, 1, 4, 9, 30, 0), datetime(2016, 1, 4, 9, 50, 0), 2, -73.97, 40.76, -73.99, 40.74, "N", 1200, 2.9, 13.5),
            ("id010", 1, datetime(2016, 1, 4, 16, 0, 0), datetime(2016, 1, 4, 16, 25, 0), 1, -73.95, 40.78, -73.96, 40.77, "N", 1500, 1.8, 11.0)
        ]
        
        self.df = self.spark.createDataFrame(sample_data, schema)
        print("Sample data created successfully")
        self.df.show()
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in taxi trips"""
        print("\n=== TEMPORAL PATTERNS ANALYSIS ===")
        
        # Add time-based columns
        df_with_time = self.df.withColumn("hour", hour("pickup_datetime")) \
                             .withColumn("day_of_week", dayofweek("pickup_datetime")) \
                             .withColumn("month", month("pickup_datetime"))
        
        # Hourly trip distribution
        print("\n1. Trips by Hour of Day:")
        hourly_trips = df_with_time.groupBy("hour") \
                                  .count() \
                                  .orderBy("hour")
        hourly_trips.show()
        
        # Daily trip distribution
        print("\n2. Trips by Day of Week (1=Sunday, 7=Saturday):")
        daily_trips = df_with_time.groupBy("day_of_week") \
                                 .count() \
                                 .orderBy("day_of_week")
        daily_trips.show()
        
        # Save results
        self._save_results(hourly_trips, "hourly_trips")
        self._save_results(daily_trips, "daily_trips")
        
        return hourly_trips, daily_trips
    
    def analyze_distance_duration_relationship(self):
        """Analyze relationship between trip distance, duration, and fare"""
        print("\n=== DISTANCE-DURATION-FARE ANALYSIS ===")
        
        # Filter out invalid data
        clean_df = self.df.filter(
            (col("trip_distance") > 0) & 
            (col("trip_duration") > 0) & 
            (col("fare_amount") > 0)
        )
        
        # Basic statistics
        print("\n1. Basic Statistics:")
        clean_df.select("trip_distance", "trip_duration", "fare_amount").describe().show()
        
        # Calculate speed (miles per hour)
        speed_df = clean_df.withColumn(
            "speed_mph", 
            col("trip_distance") / (col("trip_duration") / 3600)
        )
        
        print("\n2. Speed Statistics (mph):")
        speed_df.select("speed_mph").describe().show()
        
        # Correlation analysis
        print("\n3. Distance vs Duration Analysis:")
        distance_duration = clean_df.select("trip_distance", "trip_duration", "fare_amount")
        
        # Convert to Pandas for correlation analysis
        pandas_df = distance_duration.toPandas()
        correlation_matrix = pandas_df.corr()
        print("Correlation Matrix:")
        print(correlation_matrix)
        
        # Save results
        self._save_results(speed_df.select("trip_distance", "trip_duration", "fare_amount", "speed_mph"), 
                          "distance_duration_analysis")
        
        return speed_df, correlation_matrix
    
    def analyze_popular_routes(self):
        """Analyze popular pickup and dropoff locations"""
        print("\n=== POPULAR ROUTES ANALYSIS ===")
        
        # Create location zones based on coordinates (simplified)
        df_with_zones = self.df.withColumn(
            "pickup_zone", 
            concat(
                round(col("pickup_latitude"), 2).cast("string"),
                lit(","),
                round(col("pickup_longitude"), 2).cast("string")
            )
        ).withColumn(
            "dropoff_zone",
            concat(
                round(col("dropoff_latitude"), 2).cast("string"),
                lit(","),
                round(col("dropoff_longitude"), 2).cast("string")
            )
        )
        
        # Most popular pickup zones
        print("\n1. Top 10 Pickup Zones:")
        popular_pickups = df_with_zones.groupBy("pickup_zone") \
                                      .count() \
                                      .orderBy(desc("count")) \
                                      .limit(10)
        popular_pickups.show()
        
        # Most popular dropoff zones
        print("\n2. Top 10 Dropoff Zones:")
        popular_dropoffs = df_with_zones.groupBy("dropoff_zone") \
                                       .count() \
                                       .orderBy(desc("count")) \
                                       .limit(10)
        popular_dropoffs.show()
        
        # Most popular routes (pickup -> dropoff)
        print("\n3. Top 10 Routes:")
        popular_routes = df_with_zones.groupBy("pickup_zone", "dropoff_zone") \
                                     .count() \
                                     .orderBy(desc("count")) \
                                     .limit(10)
        popular_routes.show()
        
        # Save results
        self._save_results(popular_pickups, "popular_pickups")
        self._save_results(popular_dropoffs, "popular_dropoffs")
        self._save_results(popular_routes, "popular_routes")
        
        return popular_pickups, popular_dropoffs, popular_routes
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n=== SUMMARY REPORT ===")
        
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        # Basic dataset information
        total_trips = self.df.count()
        print(f"Total number of trips: {total_trips:,}")
        
        # Date range
        date_range = self.df.select(
            min("pickup_datetime").alias("start_date"),
            max("pickup_datetime").alias("end_date")
        ).collect()[0]
        
        print(f"Date range: {date_range['start_date']} to {date_range['end_date']}")
        
        # Average trip statistics
        avg_stats = self.df.select(
            avg("trip_duration").alias("avg_duration"),
            avg("trip_distance").alias("avg_distance"),
            avg("fare_amount").alias("avg_fare")
        ).collect()[0]
        
        print(f"Average trip duration: {avg_stats['avg_duration']:.2f} seconds")
        print(f"Average trip distance: {avg_stats['avg_distance']:.2f} miles")
        print(f"Average fare amount: ${avg_stats['avg_fare']:.2f}")
        
        # Passenger count distribution
        print("\nPassenger count distribution:")
        self.df.groupBy("passenger_count").count().orderBy("passenger_count").show()
    
    def _save_results(self, dataframe, filename):
        """Save analysis results to CSV files"""
        try:
            output_path = os.path.join(DataConfig.RESULTS_DIR, f"{filename}.csv")
            # Convert to Pandas and save as CSV
            pandas_df = dataframe.toPandas()
            pandas_df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting NYC Taxi Data Analysis...")
        
        # Load data
        self.load_data()
        
        if self.df is None:
            print("Failed to load data. Exiting.")
            return
        
        # Generate summary report
        self.generate_summary_report()
        
        # Run all analyses
        self.analyze_temporal_patterns()
        self.analyze_distance_duration_relationship()
        self.analyze_popular_routes()
        
        print("\n=== ANALYSIS COMPLETE ===")
        print(f"Results saved in: {DataConfig.RESULTS_DIR}")
    
    def stop(self):
        """Stop the Spark session"""
        if self.spark:
            self.spark.stop()
            print("Spark session stopped.")


def main():
    """Main function to run the analysis"""
    analyzer = NYCTaxiAnalyzer()
    
    try:
        analyzer.run_complete_analysis()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        analyzer.stop()


if __name__ == "__main__":
    main()