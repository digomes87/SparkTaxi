#!/usr/bin/env python3
"""
Example usage of NYC Taxi Data Analysis with Spark
This script demonstrates how to use the analysis and prediction modules
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from taxi_analysis import NYCTaxiAnalyzer
from trip_duration_predictor import TripDurationPredictor


def run_basic_analysis():
    """Run basic taxi data analysis"""
    print("=" * 60)
    print("RUNNING BASIC TAXI DATA ANALYSIS")
    print("=" * 60)
    
    analyzer = NYCTaxiAnalyzer()
    
    try:
        # Run complete analysis
        analyzer.run_complete_analysis()
        
        print("\n✅ Basic analysis completed successfully!")
        print("📁 Check the 'results/' directory for output files")
        
    except Exception as e:
        print(f"❌ Error during basic analysis: {e}")
    finally:
        analyzer.stop()


def run_ml_prediction():
    """Run machine learning prediction pipeline"""
    print("\n" + "=" * 60)
    print("RUNNING MACHINE LEARNING PREDICTION PIPELINE")
    print("=" * 60)
    
    predictor = TripDurationPredictor()
    
    try:
        # Run complete ML pipeline
        predictor.run_complete_ml_pipeline()
        
        print("\n✅ ML prediction pipeline completed successfully!")
        print("📁 Check the 'results/' directory for model results")
        
    except Exception as e:
        print(f"❌ Error during ML pipeline: {e}")
    finally:
        predictor.stop()


def run_custom_analysis():
    """Example of custom analysis using the modules"""
    print("\n" + "=" * 60)
    print("RUNNING CUSTOM ANALYSIS EXAMPLE")
    print("=" * 60)
    
    analyzer = NYCTaxiAnalyzer()
    
    try:
        # Load data
        analyzer.load_data()
        
        # Run specific analyses
        print("\n🔍 Running temporal patterns analysis...")
        hourly_trips, daily_trips = analyzer.analyze_temporal_patterns()
        
        print("\n🔍 Running distance-duration analysis...")
        speed_df, correlation_matrix = analyzer.analyze_distance_duration_relationship()
        
        print("\n📊 Custom analysis completed!")
        
        # Show some results
        print("\nSample of hourly trips data:")
        hourly_trips.show(5)
        
    except Exception as e:
        print(f"❌ Error during custom analysis: {e}")
    finally:
        analyzer.stop()


def main():
    """Main function to run all examples"""
    print("🚕 NYC Taxi Data Analysis with Apache Spark")
    print("=" * 60)
    print("This example demonstrates the capabilities of the analysis framework")
    print()
    
    # Check if user wants to run specific analysis
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "basic":
            run_basic_analysis()
        elif mode == "ml":
            run_ml_prediction()
        elif mode == "custom":
            run_custom_analysis()
        else:
            print("❌ Invalid mode. Use: basic, ml, or custom")
            sys.exit(1)
    else:
        # Run all analyses
        print("🚀 Running all analyses (this may take a while)...")
        print("💡 Tip: Use 'python example_usage.py basic' to run only basic analysis")
        print()
        
        run_basic_analysis()
        run_ml_prediction()
        run_custom_analysis()
    
    print("\n" + "=" * 60)
    print("🎉 ALL ANALYSES COMPLETED!")
    print("=" * 60)
    print("📁 Results are saved in the 'results/' directory")
    print("📊 You can now explore the generated CSV files and reports")
    print("🔍 For more details, check the README.md file")


if __name__ == "__main__":
    main()