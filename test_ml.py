#!/usr/bin/env python3
"""
Simple test for ML functionality
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from trip_duration_predictor import TripDurationPredictor

def test_ml():
    """Test ML functionality with simplified approach"""
    predictor = TripDurationPredictor()
    
    try:
        # Load and prepare data
        predictor.load_and_prepare_data()
        
        # Train models
        predictor.train_models()
        
        print("✅ ML test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during ML test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        predictor.stop()

if __name__ == "__main__":
    test_ml()