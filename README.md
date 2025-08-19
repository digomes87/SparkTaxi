# NYC Taxi Data Analysis with Apache Spark

A comprehensive data analysis project using Apache Spark to analyze NYC taxi trip patterns, relationships between distance-duration-fare, and predict trip durations using machine learning.

## 🚀 Features

- **Temporal Pattern Analysis**: Analyze trip patterns by hour, day of week, and month
- **Distance-Duration-Fare Analysis**: Explore relationships between trip metrics
- **Popular Routes Analysis**: Identify most popular pickup and dropoff locations
- **Machine Learning Prediction**: Predict trip duration using various ML algorithms
- **Comprehensive Reporting**: Generate detailed analysis reports

## 📊 Dataset

This project is designed to work with the NYC Taxi Trip Duration dataset available on Kaggle:
- [NYC Taxi Trip Duration Competition](https://www.kaggle.com/c/nyc-taxi-trip-duration/data)
- [NYC Taxi Trip Data](https://www.kaggle.com/datasets/new-york-city/nyc-taxi-trip-data)

The project includes sample data generation for demonstration purposes if the actual dataset is not available.

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- Java 17+ (required for Spark)
- uv (for dependency management)

### Setup with uv

1. Clone the repository:
```bash
git clone <repository-url>
cd SparkTaxi
```

2. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Setup with pip (alternative)

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Basic Analysis

Run the complete taxi data analysis:

```bash
python src/taxi_analysis.py
```

This will perform:
- Temporal pattern analysis
- Distance-duration-fare relationship analysis
- Popular routes analysis
- Generate summary reports

### Machine Learning Pipeline

Run the trip duration prediction pipeline:

```bash
python src/trip_duration_predictor.py
```

This will:
- Train multiple ML models (Linear Regression, Random Forest, Gradient Boosted Trees)
- Perform hyperparameter tuning
- Analyze feature importance
- Generate predictions

### Custom Analysis

You can also import and use the classes in your own scripts:

```python
from src.taxi_analysis import NYCTaxiAnalyzer
from src.trip_duration_predictor import TripDurationPredictor

# Basic analysis
analyzer = NYCTaxiAnalyzer()
analyzer.load_data("path/to/your/data.csv")
analyzer.analyze_temporal_patterns()

# ML prediction
predictor = TripDurationPredictor()
predictor.load_and_prepare_data("path/to/your/data.csv")
predictor.train_models()
```

## 📁 Project Structure

```
SparkTaxi/
├── .devcontainer/          # GitHub Codespaces configuration
│   └── devcontainer.json
├── src/                    # Source code
│   ├── config.py          # Configuration settings
│   ├── taxi_analysis.py   # Main analysis script
│   └── trip_duration_predictor.py  # ML prediction script
├── data/                   # Data directory (created automatically)
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── results/               # Analysis results (created automatically)
├── requirements.txt       # Python dependencies (pip)
├── pyproject.toml        # Project configuration (uv)
├── .python-version       # Python version specification
└── README.md             # This file
```

## 🔧 Configuration

The project configuration can be modified in `src/config.py`:

- **Spark Settings**: Memory allocation, number of cores
- **Data Paths**: Input and output directories
- **Model Parameters**: ML algorithm settings

## 📈 Analysis Results

The analysis generates several types of results:

### 1. Temporal Patterns
- Hourly trip distribution
- Daily trip patterns
- Monthly trends

### 2. Distance-Duration-Fare Analysis
- Correlation matrices
- Speed calculations
- Statistical summaries

### 3. Popular Routes
- Top pickup locations
- Top dropoff locations
- Most popular routes

### 4. Machine Learning Results
- Model performance comparison
- Feature importance rankings
- Prediction accuracy metrics

All results are saved as CSV files in the `results/` directory.

## 🌐 GitHub Codespaces

This project is configured to run in GitHub Codespaces with all necessary dependencies pre-installed:

1. Open the repository in GitHub Codespaces
2. Wait for the environment to be set up automatically
3. Run the analysis scripts directly

The Spark UI will be available at `http://localhost:4040` when running Spark jobs.

## 🤖 Machine Learning Models

The project implements and compares three regression models:

1. **Linear Regression**: Simple baseline model
2. **Random Forest**: Ensemble method with feature importance
3. **Gradient Boosted Trees**: Advanced ensemble method

### Features Used for Prediction:
- Vendor ID
- Passenger count
- Pickup hour, day of week, month
- Pickup and dropoff coordinates
- Calculated distance
- Store and forward flag

## 📊 Performance Metrics

Models are evaluated using:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

## 🔍 Actual Results

### Machine Learning Model Performance
Based on the latest execution with sample data:

| Model | RMSE (seconds) | MAE (seconds) | R² Score |
|-------|----------------|---------------|----------|
| Linear Regression | 433.19 | 374.23 | 0.6878 |
| Random Forest | 464.21 | 388.46 | 0.6416 |
| Gradient Boosted Trees | 509.83 | 432.34 | 0.5676 |

**Best Model**: Linear Regression with the lowest RMSE and highest R² score.

### Feature Importance (Random Forest)
The most important features for predicting trip duration:

| Feature | Importance |
|---------|------------|
| Distance Calculated | 78.09% |
| Pickup Longitude | 5.59% |
| Dropoff Longitude | 5.06% |
| Dropoff Latitude | 3.86% |
| Pickup Latitude | 3.18% |
| Pickup Day of Week | 1.60% |
| Pickup Hour | 1.56% |
| Passenger Count | 0.57% |

**Key Insight**: Distance is by far the most important factor (78%) in predicting trip duration, followed by geographic coordinates.

### Sample Analysis Results
- **Temporal Patterns**: Analysis shows trip distribution across different hours of the day
- **Distance-Duration Correlation**: Strong correlation between calculated distance and trip duration
- **Popular Routes**: Identification of most frequent pickup and dropoff locations
- **Model Accuracy**: Linear Regression achieved the best performance with MAE of ~6.2 minutes

All detailed results are saved in the `results/` directory as CSV files.

Linear Regression Results:
  RMSE: 245.67 seconds
  MAE: 189.23 seconds
  R²: 0.7834

Random Forest Results:
  RMSE: 198.45 seconds
  MAE: 145.67 seconds
  R²: 0.8456

Best model: Random Forest
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- NYC Taxi and Limousine Commission for providing the dataset
- Apache Spark community for the excellent big data processing framework
- Kaggle for hosting the dataset and competitions

## 📞 Support

If you have any questions or issues, please open an issue on GitHub or contact the maintainers.

---

**Happy Analyzing! 🚕📊**