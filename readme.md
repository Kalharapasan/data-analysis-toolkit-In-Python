# Advanced Data Analysis Tool

![image](https://github.com/user-attachments/assets/8312c3a1-673d-4c7f-ba31-68ed62061d5d)


A comprehensive Python application for data analysis with a user-friendly GUI interface. This tool provides extensive data loading, analysis, visualization, and machine learning capabilities.

## Features

### Data Loading
- Load CSV files
- Load Excel files (.xlsx, .xls)
- Create sample datasets (sales, customer, stock data)
- Data preview and information display

### Data Analysis
- Basic statistical analysis (mean, median, std dev, skewness, kurtosis)
- Correlation analysis
- Outlier detection (IQR and Z-score methods)
- Group-by analysis with aggregation
- Time series analysis

### Data Visualization
- Histograms
- Box plots
- Scatter plots
- Line plots
- Bar charts
- Pie charts
- Correlation heatmaps
- Outlier visualization

### Machine Learning
- Linear Regression
- Random Forest Regression
- Logistic Regression
- Random Forest Classification
- Automatic data preprocessing
- Model evaluation metrics

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the following files in the same directory:
   - `main_app.py` (main application)
   - `data_loader.py` (data loading functionality)
   - `data_analyzer.py` (analysis functionality)
   - `data_visualizer.py` (visualization functionality)

## Usage

### Running the Application
```bash
python main_app.py
```

### Application Structure

The application consists of four main tabs:

#### 1. Data Loading Tab
- **Load CSV**: Import CSV files
- **Load Excel**: Import Excel files
- **Create Sample Data**: Generate sample datasets for testing
- **Data Information**: View dataset statistics and column information
- **Data Preview**: Browse the first 100 rows of loaded data

#### 2. Data Analysis Tab
- **Basic Stats**: Calculate comprehensive statistics for numerical columns
- **Correlation**: Compute correlation matrices
- **Outliers**: Detect outliers using IQR or Z-score methods
- **Group Analysis**: Perform group-by operations with aggregation

#### 3. Visualization Tab
- **Plot Types**: Choose from various chart types
- **Column Selection**: Select X and Y columns for plotting
- **Interactive Plots**: Create matplotlib visualizations embedded in the GUI

#### 4. Machine Learning Tab
- **Model Selection**: Choose between regression and classification models
- **Target Selection**: Select target variable for prediction
- **Model Training**: Train models and view performance metrics

## Class Structure

### DataLoader Class
Handles all data loading operations:
```python
from data_loader import DataLoader

loader = DataLoader()
data = loader.load_csv('data.csv')
sample_data = loader.create_sample_data('sales')
```

### DataAnalyzer Class
Performs statistical analysis and machine learning:
```python
from data_analyzer import DataAnalyzer

analyzer = DataAnalyzer()
stats = analyzer.basic_statistics(data)
correlation = analyzer.correlation_analysis(data)
```

### DataVisualizer Class
Creates various types of plots:
```python
from data_visualizer import DataVisualizer

visualizer = DataVisualizer()
fig = visualizer.histogram(data, 'column_name')
fig = visualizer.scatter_plot(data, 'x_col', 'y_col')
```

## Sample Data Types

The application can generate three types of sample data:

1. **Sales Data**: Date, product, category, sales amount, quantity, region
2. **Customer Data**: Customer ID, age, gender, income, spending score, membership years
3. **Stock Data**: Date, price, volume, high, low prices

## Key Features

### Data Loading
- Support for multiple file formats
- Automatic data type detection
- Missing value identification
- Memory usage optimization

### Analysis Capabilities
- Comprehensive statistical summaries
- Correlation analysis with multiple methods
- Robust outlier detection
- Flexible grouping and aggregation

### Visualization Options
- Multiple chart types for different data types
- Customizable plot parameters
- Interactive matplotlib integration
- Correlation heatmaps with annotations

### Machine Learning
- Automatic data preprocessing
- Multiple algorithm options
- Cross-validation capabilities
- Performance metrics and evaluation

## Error Handling

The application includes comprehensive error handling:
- File loading errors
- Data type mismatches
- Missing column errors
- Model training failures

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms
- **scipy**: Scientific computing
- **openpyxl**: Excel file support
- **tkinter**: GUI framework (included with Python)

## Usage Examples

### Loading and Analyzing Data
```python
# Create application instance
app = DataAnalysisApp(root)

# Load data through GUI or programmatically
loader = DataLoader()
data = loader.load_csv('your_data.csv')

# Perform analysis
analyzer = DataAnalyzer()
stats = analyzer.basic_statistics(data)
outliers = analyzer.outlier_detection(data, 'column_name')
```

### Creating Visualizations
```python
# Create visualizer instance
visualizer = DataVisualizer()

# Generate different types of plots
hist_fig = visualizer.histogram(data, 'numeric_column')
scatter_fig = visualizer.scatter_plot(data, 'x_col', 'y_col')
corr_fig = visualizer.correlation_heatmap(correlation_matrix)
```

### Machine Learning Workflow
```python
# Prepare data for ML
X_train, X_test, y_train, y_test = analyzer.prepare_data_for_ml(data, 'target_column')

# Train regression model
regression_results = analyzer.regression_analysis(X_train, X_test, y_train, y_test, 'linear')

# Train classification model
classification_results = analyzer.classification_analysis(X_train, X_test, y_train, y_test, 'random_forest')
```

## Troubleshooting

### Common Issues

1. **Import Errors"# data-analysis-toolkit-In-Python" 
