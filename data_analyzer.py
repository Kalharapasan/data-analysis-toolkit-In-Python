import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from typing import Dict, List, Tuple, Any, Optional

class DataAnalyzer:
    """Class for performing various data analysis operations"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
    
    def basic_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistics for numerical columns"""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        stats_dict = {}
        for col in numerical_cols:
            stats_dict[col] = {
                'mean': data[col].mean(),
                'median': data[col].median(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'skewness': stats.skew(data[col].dropna()),
                'kurtosis': stats.kurtosis(data[col].dropna()),
                'missing_count': data[col].isnull().sum(),
                'unique_count': data[col].nunique()
            }
        
        return stats_dict
    
    def correlation_analysis(self, data: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix"""
        numerical_data = data.select_dtypes(include=[np.number])
        return numerical_data.corr(method=method)
    
    def outlier_detection(self, data: pd.DataFrame, column: str, method: str = 'iqr') -> Dict[str, Any]:
        """Detect outliers using IQR or Z-score method"""
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        col_data = data[column].dropna()
        
        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(col_data))
            threshold = 3
            outliers = col_data[z_scores > threshold]
            lower_bound = col_data.mean() - threshold * col_data.std()
            upper_bound = col_data.mean() + threshold * col_data.std()
        
        return {
            'outliers_count': len(outliers),
            'outliers_percentage': (len(outliers) / len(col_data)) * 100,
            'outliers_values': outliers.tolist(),
            'bounds': {'lower': lower_bound, 'upper': upper_bound}
        }
    
    def group_analysis(self, data: pd.DataFrame, group_by: str, agg_column: str, 
                      agg_functions: List[str] = ['mean', 'sum', 'count']) -> pd.DataFrame:
        """Perform group by analysis"""
        if group_by not in data.columns or agg_column not in data.columns:
            raise ValueError("Specified columns not found in data")
        
        agg_dict = {agg_column: agg_functions}
        return data.groupby(group_by).agg(agg_dict).round(2)
    
    def time_series_analysis(self, data: pd.DataFrame, date_column: str, 
                           value_column: str) -> Dict[str, Any]:
        """Basic time series analysis"""
        if date_column not in data.columns or value_column not in data.columns:
            raise ValueError("Specified columns not found in data")
        
        # Convert to datetime if not already
        data[date_column] = pd.to_datetime(data[date_column])
        
        # Sort by date
        data_sorted = data.sort_values(date_column)
        
        # Calculate trends
        values = data_sorted[value_column].values
        dates_numeric = pd.to_numeric(data_sorted[date_column])
        
        # Linear trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(dates_numeric, values)
        
        # Moving averages
        data_sorted['MA_7'] = data_sorted[value_column].rolling(window=7).mean()
        data_sorted['MA_30'] = data_sorted[value_column].rolling(window=30).mean()
        
        return {
            'trend_slope': slope,
            'trend_r_squared': r_value**2,
            'trend_p_value': p_value,
            'data_with_ma': data_sorted,
            'mean_value': values.mean(),
            'volatility': values.std()
        }
    
    def prepare_data_for_ml(self, data: pd.DataFrame, target_column: str, 
                           test_size: float = 0.2) -> Tuple[Any, Any, Any, Any]:
        """Prepare data for machine learning"""
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        X_encoded = X.copy()
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X_encoded[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Handle missing values
        X_encoded = X_encoded.fillna(X_encoded.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def regression_analysis(self, X_train: Any, X_test: Any, y_train: Any, y_test: Any,
                          model_type: str = 'linear') -> Dict[str, Any]:
        """Perform regression analysis"""
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unsupported model type")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = model.score(X_train, y_train)
        test_r2 = model.score(X_test, y_test)
        
        self.models[f'{model_type}_regression'] = model
        
        return {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': y_pred_test
        }
    
    def classification_analysis(self, X_train: Any, X_test: Any, y_train: Any, y_test: Any,
                              model_type: str = 'logistic') -> Dict[str, Any]:
        """Perform classification analysis"""
        if model_type == 'logistic':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError("Unsupported model type")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        self.models[f'{model_type}_classification'] = model
        
        return {
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'predictions': y_pred_test,
            'classification_report': classification_report(y_test, y_pred_test)
        }
    
    def feature_importance(self, model_name: str) -> Optional[pd.DataFrame]:
        """Get feature importance from tree-based models"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Assuming we have access to feature names (would need to be passed)
            return pd.DataFrame({
                'feature': range(len(model.feature_importances_)),
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        return None