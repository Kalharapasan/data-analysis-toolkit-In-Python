import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import os

class DataLoader:
    """Class for loading and managing datasets"""
    
    def __init__(self):
        self.data = None
        self.file_path = None
        self.data_info = {}
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV file"""
        try:
            self.data = pd.read_csv(file_path, **kwargs)
            self.file_path = file_path
            self._update_data_info()
            return self.data
        except Exception as e:
            raise Exception(f"Error loading CSV: {str(e)}")
    
    def load_excel(self, file_path: str, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load Excel file"""
        try:
            self.data = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            self.file_path = file_path
            self._update_data_info()
            return self.data
        except Exception as e:
            raise Exception(f"Error loading Excel: {str(e)}")
    
    def load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load JSON file"""
        try:
            self.data = pd.read_json(file_path, **kwargs)
            self.file_path = file_path
            self._update_data_info()
            return self.data
        except Exception as e:
            raise Exception(f"Error loading JSON: {str(e)}")
    
    def create_sample_data(self, dataset_type: str = 'sales') -> pd.DataFrame:
        """Create sample datasets for testing"""
        np.random.seed(42)
        
        if dataset_type == 'sales':
            dates = pd.date_range('2023-01-01', periods=1000, freq='D')
            self.data = pd.DataFrame({
                'date': np.random.choice(dates, 500),
                'product': np.random.choice(['Product A', 'Product B', 'Product C', 'Product D'], 500),
                'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], 500),
                'sales': np.random.uniform(10, 1000, 500),
                'quantity': np.random.randint(1, 50, 500),
                'region': np.random.choice(['North', 'South', 'East', 'West'], 500)
            })
        elif dataset_type == 'customer':
            self.data = pd.DataFrame({
                'customer_id': range(1, 501),
                'age': np.random.randint(18, 80, 500),
                'gender': np.random.choice(['Male', 'Female'], 500),
                'income': np.random.uniform(20000, 150000, 500),
                'spending_score': np.random.randint(1, 100, 500),
                'membership_years': np.random.randint(1, 10, 500)
            })
        elif dataset_type == 'stock':
            dates = pd.date_range('2023-01-01', periods=365, freq='D')
            price = 100
            prices = []
            for _ in range(365):
                price += np.random.uniform(-5, 5)
                prices.append(max(price, 10))
            
            self.data = pd.DataFrame({
                'date': dates,
                'price': prices,
                'volume': np.random.randint(1000, 50000, 365),
                'high': [p + np.random.uniform(0, 10) for p in prices],
                'low': [p - np.random.uniform(0, 10) for p in prices]
            })
        
        self._update_data_info()
        return self.data
    
    def _update_data_info(self):
        """Update data information"""
        if self.data is not None:
            self.data_info = {
                'shape': self.data.shape,
                'columns': list(self.data.columns),
                'dtypes': self.data.dtypes.to_dict(),
                'memory_usage': self.data.memory_usage(deep=True).sum(),
                'missing_values': self.data.isnull().sum().to_dict()
            }
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get comprehensive data information"""
        return self.data_info
    
    def get_data_sample(self, n: int = 5) -> pd.DataFrame:
        """Get sample of data"""
        if self.data is not None:
            return self.data.head(n)
        return pd.DataFrame()
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get statistical summary of data"""
        if self.data is not None:
            return self.data.describe(include='all')
        return pd.DataFrame()