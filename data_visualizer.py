import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DataVisualizer:
    """Class for creating various data visualizations"""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 6)):
        plt.style.use('default')  # Use default style as seaborn styles may not be available
        sns.set_palette("husl")
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def histogram(self, data: pd.DataFrame, column: str, bins: int = 30, 
                 title: Optional[str] = None) -> plt.Figure:
        """Create histogram"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.hist(data[column].dropna(), bins=bins, alpha=0.7, color=self.colors[0])
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.set_title(title or f'Distribution of {column}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def box_plot(self, data: pd.DataFrame, column: str, group_by: Optional[str] = None,
                title: Optional[str] = None) -> plt.Figure:
        """Create box plot"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if group_by:
            data.boxplot(column=column, by=group_by, ax=ax)
            ax.set_title(title or f'{column} by {group_by}')
        else:
            ax.boxplot(data[column].dropna())
            ax.set_xlabel(column)
            ax.set_title(title or f'Box Plot of {column}')
        
        plt.tight_layout()
        return fig
    
    def scatter_plot(self, data: pd.DataFrame, x_column: str, y_column: str,
                    color_by: Optional[str] = None, title: Optional[str] = None) -> plt.Figure:
        """Create scatter plot"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if color_by and color_by in data.columns:
            unique_values = data[color_by].unique()
            for i, value in enumerate(unique_values):
                mask = data[color_by] == value
                ax.scatter(data[mask][x_column], data[mask][y_column], 
                          label=str(value), alpha=0.7, color=self.colors[i % len(self.colors)])
            ax.legend()
        else:
            ax.scatter(data[x_column], data[y_column], alpha=0.7, color=self.colors[0])
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(title or f'{x_column} vs {y_column}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def line_plot(self, data: pd.DataFrame, x_column: str, y_column: str,
                 title: Optional[str] = None) -> plt.Figure:
        """Create line plot"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by x column for proper line plot
        data_sorted = data.sort_values(x_column)
        ax.plot(data_sorted[x_column], data_sorted[y_column], 
                color=self.colors[0], linewidth=2, marker='o', markersize=4)
        
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_title(title or f'{y_column} over {x_column}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def bar_plot(self, data: pd.DataFrame, x_column: str, y_column: str,
                title: Optional[str] = None, horizontal: bool = False) -> plt.Figure:
        """Create bar plot"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Group data if needed
        if data[x_column].dtype == 'object':
            grouped_data = data.groupby(x_column)[y_column].sum().sort_values(ascending=False)
        else:
            grouped_data = data.set_index(x_column)[y_column]
        
        if horizontal:
            ax.barh(range(len(grouped_data)), grouped_data.values, color=self.colors[0])
            ax.set_yticks(range(len(grouped_data)))
            ax.set_yticklabels(grouped_data.index)
            ax.set_xlabel(y_column)
        else:
            ax.bar(range(len(grouped_data)), grouped_data.values, color=self.colors[0])
            ax.set_xticks(range(len(grouped_data)))
            ax.set_xticklabels(grouped_data.index, rotation=45 if len(grouped_data) > 5 else 0)
            ax.set_ylabel(y_column)
        
        ax.set_title(title or f'{y_column} by {x_column}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def correlation_heatmap(self, correlation_matrix: pd.DataFrame,
                           title: Optional[str] = None) -> plt.Figure:
        """Create correlation heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', ax=ax)
        
        ax.set_title(title or 'Correlation Matrix')
        plt.tight_layout()
        return fig
    
    def pie_chart(self, data: pd.DataFrame, column: str, 
                 title: Optional[str] = None, top_n: int = 10) -> plt.Figure:
        """Create pie chart"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        value_counts = data[column].value_counts().head(top_n)
        
        ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%',
               colors=self.colors[:len(value_counts)])
        ax.set_title(title or f'Distribution of {column}')
        
        plt.tight_layout()
        return fig
    
    def time_series_plot(self, data: pd.DataFrame, date_column: str, 
                        value_columns: List[str], title: Optional[str] = None) -> plt.Figure:
        """Create time series plot"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Ensure date column is datetime
        data[date_column] = pd.to_datetime(data[date_column])
        data_sorted = data.sort_values(date_column)
        
        for i, col in enumerate(value_columns):
            ax.plot(data_sorted[date_column], data_sorted[col], 
                   label=col, color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax.set_xlabel(date_column)
        ax.set_ylabel('Value')
        ax.set_title(title or 'Time Series Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        fig.autofmt_xdate()
        plt.tight_layout()
        return fig
    
    def subplot_grid(self, data: pd.DataFrame, columns: List[str], 
                    plot_type: str = 'hist', figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
        """Create subplot grid for multiple columns"""
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3  # 3 columns per row
        
        fig_size = figsize or (15, 5 * n_rows)
        fig, axes = plt.subplots(n_rows, 3, figsize=fig_size)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, col in enumerate(columns):
            ax = axes[i] if n_cols > 1 else axes
            
            if plot_type == 'hist':
                ax.hist(data[col].dropna(), bins=20, alpha=0.7, color=self.colors[i % len(self.colors)])
                ax.set_title(f'Distribution of {col}')
            elif plot_type == 'box':
                ax.boxplot(data[col].dropna())
                ax.set_title(f'Box Plot of {col}')
        
        # Hide empty subplots
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def outlier_visualization(self, data: pd.DataFrame, column: str, 
                            outliers_info: dict) -> plt.Figure:
        """Visualize outliers"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot
        ax1.boxplot(data[column].dropna())
        ax1.set_title(f'Box Plot - {column}')
        ax1.set_ylabel(column)
        
        # Histogram with outlier boundaries
        ax2.hist(data[column].dropna(), bins=30, alpha=0.7, color=self.colors[0])
        ax2.axvline(outliers_info['bounds']['lower'], color='red', linestyle='--', 
                   label=f"Lower Bound: {outliers_info['bounds']['lower']:.2f}")
        ax2.axvline(outliers_info['bounds']['upper'], color='red', linestyle='--', 
                   label=f"Upper Bound: {outliers_info['bounds']['upper']:.2f}")
        ax2.set_title(f'Distribution with Outlier Bounds - {column}')
        ax2.set_xlabel(column)
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        return fig