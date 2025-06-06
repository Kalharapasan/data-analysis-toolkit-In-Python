import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from data_loader import DataLoader
from data_analyzer import DataAnalyzer
from data_visualizer import DataVisualizer

class DataAnalysisApp:
    """Main application class for data analysis tool"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Data Analysis Tool")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_analyzer = DataAnalyzer()
        self.data_visualizer = DataVisualizer()
        
        # Current data
        self.current_data = None
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_data_tab()
        self.create_analysis_tab()
        self.create_visualization_tab()
        self.create_ml_tab()
        
    def create_data_tab(self):
        """Create data loading and viewing tab"""
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Loading")
        
        # File loading section
        load_frame = ttk.LabelFrame(self.data_frame, text="Load Data", padding="10")
        load_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(load_frame, text="Load CSV", command=self.load_csv).pack(side='left', padx=5)
        ttk.Button(load_frame, text="Load Excel", command=self.load_excel).pack(side='left', padx=5)
        ttk.Button(load_frame, text="Create Sample Data", command=self.create_sample_data).pack(side='left', padx=5)
        
        # Sample data type selection
        self.sample_type = ttk.Combobox(load_frame, values=['sales', 'customer', 'stock'], width=10)
        self.sample_type.set('sales')
        self.sample_type.pack(side='left', padx=5)
        
        # Data info section
        info_frame = ttk.LabelFrame(self.data_frame, text="Data Information", padding="10")
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, width=60)
        self.info_text.pack(side='left', fill='both', expand=True)
        
        info_scroll = ttk.Scrollbar(info_frame, orient='vertical', command=self.info_text.yview)
        info_scroll.pack(side='right', fill='y')
        self.info_text.config(yscrollcommand=info_scroll.set)
        
        # Data preview section
        preview_frame = ttk.LabelFrame(self.data_frame, text="Data Preview", padding="10")
        preview_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Treeview for data display
        self.tree = ttk.Treeview(preview_frame)
        self.tree.pack(side='left', fill='both', expand=True)
        
        tree_scroll = ttk.Scrollbar(preview_frame, orient='vertical', command=self.tree.yview)
        tree_scroll.pack(side='right', fill='y')
        self.tree.config(yscrollcommand=tree_scroll.set)
    
    def create_analysis_tab(self):
        """Create data analysis tab"""
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Data Analysis")
        
        # Analysis controls
        control_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Controls", padding="10")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # Column selection
        ttk.Label(control_frame, text="Select Column:").pack(side='left', padx=5)
        self.analysis_column = ttk.Combobox(control_frame, width=15)
        self.analysis_column.pack(side='left', padx=5)
        
        # Analysis buttons
        ttk.Button(control_frame, text="Basic Stats", command=self.basic_statistics).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Correlation", command=self.correlation_analysis).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Outliers", command=self.outlier_detection).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Group Analysis", command=self.group_analysis).pack(side='left', padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(self.analysis_frame, text="Analysis Results", padding="10")
        results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.results_text = tk.Text(results_frame, height=20, width=80)
        self.results_text.pack(side='left', fill='both', expand=True)
        
        results_scroll = ttk.Scrollbar(results_frame, orient='vertical', command=self.results_text.yview)
        results_scroll.pack(side='right', fill='y')
        self.results_text.config(yscrollcommand=results_scroll.set)
    
    def create_visualization_tab(self):
        """Create data visualization tab"""
        self.viz_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.viz_frame, text="Visualization")
        
        # Visualization controls
        viz_control_frame = ttk.LabelFrame(self.viz_frame, text="Visualization Controls", padding="10")
        viz_control_frame.pack(fill='x', padx=10, pady=5)
        
        # Plot type selection
        ttk.Label(viz_control_frame, text="Plot Type:").pack(side='left', padx=5)
        self.plot_type = ttk.Combobox(viz_control_frame, values=[
            'histogram', 'box_plot', 'scatter', 'line', 'bar', 'pie', 'correlation_heatmap'
        ], width=15)
        self.plot_type.set('histogram')
        self.plot_type.pack(side='left', padx=5)
        
        # Column selections
        ttk.Label(viz_control_frame, text="X Column:").pack(side='left', padx=5)
        self.x_column = ttk.Combobox(viz_control_frame, width=12)
        self.x_column.pack(side='left', padx=5)
        
        ttk.Label(viz_control_frame, text="Y Column:").pack(side='left', padx=5)
        self.y_column = ttk.Combobox(viz_control_frame, width=12)
        self.y_column.pack(side='left', padx=5)
        
        ttk.Button(viz_control_frame, text="Create Plot", command=self.create_plot).pack(side='left', padx=10)
        
        # Plot display area
        self.plot_frame = ttk.Frame(self.viz_frame)
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=5)
    
    def create_ml_tab(self):
        """Create machine learning tab"""
        self.ml_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_frame, text="Machine Learning")
        
        # ML controls
        ml_control_frame = ttk.LabelFrame(self.ml_frame, text="ML Controls", padding="10")
        ml_control_frame.pack(fill='x', padx=10, pady=5)
        
        # Target column selection
        ttk.Label(ml_control_frame, text="Target Column:").pack(side='left', padx=5)
        self.target_column = ttk.Combobox(ml_control_frame, width=15)
        self.target_column.pack(side='left', padx=5)
        
        # Model type selection
        ttk.Label(ml_control_frame, text="Model Type:").pack(side='left', padx=5)
        self.model_type = ttk.Combobox(ml_control_frame, values=[
            'linear_regression', 'random_forest_regression', 
            'logistic_classification', 'random_forest_classification'
        ], width=20)
        self.model_type.set('linear_regression')
        self.model_type.pack(side='left', padx=5)
        
        ttk.Button(ml_control_frame, text="Train Model", command=self.train_model).pack(side='left', padx=10)
        
        # ML results display
        ml_results_frame = ttk.LabelFrame(self.ml_frame, text="ML Results", padding="10")
        ml_results_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.ml_results_text = tk.Text(ml_results_frame, height=20, width=80)
        self.ml_results_text.pack(side='left', fill='both', expand=True)
        
        ml_scroll = ttk.Scrollbar(ml_results_frame, orient='vertical', command=self.ml_results_text.yview)
        ml_scroll.pack(side='right', fill='y')
        self.ml_results_text.config(yscrollcommand=ml_scroll.set)
    
    # Data loading methods
    def load_csv(self):
        """Load CSV file"""
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.current_data = self.data_loader.load_csv(file_path)
                self.update_data_display()
                messagebox.showinfo("Success", "CSV file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load CSV: {str(e)}")
    
    def load_excel(self):
        """Load Excel file"""
        file_path = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.current_data = self.data_loader.load_excel(file_path)
                self.update_data_display()
                messagebox.showinfo("Success", "Excel file loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load Excel: {str(e)}")
    
    def create_sample_data(self):
        """Create sample data"""
        try:
            dataset_type = self.sample_type.get()
            self.current_data = self.data_loader.create_sample_data(dataset_type)
            self.update_data_display()
            messagebox.showinfo("Success", f"Sample {dataset_type} data created successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create sample data: {str(e)}")
    
    def update_data_display(self):
        """Update data display and column lists"""
        if self.current_data is not None:
            # Update data info
            info = self.data_loader.get_data_info()
            info_text = f"Shape: {info['shape']}\n"
            info_text += f"Columns: {', '.join(info['columns'])}\n"
            info_text += f"Memory Usage: {info['memory_usage']} bytes\n\n"
            info_text += "Data Types:\n"
            for col, dtype in info['dtypes'].items():
                info_text += f"  {col}: {dtype}\n"
            info_text += "\nMissing Values:\n"
            for col, missing in info['missing_values'].items():
                if missing > 0:
                    info_text += f"  {col}: {missing}\n"
            
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(tk.END, info_text)
            
            # Update data preview
            self.update_treeview()
            
            # Update column lists
            columns = list(self.current_data.columns)
            self.analysis_column['values'] = columns
            self.x_column['values'] = columns
            self.y_column['values'] = columns
            self.target_column['values'] = columns
            
            if columns:
                self.analysis_column.set(columns[0])
                self.x_column.set(columns[0])
                if len(columns) > 1:
                    self.y_column.set(columns[1])
                    self.target_column.set(columns[-1])
    
    def update_treeview(self):
        """Update treeview with data"""
        # Clear existing data
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if self.current_data is not None:
            # Configure columns
            columns = list(self.current_data.columns)
            self.tree['columns'] = columns
            self.tree['show'] = 'headings'
            
            # Configure column headings and widths
            for col in columns:
                self.tree.heading(col, text=col)
                self.tree.column(col, width=100, minwidth=80)
            
            # Add data (first 100 rows)
            for idx, row in self.current_data.head(100).iterrows():
                values = [str(value) for value in row.values]
                self.tree.insert('', 'end', values=values)
    
    # Analysis methods
    def basic_statistics(self):
        """Perform basic statistical analysis"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        try:
            stats = self.data_analyzer.basic_statistics(self.current_data)
            
            result_text = "BASIC STATISTICS\n" + "="*50 + "\n\n"
            
            for column, column_stats in stats.items():
                result_text += f"{column}:\n"
                result_text += f"  Mean: {column_stats['mean']:.4f}\n"
                result_text += f"  Median: {column_stats['median']:.4f}\n"
                result_text += f"  Std Dev: {column_stats['std']:.4f}\n"
                result_text += f"  Min: {column_stats['min']:.4f}\n"
                result_text += f"  Max: {column_stats['max']:.4f}\n"
                result_text += f"  Skewness: {column_stats['skewness']:.4f}\n"
                result_text += f"  Kurtosis: {column_stats['kurtosis']:.4f}\n"
                result_text += f"  Missing Values: {column_stats['missing_count']}\n"
                result_text += f"  Unique Values: {column_stats['unique_count']}\n\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute statistics: {str(e)}")
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        try:
            corr_matrix = self.data_analyzer.correlation_analysis(self.current_data)
            
            result_text = "CORRELATION ANALYSIS\n" + "="*50 + "\n\n"
            result_text += str(corr_matrix.round(4))
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute correlations: {str(e)}")
    
    def outlier_detection(self):
        """Detect outliers in selected column"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        column = self.analysis_column.get()
        if not column:
            messagebox.showwarning("Warning", "Please select a column!")
            return
        
        try:
            outliers_info = self.data_analyzer.outlier_detection(self.current_data, column)
            
            result_text = f"OUTLIER DETECTION - {column}\n" + "="*50 + "\n\n"
            result_text += f"Number of outliers: {outliers_info['outliers_count']}\n"
            result_text += f"Percentage of outliers: {outliers_info['outliers_percentage']:.2f}%\n"
            result_text += f"Lower bound: {outliers_info['bounds']['lower']:.4f}\n"
            result_text += f"Upper bound: {outliers_info['bounds']['upper']:.4f}\n\n"
            
            if outliers_info['outliers_count'] > 0:
                result_text += "Outlier values (first 20):\n"
                outlier_values = outliers_info['outliers_values'][:20]
                result_text += ", ".join([f"{val:.4f}" for val in outlier_values])
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to detect outliers: {str(e)}")
    
    def group_analysis(self):
        """Perform group analysis"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        # Simple dialog for group by selection
        group_dialog = GroupAnalysisDialog(self.root, list(self.current_data.columns))
        if group_dialog.result:
            try:
                group_by, agg_column = group_dialog.result
                grouped_data = self.data_analyzer.group_analysis(
                    self.current_data, group_by, agg_column
                )
                
                result_text = f"GROUP ANALYSIS\n" + "="*50 + "\n"
                result_text += f"Grouped by: {group_by}\n"
                result_text += f"Aggregated column: {agg_column}\n\n"
                result_text += str(grouped_data)
                
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, result_text)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to perform group analysis: {str(e)}")
    
    # Visualization methods
    def create_plot(self):
        """Create visualization based on selected parameters"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        plot_type = self.plot_type.get()
        x_col = self.x_column.get()
        y_col = self.y_column.get()
        
        try:
            # Clear previous plot
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            fig = None
            
            if plot_type == 'histogram' and x_col:
                fig = self.data_visualizer.histogram(self.current_data, x_col)
            elif plot_type == 'box_plot' and x_col:
                fig = self.data_visualizer.box_plot(self.current_data, x_col)
            elif plot_type == 'scatter' and x_col and y_col:
                fig = self.data_visualizer.scatter_plot(self.current_data, x_col, y_col)
            elif plot_type == 'line' and x_col and y_col:
                fig = self.data_visualizer.line_plot(self.current_data, x_col, y_col)
            elif plot_type == 'bar' and x_col and y_col:
                fig = self.data_visualizer.bar_plot(self.current_data, x_col, y_col)
            elif plot_type == 'pie' and x_col:
                fig = self.data_visualizer.pie_chart(self.current_data, x_col)
            elif plot_type == 'correlation_heatmap':
                corr_matrix = self.data_analyzer.correlation_analysis(self.current_data)
                fig = self.data_visualizer.correlation_heatmap(corr_matrix)
            
            if fig:
                canvas = FigureCanvasTkAgg(fig, self.plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)
            else:
                messagebox.showwarning("Warning", "Please select appropriate columns for the plot type!")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plot: {str(e)}")
    
    # Machine Learning methods
    def train_model(self):
        """Train machine learning model"""
        if self.current_data is None:
            messagebox.showwarning("Warning", "Please load data first!")
            return
        
        target_col = self.target_column.get()
        model_type = self.model_type.get()
        
        if not target_col:
            messagebox.showwarning("Warning", "Please select a target column!")
            return
        
        try:
            # Prepare data
            X_train, X_test, y_train, y_test = self.data_analyzer.prepare_data_for_ml(
                self.current_data, target_col
            )
            
            result_text = f"MACHINE LEARNING RESULTS\n" + "="*50 + "\n"
            result_text += f"Model Type: {model_type}\n"
            result_text += f"Target Column: {target_col}\n"
            result_text += f"Training Set Size: {len(X_train)}\n"
            result_text += f"Test Set Size: {len(X_test)}\n\n"
            
            # Train model based on type
            if 'regression' in model_type:
                if model_type == 'linear_regression':
                    results = self.data_analyzer.regression_analysis(
                        X_train, X_test, y_train, y_test, 'linear'
                    )
                else:
                    results = self.data_analyzer.regression_analysis(
                        X_train, X_test, y_train, y_test, 'random_forest'
                    )
                
                result_text += f"Training R²: {results['train_r2']:.4f}\n"
                result_text += f"Test R²: {results['test_r2']:.4f}\n"
                result_text += f"Training MSE: {results['train_mse']:.4f}\n"
                result_text += f"Test MSE: {results['test_mse']:.4f}\n"
                
            elif 'classification' in model_type:
                if model_type == 'logistic_classification':
                    results = self.data_analyzer.classification_analysis(
                        X_train, X_test, y_train, y_test, 'logistic'
                    )
                else:
                    results = self.data_analyzer.classification_analysis(
                        X_train, X_test, y_train, y_test, 'random_forest'
                    )
                
                result_text += f"Training Accuracy: {results['train_accuracy']:.4f}\n"
                result_text += f"Test Accuracy: {results['test_accuracy']:.4f}\n\n"
                result_text += "Classification Report:\n"
                result_text += results['classification_report']
            
            self.ml_results_text.delete(1.0, tk.END)
            self.ml_results_text.insert(tk.END, result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")


class GroupAnalysisDialog:
    """Dialog for group analysis parameter selection"""
    
    def __init__(self, parent, columns):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Group Analysis Parameters")
        self.dialog.geometry("300x150")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Group by column
        ttk.Label(self.dialog, text="Group by column:").pack(pady=5)
        self.group_by_var = tk.StringVar()
        group_by_combo = ttk.Combobox(self.dialog, textvariable=self.group_by_var, values=columns)
        group_by_combo.pack(pady=5)
        
        # Aggregation column
        ttk.Label(self.dialog, text="Aggregation column:").pack(pady=5)
        self.agg_col_var = tk.StringVar()
        agg_col_combo = ttk.Combobox(self.dialog, textvariable=self.agg_col_var, values=columns)
        agg_col_combo.pack(pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="OK", command=self.ok_clicked).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.cancel_clicked).pack(side='left', padx=5)
        
        self.dialog.wait_window()
    
    def ok_clicked(self):
        if self.group_by_var.get() and self.agg_col_var.get():
            self.result = (self.group_by_var.get(), self.agg_col_var.get())
        self.dialog.destroy()
    
    def cancel_clicked(self):
        self.dialog.destroy()


def main():
    """Main function to run the application"""
    root = tk.Tk()
    app = DataAnalysisApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
        