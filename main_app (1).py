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
        