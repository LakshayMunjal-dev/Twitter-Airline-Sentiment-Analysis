# Import necessary functions from modules within the package
from .data_preprocessing import preprocess_data
from .model_training import train_model
from .evaluation import evaluate_model
from .visualizations import visualize

# Define package metadata
__version__ = '0.1.0'
__author__ = 'LakshayMunjal-dev'
__email__ = 'lakshaymunjaldev@gmail.com'

def package_info():
    print(f"Package Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Contact: {__email__}")
