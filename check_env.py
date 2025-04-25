import os
import sys
import pandas as pd
import streamlit as st

def check_environment():
    """
    Check and report on the environment, which can help debug deployment issues.
    """
    # Basic environment info
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    # Check for src directory
    if os.path.exists('src'):
        print(f"src directory exists, contents: {os.listdir('src')}")
    else:
        print("src directory does not exist")
    
    # Look for the data file in various locations
    possible_paths = [
        os.path.join("src", "earthquake_cleandata_posteda.csv"),
        "earthquake_cleandata_posteda.csv",
        os.path.join("..", "src", "earthquake_cleandata_posteda.csv"),
        "/mount/src/earthquake-analysis/src/earthquake_cleandata_posteda.csv",
        "/app/src/earthquake_cleandata_posteda.csv",
        "/mount/src/karthikmanjunath_hariharannadanasabapathi_naveenmanikandan_phase_2/src/earthquake_cleandata_posteda.csv"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Data file found at: {path}")
            print(f"File size: {os.path.getsize(path)} bytes")
            # Try to read the first few rows to verify it's valid
            try:
                df = pd.read_csv(path, nrows=5)
                print(f"File is readable, columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"Error reading file: {str(e)}")
    
    # Check for environment variables
    print("\nEnvironment Variables:")
    for key, value in os.environ.items():
        if 'PATH' in key or 'PYTHON' in key or 'STREAMLIT' in key or 'EARTHQUAKE' in key:
            print(f"{key}: {value}")

if __name__ == "__main__":
    check_environment() 