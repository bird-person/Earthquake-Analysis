#!/bin/bash

# Create src directory if it doesn't exist
mkdir -p src

# Check if the data file is in the current src directory 
if [ -f "src/earthquake_cleandata_posteda.csv" ]; then
    echo "Data file already exists in src directory."
else
    # Look for the data file in various locations
    potential_locations=(
        "../src/earthquake_cleandata_posteda.csv"
        "earthquake_cleandata_posteda.csv"
        "../earthquake_cleandata_posteda.csv"
    )
    
    for location in "${potential_locations[@]}"; do
        if [ -f "$location" ]; then
            echo "Found data file at $location. Copying to src directory..."
            cp "$location" "src/earthquake_cleandata_posteda.csv"
            echo "File copied successfully!"
            break
        fi
    done
fi

# Check if the file is now in the correct location
if [ -f "src/earthquake_cleandata_posteda.csv" ]; then
    echo "Data file is ready for the application."
else
    echo "WARNING: Data file not found in any expected location."
    echo "Please manually ensure earthquake_cleandata_posteda.csv is in the src directory."
fi 