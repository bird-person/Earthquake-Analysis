# Earthquake Risk Analysis

An interactive web application for visualizing earthquake data across North America and predicting potential earthquake magnitudes using machine learning.

## Features

- **Interactive Heatmap**: Visualize earthquake intensity across North America with customizable filters for magnitude and date range
- **Zipcode Risk Checker**: Check earthquake risk for specific zip codes, with detailed statistics and visualizations
- **Magnitude Predictor**: Use machine learning (Random Forest or Gradient Boosting) to predict potential earthquake magnitudes for specific locations
- **Data Insights**: Explore trends, patterns, and correlations in earthquake data
- **Custom Dataset Support**: Upload your own earthquake dataset in CSV format

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/earthquake-risk-analysis.git
cd earthquake-risk-analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

Then open your browser and navigate to http://localhost:8501

## Data Requirements

If you want to upload your own dataset, it must include these columns:
- `LATITUDE`: Earthquake latitude
- `LONGITUDE`: Earthquake longitude  
- `MAGNITUDE`: Earthquake magnitude
- `DATE`: Date of the earthquake

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn (Random Forest & Gradient Boosting)
- Plotly
- GeoPy

## Contributors

- Karthik Manjunath
- Hariharan Nadanasabapathi
- Naveen Manikandan 