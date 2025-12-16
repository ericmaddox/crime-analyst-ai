"""
Crime Analyst AI - Core Analysis Engine
Predictive crime analysis using Ollama LLM (ministral-3:3b)
"""

import subprocess
import folium
from folium.plugins import HeatMap, MarkerCluster
import pandas as pd
import numpy as np
import os
import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# Configure logging
LOG_FILE = OUTPUT_DIR / "crime_analyst_ai.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Model configuration
OLLAMA_MODEL = "ministral-3:3b"


def run_ollama_predictive_model(prompt: str) -> str:
    """
    Run Ollama AI model for predictive analysis.
    Uses stdin to pass prompt (handles large prompts correctly).
    
    Args:
        prompt: The analysis prompt to send to the model
        
    Returns:
        The model's response text
    """
    try:
        process = subprocess.run(
            ['ollama', 'run', OLLAMA_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Ollama model ({OLLAMA_MODEL}) ran successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running Ollama model: {e}")
        logging.error(f"Ollama stderr: {e.stderr}")
        raise RuntimeError(f"Ollama model failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("Ollama is not installed or not in PATH. Please install Ollama first.")
    
    output = process.stdout
    logging.debug(f"Ollama output: {output[:500]}...")
    
    if not output.strip():
        raise ValueError("The Ollama model output is empty. Please check the model and try again.")
    
    return output


def read_crime_data(file_path: str) -> pd.DataFrame:
    """
    Read crime data from a file (CSV or XLSX).
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame containing the crime data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        # Read in chunks for large files
        chunks = pd.read_csv(file_path, encoding='ISO-8859-1', chunksize=10000)
        df = pd.concat(chunk for chunk in chunks)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use a .csv or .xlsx file.")
    
    logging.info(f"Loaded {len(df)} records from {file_path}")
    return df


def validate_columns(df: pd.DataFrame, lat_col: str, lon_col: str, type_col: str) -> pd.DataFrame:
    """
    Validate and normalize required columns in the DataFrame.
    
    Args:
        df: The crime data DataFrame
        lat_col: Name of the latitude column
        lon_col: Name of the longitude column  
        type_col: Name of the crime type column
        
    Returns:
        DataFrame with standardized column names
    """
    required = {lat_col: 'Latitude', lon_col: 'Longitude', type_col: 'CrimeType'}
    
    for col in required.keys():
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data. Available columns: {list(df.columns)}")
    
    # Rename to standard names
    df = df.rename(columns=required)
    
    # Clean latitude/longitude - convert to numeric
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    # Remove rows with invalid coordinates
    initial_count = len(df)
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    if len(df) < initial_count:
        logging.warning(f"Removed {initial_count - len(df)} rows with invalid coordinates")
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after cleaning coordinates")
    
    logging.info(f"Validated {len(df)} records with columns: Latitude, Longitude, CrimeType")
    return df


def compute_crime_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistics about the crime data.
    
    Args:
        df: DataFrame with Latitude, Longitude, CrimeType columns
        
    Returns:
        Dictionary containing crime statistics
    """
    stats = {
        'total_records': len(df),
        'geographic_bounds': {
            'lat_min': float(df['Latitude'].min()),
            'lat_max': float(df['Latitude'].max()),
            'lon_min': float(df['Longitude'].min()),
            'lon_max': float(df['Longitude'].max()),
            'center_lat': float(df['Latitude'].mean()),
            'center_lon': float(df['Longitude'].mean())
        },
        'crime_distribution': {},
        'top_crime_types': []
    }
    
    # Crime type distribution
    crime_counts = df['CrimeType'].value_counts()
    total = len(df)
    
    for crime_type, count in crime_counts.items():
        percentage = round((count / total) * 100, 1)
        stats['crime_distribution'][str(crime_type)] = {
            'count': int(count),
            'percentage': percentage
        }
    
    # Top 5 crime types
    stats['top_crime_types'] = [
        {'type': str(ct), 'count': int(c), 'percentage': round((c/total)*100, 1)}
        for ct, c in crime_counts.head(5).items()
    ]
    
    logging.info(f"Computed statistics: {stats['total_records']} records, {len(crime_counts)} crime types")
    return stats


def detect_hotspots(df: pd.DataFrame, n_hotspots: int = 10) -> List[Dict[str, Any]]:
    """
    Detect crime hotspots using geographic density analysis.
    Uses a simple grid-based approach for hotspot detection.
    
    Args:
        df: DataFrame with Latitude, Longitude, CrimeType columns
        n_hotspots: Number of top hotspots to return
        
    Returns:
        List of hotspot dictionaries with location and crime info
    """
    # Round coordinates to create grid cells (approximately 0.01 degree ~ 1km)
    df_copy = df.copy()
    df_copy['lat_grid'] = (df_copy['Latitude'] * 100).round() / 100
    df_copy['lon_grid'] = (df_copy['Longitude'] * 100).round() / 100
    
    # Count crimes per grid cell
    grid_counts = df_copy.groupby(['lat_grid', 'lon_grid']).agg({
        'CrimeType': ['count', lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown']
    }).reset_index()
    
    grid_counts.columns = ['lat', 'lon', 'count', 'dominant_crime']
    grid_counts = grid_counts.sort_values('count', ascending=False)
    
    hotspots = []
    for _, row in grid_counts.head(n_hotspots).iterrows():
        # Get crime breakdown for this cell
        cell_crimes = df_copy[
            (df_copy['lat_grid'] == row['lat']) & 
            (df_copy['lon_grid'] == row['lon'])
        ]['CrimeType'].value_counts().head(3).to_dict()
        
        hotspots.append({
            'latitude': float(row['lat']),
            'longitude': float(row['lon']),
            'incident_count': int(row['count']),
            'dominant_crime': str(row['dominant_crime']),
            'crime_breakdown': {str(k): int(v) for k, v in cell_crimes.items()}
        })
    
    logging.info(f"Detected {len(hotspots)} hotspots")
    return hotspots


def build_analysis_prompt(stats: Dict, hotspots: List[Dict]) -> str:
    """
    Build a context-rich prompt for the LLM based on crime statistics.
    
    Args:
        stats: Crime statistics dictionary
        hotspots: List of detected hotspots
        
    Returns:
        Formatted prompt string for the LLM
    """
    bounds = stats['geographic_bounds']
    
    prompt = f"""You are a crime analyst AI. Based on the following historical crime data analysis, predict 10 likely future crime locations with likelihood scores.

## HISTORICAL CRIME DATA SUMMARY

**Total Records:** {stats['total_records']} incidents

**Geographic Area:**
- Latitude Range: {bounds['lat_min']:.4f} to {bounds['lat_max']:.4f}
- Longitude Range: {bounds['lon_min']:.4f} to {bounds['lon_max']:.4f}
- Center Point: ({bounds['center_lat']:.4f}, {bounds['center_lon']:.4f})

**Crime Type Distribution:**
"""
    
    for crime_info in stats['top_crime_types']:
        prompt += f"- {crime_info['type']}: {crime_info['count']} incidents ({crime_info['percentage']}%)\n"
    
    prompt += "\n**Identified Hotspot Areas:**\n"
    for i, hotspot in enumerate(hotspots[:5], 1):
        prompt += f"{i}. Location ({hotspot['latitude']:.4f}, {hotspot['longitude']:.4f}): "
        prompt += f"{hotspot['incident_count']} incidents, primarily {hotspot['dominant_crime']}\n"
    
    prompt += """
## YOUR TASK

Based on this analysis, predict 10 locations where crimes are likely to occur in the future. 
Consider patterns in the data, hotspot clustering, and crime type distributions.

**IMPORTANT: You must respond with ONLY a valid JSON array. No other text before or after.**

Respond with this exact JSON format:
```json
[
  {
    "latitude": 33.7550,
    "longitude": -84.3900,
    "crime_type": "Theft",
    "prediction": "High-traffic commercial area with historical theft patterns",
    "likelihood": 85
  }
]
```

The likelihood should be a number from 0-100 representing the probability percentage.
Make sure all predicted coordinates are within the geographic bounds provided.
"""
    
    return prompt


def parse_llm_response(output: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM response to extract predictions.
    Attempts JSON parsing first, then falls back to regex parsing.
    
    Args:
        output: Raw LLM output string
        
    Returns:
        List of prediction dictionaries
    """
    insights = []
    
    # Try to extract JSON from the response
    json_match = re.search(r'\[\s*\{.*?\}\s*\]', output, re.DOTALL)
    
    if json_match:
        try:
            json_str = json_match.group()
            predictions = json.loads(json_str)
            
            for pred in predictions:
                if 'latitude' in pred and 'longitude' in pred:
                    insights.append({
                        'Latitude': float(pred.get('latitude', 0)),
                        'Longitude': float(pred.get('longitude', 0)),
                        'CrimeType': str(pred.get('crime_type', 'Unknown')),
                        'Prediction': str(pred.get('prediction', 'No details')),
                        'Likelihood': str(pred.get('likelihood', 50)) + '%'
                    })
            
            if insights:
                logging.info(f"Successfully parsed {len(insights)} predictions from JSON")
                return insights
        except json.JSONDecodeError as e:
            logging.warning(f"JSON parsing failed: {e}, trying regex fallback")
    
    # Fallback: regex parsing for older format
    lines = output.split('\n')
    for line in lines:
        if 'latitude' in line.lower() and 'longitude' in line.lower():
            try:
                lat_match = re.search(r'latitude[:\s]+(-?\d+\.?\d*)', line, re.IGNORECASE)
                lon_match = re.search(r'longitude[:\s]+(-?\d+\.?\d*)', line, re.IGNORECASE)
                
                if lat_match and lon_match:
                    crime_match = re.search(r'crime[_\s]?type[:\s]+([^,]+)', line, re.IGNORECASE)
                    likelihood_match = re.search(r'likelihood[:\s]+(\d+)', line, re.IGNORECASE)
                    
                    insights.append({
                        'Latitude': float(lat_match.group(1)),
                        'Longitude': float(lon_match.group(1)),
                        'CrimeType': crime_match.group(1).strip() if crime_match else 'Unknown',
                        'Prediction': 'Predicted crime location',
                        'Likelihood': (likelihood_match.group(1) + '%') if likelihood_match else '50%'
                    })
            except (ValueError, AttributeError):
                continue
    
    logging.info(f"Parsed {len(insights)} predictions using regex fallback")
    return insights


def validate_predictions(insights: List[Dict], stats: Dict) -> List[Dict]:
    """
    Validate predictions against the historical data bounds.
    
    Args:
        insights: List of prediction dictionaries
        stats: Crime statistics with geographic bounds
        
    Returns:
        Filtered list of valid predictions
    """
    bounds = stats['geographic_bounds']
    valid_insights = []
    
    for insight in insights:
        lat = insight['Latitude']
        lon = insight['Longitude']
        
        # Check if within bounds (with 10% margin)
        lat_margin = (bounds['lat_max'] - bounds['lat_min']) * 0.1
        lon_margin = (bounds['lon_max'] - bounds['lon_min']) * 0.1
        
        if (bounds['lat_min'] - lat_margin <= lat <= bounds['lat_max'] + lat_margin and
            bounds['lon_min'] - lon_margin <= lon <= bounds['lon_max'] + lon_margin):
            valid_insights.append(insight)
        else:
            logging.warning(f"Prediction outside bounds: ({lat}, {lon})")
    
    # Validate likelihood values
    for insight in valid_insights:
        try:
            likelihood = float(insight['Likelihood'].strip('%'))
            if likelihood < 0 or likelihood > 100:
                insight['Likelihood'] = '50%'
                logging.warning(f"Corrected out-of-bounds likelihood for {insight['CrimeType']}")
        except ValueError:
            insight['Likelihood'] = '50%'
    
    return valid_insights


def get_risk_color(likelihood: str) -> str:
    """Get marker color based on likelihood percentage."""
    try:
        value = float(likelihood.strip('%'))
        if value >= 70:
            return 'red'
        elif value >= 40:
            return 'orange'
        else:
            return 'green'
    except ValueError:
        return 'gray'


def create_crime_map(
    actual_data: pd.DataFrame,
    insights: List[Dict],
    stats: Dict,
    output_file: Optional[str] = None
) -> str:
    """
    Create an interactive map with actual crime data and predictions.
    
    Args:
        actual_data: DataFrame with actual crime data
        insights: List of prediction dictionaries
        stats: Crime statistics for centering
        output_file: Output HTML file path (optional, defaults to output directory)
        
    Returns:
        Path to the generated HTML file
    """
    if output_file is None:
        output_file = str(OUTPUT_DIR / 'crime_analyst_ai_map.html')
    
    bounds = stats['geographic_bounds']
    map_center = [bounds['center_lat'], bounds['center_lon']]
    
    # Create map with dark tiles for professional look
    crime_map = folium.Map(
        location=map_center,
        zoom_start=12,
        tiles='CartoDB dark_matter'
    )
    
    # Add heatmap layer for actual crimes
    heat_data = [[row['Latitude'], row['Longitude']] for _, row in actual_data.iterrows()]
    HeatMap(
        heat_data,
        name='Crime Density Heatmap',
        radius=15,
        blur=10,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
    ).add_to(crime_map)
    
    # Add marker cluster for actual crimes
    actual_cluster = MarkerCluster(name='Actual Crimes').add_to(crime_map)
    
    # Sample actual data if too large (for performance)
    sample_size = min(500, len(actual_data))
    sample_data = actual_data.sample(n=sample_size) if len(actual_data) > sample_size else actual_data
    
    for _, row in sample_data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=f"<b>Actual Crime</b><br>{row['CrimeType']}",
            color='#3388ff',
            fill=True,
            fillOpacity=0.7
        ).add_to(actual_cluster)
    
    # Add predicted crime markers
    predictions_group = folium.FeatureGroup(name='Predicted Crimes').add_to(crime_map)
    
    for insight in insights:
        color = get_risk_color(insight['Likelihood'])
        
        folium.Marker(
            location=[insight['Latitude'], insight['Longitude']],
            popup=folium.Popup(
                f"""<div style='font-family: Arial; min-width: 200px;'>
                    <h4 style='margin: 0 0 10px 0; color: #333;'>Predicted Crime</h4>
                    <b>Type:</b> {insight['CrimeType']}<br>
                    <b>Likelihood:</b> {insight['Likelihood']}<br>
                    <b>Analysis:</b> {insight['Prediction']}
                </div>""",
                max_width=300
            ),
            icon=folium.Icon(color=color, icon='exclamation-triangle', prefix='fa')
        ).add_to(predictions_group)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: rgba(30, 30, 30, 0.9); padding: 15px; 
                border-radius: 8px; font-family: Arial; color: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3);">
        <h4 style="margin: 0 0 10px 0; border-bottom: 1px solid #555; padding-bottom: 5px;">
            Crime Analyst AI
        </h4>
        <div style="margin: 5px 0;"><span style="color: #3388ff;">●</span> Actual Crime</div>
        <div style="margin: 5px 0;"><span style="color: #dc3545;">▲</span> High Risk (&gt;70%)</div>
        <div style="margin: 5px 0;"><span style="color: #fd7e14;">▲</span> Medium Risk (40-70%)</div>
        <div style="margin: 5px 0;"><span style="color: #28a745;">▲</span> Low Risk (&lt;40%)</div>
    </div>
    """
    crime_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(crime_map)
    
    crime_map.save(output_file)
    logging.info(f"Map saved to '{output_file}'")
    
    return output_file


def save_analysis_report(
    output: str,
    stats: Dict,
    insights: List[Dict],
    file_name: Optional[str] = None
) -> str:
    """
    Save a comprehensive analysis report to a text file.
    
    Args:
        output: Raw LLM output
        stats: Crime statistics
        insights: Parsed predictions
        file_name: Output file path (optional, defaults to output directory)
        
    Returns:
        Path to the saved report
    """
    if file_name is None:
        file_name = str(OUTPUT_DIR / 'predicted_crime_analysis.txt')
    
    with open(file_name, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("CRIME ANALYST AI - PREDICTIVE ANALYSIS REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATA SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Records Analyzed: {stats['total_records']}\n")
        f.write(f"Crime Types: {len(stats['crime_distribution'])}\n\n")
        
        f.write("TOP CRIME TYPES\n")
        f.write("-" * 40 + "\n")
        for crime in stats['top_crime_types']:
            f.write(f"  {crime['type']}: {crime['count']} ({crime['percentage']}%)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("PREDICTIONS\n")
        f.write("=" * 60 + "\n\n")
        
        for i, insight in enumerate(insights, 1):
            f.write(f"Prediction #{i}\n")
            f.write(f"  Location: ({insight['Latitude']:.4f}, {insight['Longitude']:.4f})\n")
            f.write(f"  Crime Type: {insight['CrimeType']}\n")
            f.write(f"  Likelihood: {insight['Likelihood']}\n")
            f.write(f"  Analysis: {insight['Prediction']}\n\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("RAW MODEL OUTPUT\n")
        f.write("=" * 60 + "\n\n")
        f.write(output)
    
    logging.info(f"Analysis report saved to '{file_name}'")
    return file_name


def run_analysis(
    df: pd.DataFrame,
    lat_col: str = 'Latitude',
    lon_col: str = 'Longitude',
    type_col: str = 'CrimeType'
) -> Tuple[Dict, List[Dict], str, str]:
    """
    Run the complete crime analysis pipeline.
    
    Args:
        df: Raw crime data DataFrame
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        type_col: Name of crime type column
        
    Returns:
        Tuple of (statistics, predictions, map_path, report_path)
    """
    # Validate and normalize data
    df = validate_columns(df, lat_col, lon_col, type_col)
    
    # Compute statistics
    stats = compute_crime_statistics(df)
    
    # Detect hotspots
    hotspots = detect_hotspots(df)
    
    # Build prompt and query LLM
    prompt = build_analysis_prompt(stats, hotspots)
    logging.info("Querying LLM for predictions...")
    
    llm_output = run_ollama_predictive_model(prompt)
    
    # Parse and validate predictions
    insights = parse_llm_response(llm_output)
    
    if not insights:
        logging.warning("No predictions extracted from LLM output")
        # Generate fallback predictions based on hotspots
        insights = [
            {
                'Latitude': h['latitude'],
                'Longitude': h['longitude'],
                'CrimeType': h['dominant_crime'],
                'Prediction': f"Historical hotspot with {h['incident_count']} incidents",
                'Likelihood': f"{min(90, 50 + h['incident_count'])}%"
            }
            for h in hotspots[:10]
        ]
        logging.info("Using hotspot-based fallback predictions")
    
    insights = validate_predictions(insights, stats)
    
    # Generate outputs
    map_path = create_crime_map(df, insights, stats)
    report_path = save_analysis_report(llm_output, stats, insights)
    
    return stats, insights, map_path, report_path


def main():
    """Main entry point for command-line usage."""
    sample_file = DATA_DIR / 'sample_crime_data.csv'
    
    if not sample_file.exists():
        logging.error(f"Data file not found: {sample_file}")
        logging.info("Please provide a crime data file with columns: Latitude, Longitude, CrimeType")
        return
    
    try:
        df = read_crime_data(str(sample_file))
        stats, insights, map_path, report_path = run_analysis(df)
        
        print("\n" + "=" * 50)
        print("CRIME ANALYST AI - ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"\nRecords analyzed: {stats['total_records']}")
        print(f"Predictions generated: {len(insights)}")
        print(f"\nOutputs:")
        print(f"  - Map: {map_path}")
        print(f"  - Report: {report_path}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise


if __name__ == '__main__':
    main()

