import subprocess
import folium
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("crime_analyst_ai.log"),
                        logging.StreamHandler()
                    ])

# Function to run Ollama AI model for predictive analysis
def run_ollama_predictive_model(prompt, data_for_model):
    try:
        process = subprocess.run(
            ['ollama', 'run', 'llama3.2', prompt],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info("Ollama model ran successfully")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running Ollama model: {e}")
        logging.error("Ollama stderr:")
        logging.error(e.stderr)
        raise
    output = process.stdout
    logging.debug("Ollama stdout:")
    logging.debug(output)
    if not output.strip():
        raise ValueError("The Ollama model output is empty. Please check the model and try again.")
    return output

# Function to read crime data from a file (CSV or XLSX)
def read_crime_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        # Read in chunks and concatenate
        chunks = pd.read_csv(file_path, encoding='ISO-8859-1', chunksize=10000)
        return pd.concat(chunk for chunk in chunks)
    elif file_extension == '.xlsx':
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use a .csv or .xlsx file.")

# Function to dynamically generate insights based on the model output
def extract_insights_from_output(output):
    insights = []
    lines = output.split('\n')
    for line in lines:
        if "latitude:" in line.lower() and "longitude:" in line.lower():
            try:
                parts = line.split(',')
                latitude = longitude = None
                crime_type = "Unknown"
                prediction = "No prediction"
                likelihood = "Unknown"
                for part in parts:
                    if "latitude:" in part.lower():
                        latitude = float(part.split(':')[-1].strip())
                    if "longitude:" in part.lower():
                        longitude = float(part.split(':')[-1].strip())
                    if "crime type:" in part.lower():
                        crime_type = part.split(':')[-1].strip()
                    if "prediction:" in part.lower():
                        prediction = part.split(':')[-1].strip()
                    if "likelihood:" in part.lower():
                        likelihood = part.split(':')[-1].strip()
                if latitude is not None and longitude is not None:
                    insights.append({
                        "Latitude": latitude,
                        "Longitude": longitude,
                        "CrimeType": crime_type,
                        "Prediction": prediction,
                        "Likelihood": likelihood
                    })
            except ValueError:
                continue
    return insights

# Function to validate predictions based on historical context
def validate_predictions_with_historical_data(actual_data, insights):
    predicted_crimes = [insight['CrimeType'] for insight in insights]
    actual_crimes = actual_data['CrimeType'].value_counts().index.tolist()

    for predicted_crime in predicted_crimes:
        if predicted_crime not in actual_crimes:
            logging.warning(f"Predicted crime type '{predicted_crime}' is not commonly observed in the provided data.")
    
    # You can add more checks for geographical or temporal consistency based on the crime data.

# Function to check likelihood values
def check_prediction_likelihood(insights):
    for insight in insights:
        try:
            likelihood = float(insight['Likelihood'].strip('%'))
            if likelihood < 0 or likelihood > 100:
                logging.warning(f"Likelihood value {insight['Likelihood']} for crime type '{insight['CrimeType']}' is out of bounds.")
        except ValueError:
            logging.warning(f"Invalid likelihood value '{insight['Likelihood']}' for crime type '{insight['CrimeType']}'.")

# Function to create the map with visualizations
def create_crime_map(actual_data, insights, output_file='crime_analyst_ai_map.html'):
    map_center = [33.75, -84.5]
    crime_map = folium.Map(location=map_center, zoom_start=10)
    
    from folium.plugins import HeatMap
    heat_data_actual = [[row['Latitude'], row['Longitude']] for _, row in actual_data.iterrows()]
    HeatMap(heat_data_actual).add_to(crime_map)
    
    for _, row in actual_data.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"{row['CrimeType']} - Actual Data",
            icon=folium.Icon(color='green')
        ).add_to(crime_map)
    
    for insight in insights:
        folium.Marker(
            location=[insight['Latitude'], insight['Longitude']],
            popup=folium.Popup(f"<b>{insight['CrimeType']}</b><br>Prediction: {insight['Prediction']}<br>Likelihood: {insight['Likelihood']}", max_width=250),
            icon=folium.Icon(color='purple', icon='info-sign')
        ).add_to(crime_map)

    crime_map.save(output_file)
    logging.info(f"Map has been created and saved as '{output_file}'.")

# Function to save the narrative analysis to a text file
def save_analysis_to_file(output, file_name='predicted_crime_analysis.txt'):
    with open(file_name, 'w') as f:
        f.write(output)
    logging.info(f"Predicted crime analysis has been saved as '{file_name}'.")

# Function to compare predictions from scikit-learn and Ollama
def compare_predictions(y_test, y_pred_sklearn, ollama_predictions):
    if len(y_test) != len(ollama_predictions):
        logging.warning("Inconsistent number of samples between y_test and ollama_predictions")
        return None
    comparison_df = pd.DataFrame({
        'Actual': y_test,
        'scikit-learn Prediction': y_pred_sklearn,
        'Ollama Prediction': ollama_predictions
    })
    logging.info("\nComparison of Predictions:\n")
    logging.info(comparison_df.head())
    return comparison_df

# Function to perform detailed analysis
def detailed_analysis(comparison_df):
    if comparison_df is not None:
        mismatches = comparison_df[comparison_df['scikit-learn Prediction'] != comparison_df['Ollama Prediction']]
        logging.info("\nMismatches between models:\n")
        logging.info(mismatches.head())

def main():
    file_path = 'sample_crime_data.xlsx'
    actual_crime_data = read_crime_data(file_path)
    
    # Updated prompt to ensure data usage and historical knowledge consideration
    prompt = (
        "Based on the following crime data (latitude, longitude, crime type), use your historical crime knowledge to "
        "make predictions on potential future crime hotspots. Your predictions should include latitude, longitude, crime type, "
        "and likelihood of the crime happening at that location. Make sure your predictions account for crime patterns, "
        "geographic hotspots, and trends over time. Output 10 predictions formatted as follows: "
        "'Latitude: <value>, Longitude: <value>, Crime Type: <type>, Prediction: <prediction>, Likelihood: <likelihood>'.\n"
        + actual_crime_data.to_csv(index=False)
    )
    
    predicted_crime_data_output = run_ollama_predictive_model(prompt, actual_crime_data)
    
    logging.info("Predicted Crime Data Output:")
    logging.info(predicted_crime_data_output)
    
    insights = extract_insights_from_output(predicted_crime_data_output)
    if not insights:
        logging.error("No insights extracted. Please check the Ollama model output and ensure it follows the expected format.")
        return
    
    validate_predictions_with_historical_data(actual_crime_data, insights)
    check_prediction_likelihood(insights)
    
    df = actual_crime_data.copy()
    df['CrimeTypeNum'] = df['CrimeType'].astype('category').cat.codes
    X = df[['Latitude', 'Longitude']]
    y = df['CrimeTypeNum']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred_sklearn = clf.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, y_pred_sklearn)
    logging.info(f"scikit-learn Model Accuracy: {sklearn_accuracy:.3f}")
    
    # Extracting predictions from insights and ensuring the length matches
    ollama_predictions = [insight['CrimeType'] for insight in insights if 'CrimeType' in insight]
    
    # Truncate or fill the predictions to ensure the length matches y_test
    if len(ollama_predictions) > len(y_test):
        ollama_predictions = ollama_predictions[:len(y_test)]
    elif len(ollama_predictions) < len(y_test):
        ollama_predictions.extend(["Unknown"] * (len(y_test) - len(ollama_predictions)))

    ollama_accuracy = accuracy_score(y_test, ollama_predictions)
    logging.info(f"Ollama Model Accuracy: {ollama_accuracy:.3f}")

    comparison_df = compare_predictions(y_test, y_pred_sklearn, ollama_predictions)
    detailed_analysis(comparison_df)

    create_crime_map(actual_crime_data, insights)
    save_analysis_to_file(predicted_crime_data_output)

if __name__ == '__main__':
    main()
