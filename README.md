<div align="center">
  <img src="https://github.com/ericmaddox/crime-analyst-ai/raw/main/media/crime_analyst_ai.JPEG" alt="Crime Analyst AI" width="200" />
</div>

# Crime Analyst AI

[![Powered by Ollama](https://img.shields.io/badge/Powered%20by-Ollama-blue)](https://ollama.com)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)

An AI-powered predictive crime analysis tool that leverages Python, Folium, Gradio, and the Ministral AI model (via Ollama) to analyze historical crime data and predict future crime trends. Features an enterprise-grade dark-themed web interface for uploading data, running predictive analysis, and visualizing results on an interactive map.

![Crime Analyst AI Map](https://github.com/ericmaddox/crime-analyst-ai/blob/main/media/crime_analyst_ai_map.JPG)

## Features

### Enterprise Web Interface
- **Professional Dark Theme** - Modern, enterprise-grade UI with carefully designed aesthetics
- **Drag-and-Drop Upload** - Easy CSV/Excel file import with automatic column detection
- **Smart Column Mapping** - Auto-detects latitude, longitude, and crime type columns
- **Real-time Progress** - Visual progress indicators during analysis

### AI-Powered Analysis
- **Statistical Preprocessing** - Computes crime distribution, geographic hotspots, and patterns
- **Intelligent Predictions** - Uses Ministral-3:3b model for context-aware crime prediction
- **Hotspot Detection** - Identifies high-crime areas using density-based clustering
- **Validation** - Ensures predictions are geographically plausible

### Interactive Visualization
- **Embedded Map View** - Interactive Folium map displayed directly in the UI
- **Heatmap Layer** - Crime density visualization with gradient coloring
- **Risk-Coded Markers** - Predictions colored by likelihood (red=high, orange=medium, green=low)
- **Marker Clustering** - Groups nearby markers for better performance
- **Legend** - Clear identification of actual vs. predicted crimes

### Analysis Results
- **Statistics Tab** - Visual crime type distribution with progress bars
- **Predictions Tab** - Sortable table with location, type, and risk levels
- **Full Report Tab** - Complete analysis summary with AI output
- **Export Options** - Download interactive map (HTML) and full report (TXT)

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai) installed and running

### Step 1: Clone the Repository

```bash
git clone https://github.com/ericmaddox/crime-analyst-ai.git
cd crime-analyst-ai
```

### Step 2: Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Ollama

1. **Download and Install Ollama**:
   Follow the instructions on the [Ollama website](https://ollama.ai) to install Ollama on your system.

2. **Pull the AI Model**:
   ```bash
   ollama pull ministral-3:3b
   ```

3. **Verify Installation**:
   ```bash
   ollama list
   ```
   You should see `ministral-3:3b` in the list.

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Launch the application (opens browser automatically)
python app.py
```

The application will automatically open in your default browser at `http://localhost:7860`.

## Usage

### Web Interface (Recommended)

1. **Launch the App**:
   ```bash
   python app.py
   ```
   The browser opens automatically to the Crime Analyst AI interface.

2. **Upload Data**:
   - Drag and drop a CSV or Excel file into the upload area
   - Or click to browse and select your file

3. **Map Columns**:
   - The system auto-detects common column names
   - Verify or adjust the Latitude, Longitude, and Crime Type mappings

4. **Run Analysis**:
   - Click "🔍 Run Predictive Analysis"
   - Watch the progress as data is processed

5. **View Results**:
   - **Interactive Map** - Explore the embedded map with heatmaps and markers
   - **Statistics** - View crime type distribution charts
   - **Predictions** - Review AI predictions with risk levels
   - **Full Report** - Read the complete analysis with AI insights

6. **Export**:
   - Download the interactive map as HTML
   - Download the full analysis report as TXT

### Command Line Interface

For automated/scripted usage:

```bash
python crime_analyst_ai.py
```

This expects a `sample_crime_data.csv` file in the project directory.

### Data Format

Your crime data file should include these columns:

| Column | Required | Description |
|--------|----------|-------------|
| Latitude | Yes | Geographic latitude (decimal degrees) |
| Longitude | Yes | Geographic longitude (decimal degrees) |
| CrimeType | Yes | Category of crime (e.g., Theft, Assault) |
| Date | No | Date of incident |
| Time | No | Time of incident |
| Address | No | Street address |

A sample data file (`sample_crime_data.csv`) with 100 Atlanta-area records is included for testing.

## Project Structure

```
crime-analyst-ai/
├── app.py                    # Gradio web interface
├── crime_analyst_ai.py       # Core analysis engine
├── sample_crime_data.csv     # Sample data for testing
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT License
├── venv/                     # Virtual environment (created during setup)
└── media/                    # Project images
    ├── crime_analyst_ai.JPEG
    └── crime_analyst_ai_map.JPG
```

## Outputs

### Interactive Map (`crime_analyst_ai_map.html`)

- **Dark-themed CartoDB basemap** for professional appearance
- **Heatmap layer** showing crime density with color gradient
- **Clustered markers** for actual crime locations (blue)
- **Prediction markers** color-coded by risk level:
  - 🔴 Red = High risk (>70% likelihood)
  - 🟠 Orange = Medium risk (40-70%)
  - 🟢 Green = Low risk (<40%)
- **Interactive legend** explaining all markers
- **Layer controls** to toggle visibility

### Analysis Report (`predicted_crime_analysis.txt`)

- Data summary (total records, date range)
- Crime type distribution with percentages
- Top crime categories
- AI predictions with:
  - Geographic coordinates
  - Crime type
  - Likelihood percentage
  - Analysis reasoning
- Raw model output

## Customization

### Change the AI Model

Edit `crime_analyst_ai.py` line 26:

```python
OLLAMA_MODEL = "your-preferred-model"
```

Compatible models include any Ollama model capable of text generation.

### Adjust Hotspot Detection

Modify the `detect_hotspots()` function in `crime_analyst_ai.py`:
- Change grid size (default: ~1km cells)
- Adjust number of hotspots returned

### Customize the UI Theme

Edit the `CUSTOM_CSS` variable in `app.py` to modify:
- Color scheme (CSS variables at top)
- Typography
- Component styling
- Layout spacing

### Modify the Analysis Prompt

Edit `build_analysis_prompt()` in `crime_analyst_ai.py` to change:
- Instructions to the AI
- Output format requirements
- Number of predictions requested

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.5.3 | Data manipulation |
| folium | ≥0.14.0 | Interactive maps |
| gradio | ≥4.0.0 | Web interface |
| numpy | ≥1.24.0 | Numerical operations |
| openpyxl | ≥3.1.2 | Excel file support |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **[Folium](https://github.com/python-visualization/folium)** - Interactive map visualization
- **[Pandas](https://pandas.pydata.org/)** - Data analysis and manipulation
- **[Gradio](https://gradio.app)** - Web interface framework
- **[Ollama](https://ollama.ai)** - Local AI model runtime
- **[Mistral AI](https://mistral.ai)** - Ministral-3:3b language model

---

<div align="center">
  <strong>Built with ❤️ for public safety</strong>
</div>
