<div align="center">
  <img src="https://github.com/ericmaddox/crime-analyst-ai/raw/main/media/crime_analyst_ai.JPEG" alt="Crime Analyst AI" width="200" />
</div>

# Crime Analyst AI

[![Powered by Ollama](https://img.shields.io/badge/Powered%20by-Ollama-blue)](https://ollama.com)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered predictive crime analysis tool that leverages Python, Folium, Gradio, and the Ministral AI model (via Ollama) to analyze historical crime data and predict future crime trends. Features an enterprise-grade dark-themed web interface for uploading data, running predictive analysis, and visualizing results on an interactive map.

![Crime Analyst AI Interface](https://github.com/ericmaddox/crime-analyst-ai/blob/main/media/crime_analyst_ai_v2.png)

## Features

### Enterprise Web Interface
- **Professional Dark Theme** - Modern, enterprise-grade UI with carefully designed aesthetics
- **Drag-and-Drop Upload** - Easy CSV/Excel file import with automatic column detection
- **Smart Column Mapping** - Auto-detects latitude, longitude, crime type, date, and time columns
- **Real-time Progress** - Visual progress indicators during analysis

### AI-Powered Analysis
- **Statistical Preprocessing** - Computes crime distribution, geographic hotspots, and patterns
- **Intelligent Predictions** - Uses Ministral-3:3b model for context-aware crime prediction
- **DBSCAN Hotspot Detection** - Identifies natural crime clusters using density-based spatial clustering
- **Seasonal Intelligence** - AI considers holiday effects, payday patterns, and seasonal trends
- **Crime-Type Intelligence** - AI learns when specific crime types typically occur for targeted predictions
- **Validation** - Ensures predictions are geographically plausible

### Temporal Analysis
- **Time-of-Day Patterns** - Identifies peak crime hours (morning, afternoon, evening, night)
- **Day-of-Week Analysis** - Detects which days have the highest crime activity
- **Trend Detection** - Determines if crime is rising, falling, or stable over time
- **Per-Hotspot Timing** - Each hotspot shows its own peak hours and days
- **Crime-Specific Patterns** - Analyzes when each crime type typically occurs (e.g., thefts peak at 2 PM on Fridays)

### Recency Weighting
- **Exponential Decay** - Recent crimes weighted more heavily than older data (30-day half-life)
- **Emerging Hotspots** - Automatically flags areas with recent activity spikes
- **Recency Score** - Visual indicator showing how recent your data is (0-100 scale)
- **Smart Ranking** - Hotspots ranked by recency-weighted activity, not just raw counts

### Hotspot Trend Detection
- **Trend Analysis** - Each hotspot classified as Growing, Shrinking, Stable, or New
- **Historical Comparison** - Compares first-half vs second-half incident counts to detect momentum
- **Visual Indicators** - Clear trend labels (📈 Growing, 📉 Shrinking, ➡️ Stable, 🆕 New)
- **LLM Context** - Trend data fed to AI for smarter, momentum-aware predictions

### Seasonal Patterns
- **Monthly Trends** - Identifies which months have the highest crime activity
- **Seasonal Breakdown** - Compares crime rates across Winter, Spring, Summer, and Fall
- **Holiday Effect Detection** - Analyzes if crime increases or decreases near major US holidays
- **Payday Pattern Analysis** - Detects if crimes spike around 1st/15th of month (payday effect)
- **Year-over-Year Trends** - Tracks if crime is increasing or decreasing over time

### DBSCAN Clustering
- **Density-Based Hotspots** - Uses DBSCAN algorithm to detect natural crime clusters
- **Irregular Shapes** - Identifies hotspots of any shape, not just grid squares
- **Cluster Radius** - Each hotspot reports its geographic extent in kilometers
- **Noise Filtering** - Automatically excludes isolated incidents as noise
- **Haversine Distance** - Uses accurate geographic distance calculations

### Interactive Visualization
- **Embedded Map View** - Interactive Folium map displayed directly in the UI
- **Heatmap Layer** - Crime density visualization with gradient coloring
- **Risk-Coded Markers** - Predictions colored by likelihood (red=high, orange=medium, green=low)
- **Marker Clustering** - Groups nearby markers for better performance
- **Legend** - Clear identification of actual vs. predicted crimes

### Analysis Results
- **Statistics Tab** - Crime distribution, temporal patterns, seasonal patterns, recency indicators, and crime-specific timing
- **Predictions Tab** - Scrollable table with location, type, risk levels, and full analysis
- **Full Report Tab** - Complete analysis summary with temporal, seasonal insights, and AI output
- **Export Options** - Download interactive map (HTML) and full report (TXT)

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
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

Or install as a package:
```bash
pip install -e .
```

### Step 4: Set Up Ollama

1. **Download and Install Ollama** from [ollama.ai](https://ollama.ai)

2. **Pull the AI Model:**
   ```bash
   ollama pull ministral-3:3b
   ```

3. **Verify Installation:**
   ```bash
   ollama list
   ```

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Launch the application (opens browser automatically)
python run.py
```

The application will automatically open in your default browser at `http://localhost:7860`.

## Usage

### Web Interface (Recommended)

```bash
python run.py
```

Options:
- `--port 8080` - Use a different port
- `--no-browser` - Don't auto-open browser
- `--cli` - Run in command-line mode

### Command Line Interface

```bash
python run.py --cli
```

This expects `data/sample_crime_data.csv` to exist.

### Data Format

Your crime data file should include these columns:

| Column | Required | Description |
|--------|----------|-------------|
| Latitude | Yes | Geographic latitude (decimal degrees) |
| Longitude | Yes | Geographic longitude (decimal degrees) |
| CrimeType | Yes | Category of crime (e.g., Theft, Assault) |
| Date | Recommended | Date of incident (enables temporal analysis & recency weighting) |
| Time | Recommended | Time of incident (enables time-of-day pattern detection) |
| Address | No | Street address |

> **Tip**: Including Date and Time columns significantly improves prediction accuracy. The system uses temporal patterns (peak hours, day-of-week trends) and recency weighting (recent crimes weighted higher) to generate smarter predictions.

A sample data file is included at `data/sample_crime_data.csv`.

## Project Structure

```
crime-analyst-ai/
├── .github/                     # GitHub templates
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── PULL_REQUEST_TEMPLATE.md
├── data/                        # Sample and user data
│   └── sample_crime_data.csv
├── media/                       # Project images
│   ├── crime_analyst_ai.JPEG
│   └── crime_analyst_ai_v2.png
├── output/                      # Generated files (gitignored)
├── src/
│   └── crime_analyst_ai/        # Main package
│       ├── __init__.py
│       ├── app.py               # Gradio web interface
│       └── core.py              # Analysis engine
├── .gitignore
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml               # Package configuration
├── README.md
├── requirements.txt
└── run.py                       # Application launcher
```

## Configuration

### Change the AI Model

Edit `src/crime_analyst_ai/core.py`:

```python
OLLAMA_MODEL = "your-preferred-model"
```

### Adjust Analysis Parameters

- **DBSCAN Clustering**: Modify `eps_km` (default: 0.5km) and `min_samples` (default: 3) in `detect_hotspots()`
- **Seasonal Patterns**: Adjust `compute_seasonal_patterns()` for holiday proximity and payday analysis
- **Trend Detection**: Customize hotspot trend thresholds in `detect_hotspots()` (growth/shrinkage sensitivity)
- **Crime-Type Patterns**: Adjust `compute_crime_type_patterns()` for per-type temporal analysis
- **LLM Prompt**: Customize `build_analysis_prompt()` in `core.py`
- **Map Styling**: Edit `create_crime_map()` in `core.py`
- **Recency Half-Life**: Change the `half_life_days` parameter in `compute_recency_weights()` (default: 30 days)

### Recency Weighting

The system uses exponential decay to weight recent crimes more heavily:

```
weight = 0.5^(days_ago / half_life_days)
```

| Days Ago | Weight (30-day half-life) |
|----------|---------------------------|
| 0 (today) | 1.00 |
| 30 days | 0.50 |
| 60 days | 0.25 |
| 90 days | 0.125 |

Adjust `half_life_days` in `compute_recency_weights()` to change decay rate.

### DBSCAN Clustering Parameters

The system uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) for hotspot detection:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps_km` | 0.5 | Cluster radius in kilometers |
| `min_samples` | 3 | Minimum incidents to form a cluster |

Adjust these in `detect_hotspots()` to change sensitivity:
- **Smaller `eps_km`**: More, smaller clusters (finer detail)
- **Larger `eps_km`**: Fewer, larger clusters (broader areas)
- **Higher `min_samples`**: Only significant clusters (fewer false positives)

### Seasonal Pattern Detection

The system analyzes several calendar-based patterns:

| Pattern | Detection Method |
|---------|------------------|
| **Holiday Effect** | Compares crimes within 3 days of US holidays vs normal days |
| **Payday Effect** | Compares crimes on 1st/2nd/15th/16th vs other days of month |
| **Seasonal** | Winter (Dec-Feb), Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Nov) |
| **Year-over-Year** | Compares first year vs last year if multi-year data |

Holiday proximity is calculated for: New Year's Day, MLK Day, Presidents Day, Memorial Day, July 4th, Labor Day, Columbus Day, Veterans Day, Thanksgiving, Christmas, and New Year's Eve.

### Hotspot Trend Classification

The system analyzes incident patterns over time to classify each hotspot's momentum:

| Trend | Condition | Meaning |
|-------|-----------|---------|
| 📈 Growing | Second-half incidents > first-half by 20%+ | Crime increasing in this area |
| 📉 Shrinking | Second-half incidents < first-half by 20%+ | Crime decreasing in this area |
| ➡️ Stable | Change within ±20% | Consistent crime levels |
| 🆕 New | All incidents in second half | Recently emerged hotspot |

This trend data is passed to the AI model for momentum-aware predictions.

### UI Customization

Edit the `CUSTOM_CSS` variable in `src/crime_analyst_ai/app.py`.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) first.

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
- **[Scikit-learn](https://scikit-learn.org/)** - DBSCAN clustering algorithm
- **[Gradio](https://gradio.app)** - Web interface framework
- **[Ollama](https://ollama.ai)** - Local AI model runtime
- **[Mistral AI](https://mistral.ai)** - Ministral-3:3b language model

---

<div align="center">
  <strong>Built with ❤️ for public safety</strong>
</div>
