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
| Date | No | Date of incident |
| Time | No | Time of incident |
| Address | No | Street address |

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

- **Hotspot Detection**: Modify `detect_hotspots()` in `core.py`
- **LLM Prompt**: Customize `build_analysis_prompt()` in `core.py`
- **Map Styling**: Edit `create_crime_map()` in `core.py`

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
- **[Gradio](https://gradio.app)** - Web interface framework
- **[Ollama](https://ollama.ai)** - Local AI model runtime
- **[Mistral AI](https://mistral.ai)** - Ministral-3:3b language model

---

<div align="center">
  <strong>Built with ❤️ for public safety</strong>
</div>
