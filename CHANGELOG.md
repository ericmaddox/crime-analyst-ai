# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-16

### Added
- Enterprise-grade Gradio web interface with dark theme
- Interactive map visualization with Folium
- AI-powered crime predictions using Ollama (ministral-3:3b)
- Statistical preprocessing and hotspot detection
- Drag-and-drop file upload (CSV/Excel)
- Smart column auto-detection
- Real-time progress indicators
- Embedded map display in UI
- Full analysis report view
- Export functionality for maps and reports
- Sample crime data for testing
- Comprehensive documentation

### Changed
- Reorganized project structure to follow GitHub standards
- Moved source code to `src/crime_analyst_ai/` package
- Updated file paths to use `data/` and `output/` directories

### Fixed
- Ollama subprocess now uses stdin for large prompts
- Dynamic map centering based on actual data
- Robust JSON parsing with regex fallback

## [0.1.0] - Initial Release

### Added
- Basic crime data analysis
- Ollama LLM integration
- Folium map generation
- Command-line interface

