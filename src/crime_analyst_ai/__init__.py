"""
Crime Analyst AI
================

An AI-powered predictive crime analysis tool using Ollama LLM.

Features:
- Statistical preprocessing of crime data
- Hotspot detection using geographic clustering
- AI-powered predictions via Ollama (ministral-3:3b)
- Interactive map visualization
- Enterprise-grade Gradio web interface
"""

__version__ = "1.0.0"
__author__ = "Eric Maddox"
__email__ = "eric.maddox@outlook.com"

from .core import (
    OLLAMA_MODEL,
    run_ollama_predictive_model,
    read_crime_data,
    validate_columns,
    compute_crime_statistics,
    detect_hotspots,
    build_analysis_prompt,
    parse_llm_response,
    validate_predictions,
    create_crime_map,
    save_analysis_report,
    run_analysis,
)

__all__ = [
    "__version__",
    "OLLAMA_MODEL",
    "run_ollama_predictive_model",
    "read_crime_data",
    "validate_columns",
    "compute_crime_statistics",
    "detect_hotspots",
    "build_analysis_prompt",
    "parse_llm_response",
    "validate_predictions",
    "create_crime_map",
    "save_analysis_report",
    "run_analysis",
]

