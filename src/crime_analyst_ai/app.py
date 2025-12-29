"""
Crime Analyst AI - Enterprise Gradio UI
Professional interface for predictive crime analysis
"""

import gradio as gr
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List
import logging

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.crime_analyst_ai.core import (
    read_crime_data,
    validate_columns,
    compute_crime_statistics,
    compute_temporal_patterns,
    compute_crime_type_patterns,
    detect_hotspots,
    build_analysis_prompt,
    run_ollama_predictive_model,
    parse_llm_response,
    validate_predictions,
    create_crime_map,
    save_analysis_report,
    run_analysis,
    OLLAMA_MODEL,
    OUTPUT_DIR
)

# Configure logging for UI
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enterprise Dark Theme CSS
CUSTOM_CSS = """
/* Root variables for consistent theming */
:root {
    --bg-primary: #0f1419;
    --bg-secondary: #1a1f2e;
    --bg-tertiary: #252d3d;
    --bg-elevated: #2a3441;
    --accent-primary: #3b82f6;
    --accent-secondary: #60a5fa;
    --accent-success: #10b981;
    --accent-warning: #f59e0b;
    --accent-danger: #ef4444;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border-color: #334155;
    --shadow-color: rgba(0, 0, 0, 0.3);
}

/* Main container styling */
.gradio-container {
    background: linear-gradient(135deg, var(--bg-primary) 0%, #131820 100%) !important;
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    min-height: 100vh;
}

/* Header styling */
.header-container {
    background: linear-gradient(90deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
    border-bottom: 1px solid var(--border-color);
    padding: 1.5rem 2rem;
    margin: -1rem -1rem 1.5rem -1rem;
    border-radius: 12px 12px 0 0;
}

.header-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.header-subtitle {
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-top: 0.25rem;
}

.header-badge {
    background: linear-gradient(135deg, var(--accent-primary) 0%, #1d4ed8 100%);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Panel styling */
.panel {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    box-shadow: 0 4px 20px var(--shadow-color) !important;
}

.panel-header {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-color);
}

/* Input components */
.gr-input, .gr-dropdown, .gr-file {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.gr-input:focus, .gr-dropdown:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
}

/* File upload area */
.upload-area {
    border: 2px dashed var(--border-color) !important;
    border-radius: 12px !important;
    background: var(--bg-tertiary) !important;
    transition: all 0.3s ease !important;
}

.upload-area:hover {
    border-color: var(--accent-primary) !important;
    background: rgba(59, 130, 246, 0.05) !important;
}

/* Button styling */
.primary-btn {
    background: linear-gradient(135deg, var(--accent-primary) 0%, #2563eb 100%) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4) !important;
}

.secondary-btn {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
}

/* Data preview table */
.dataframe {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

.dataframe th {
    background: var(--bg-elevated) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.5px !important;
    padding: 0.75rem !important;
}

.dataframe td {
    color: var(--text-primary) !important;
    padding: 0.625rem 0.75rem !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.dataframe tr:hover td {
    background: rgba(59, 130, 246, 0.05) !important;
}

/* Results section */
.results-container {
    background: var(--bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--border-color);
    overflow: hidden;
}

/* Stats cards */
.stat-card {
    background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-elevated) 100%);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}

.stat-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--accent-primary);
}

.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
}

/* Map container */
.map-container {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border-color);
    box-shadow: 0 4px 20px var(--shadow-color);
}

/* Predictions table */
.predictions-table {
    background: var(--bg-tertiary);
    border-radius: 10px;
    overflow: hidden;
}

/* Risk badges */
.risk-high {
    background: linear-gradient(135deg, var(--accent-danger) 0%, #dc2626 100%);
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.risk-medium {
    background: linear-gradient(135deg, var(--accent-warning) 0%, #d97706 100%);
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

.risk-low {
    background: linear-gradient(135deg, var(--accent-success) 0%, #059669 100%);
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
}

/* Status indicators */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    font-size: 0.875rem;
}

.status-success {
    background: rgba(16, 185, 129, 0.15);
    color: var(--accent-success);
    border: 1px solid var(--accent-success);
}

.status-processing {
    background: rgba(59, 130, 246, 0.15);
    color: var(--accent-secondary);
    border: 1px solid var(--accent-primary);
}

.status-error {
    background: rgba(239, 68, 68, 0.15);
    color: var(--accent-danger);
    border: 1px solid var(--accent-danger);
}

/* Tabs styling */
.tabs {
    background: var(--bg-secondary) !important;
    border-radius: 10px !important;
}

.tab-nav {
    background: var(--bg-tertiary) !important;
    border-bottom: 1px solid var(--border-color) !important;
}

.tab-nav button {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}

.tab-nav button.selected {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-primary) !important;
}

/* Label styling */
label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
}

/* Accordion styling */
.accordion {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}

/* Progress bar */
.progress-bar {
    background: var(--bg-tertiary) !important;
    border-radius: 20px !important;
    overflow: hidden !important;
}

.progress-bar-fill {
    background: linear-gradient(90deg, var(--accent-primary) 0%, var(--accent-secondary) 100%) !important;
    transition: width 0.5s ease !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-primary);
}

::-webkit-scrollbar-thumb {
    background: var(--bg-elevated);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--border-color);
}

/* Markdown content */
.markdown-content {
    color: var(--text-primary) !important;
}

.markdown-content h1, .markdown-content h2, .markdown-content h3 {
    color: var(--text-primary) !important;
}

.markdown-content code {
    background: var(--bg-tertiary) !important;
    color: var(--accent-secondary) !important;
    padding: 0.125rem 0.375rem !important;
    border-radius: 4px !important;
}

/* Footer */
.footer {
    text-align: center;
    padding: 1rem;
    color: var(--text-muted);
    font-size: 0.8rem;
    border-top: 1px solid var(--border-color);
    margin-top: 2rem;
}
"""


def get_column_choices(df: Optional[pd.DataFrame]) -> List[str]:
    """Get column names from dataframe for dropdown choices."""
    if df is None or df.empty:
        return []
    return list(df.columns)


def load_file(file) -> Tuple[Optional[pd.DataFrame], str, str]:
    """
    Load and preview uploaded file.
    
    Returns:
        Tuple of (dataframe, preview_html, status_message)
    """
    if file is None:
        return None, "", "⬆️ Upload a CSV or Excel file to begin"
    
    try:
        df = read_crime_data(file.name)
        
        # Create preview HTML
        preview_rows = min(10, len(df))
        preview_html = f"""
        <div style="padding: 1rem;">
            <div style="display: flex; gap: 2rem; margin-bottom: 1rem;">
                <div class="stat-card" style="flex: 1;">
                    <div class="stat-value">{len(df):,}</div>
                    <div class="stat-label">Total Records</div>
                </div>
                <div class="stat-card" style="flex: 1;">
                    <div class="stat-value">{len(df.columns)}</div>
                    <div class="stat-label">Columns</div>
                </div>
            </div>
        </div>
        """
        
        status = f"✅ Loaded {len(df):,} records with {len(df.columns)} columns"
        
        return df, preview_html, status
        
    except Exception as e:
        return None, "", f"❌ Error loading file: {str(e)}"


def update_column_dropdowns(df: Optional[pd.DataFrame]):
    """Update column dropdown choices when file is loaded."""
    if df is None:
        return (
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[]),
            gr.update(choices=[])
        )
    
    columns = list(df.columns)
    columns_with_none = ["(None)"] + columns
    
    # Try to auto-detect columns
    lat_default = None
    lon_default = None
    type_default = None
    date_default = "(None)"
    time_default = "(None)"
    
    for col in columns:
        col_lower = col.lower()
        if 'lat' in col_lower and lat_default is None:
            lat_default = col
        if 'lon' in col_lower and lon_default is None:
            lon_default = col
        if any(x in col_lower for x in ['type', 'crime', 'offense', 'category']) and type_default is None:
            type_default = col
        if any(x in col_lower for x in ['date', 'occurred', 'reported']) and date_default == "(None)":
            date_default = col
        if any(x in col_lower for x in ['time', 'hour']) and 'datetime' not in col_lower and time_default == "(None)":
            time_default = col
    
    return (
        gr.update(choices=columns, value=lat_default),
        gr.update(choices=columns, value=lon_default),
        gr.update(choices=columns, value=type_default),
        gr.update(choices=columns_with_none, value=date_default),
        gr.update(choices=columns_with_none, value=time_default)
    )


def run_crime_analysis(
    df: Optional[pd.DataFrame],
    lat_col: str,
    lon_col: str,
    type_col: str,
    date_col: Optional[str] = None,
    time_col: Optional[str] = None,
    progress=gr.Progress()
) -> Tuple[str, str, str, str, str, Optional[str], Optional[str]]:
    """
    Run the crime analysis pipeline with optional temporal analysis.
    
    Returns:
        Tuple of (status, stats_html, predictions_html, map_iframe, report_html, map_file, report_file)
    """
    if df is None:
        return "❌ Please upload a data file first", "", "", "", "", None, None
    
    if not all([lat_col, lon_col, type_col]):
        return "❌ Please select all required columns", "", "", "", "", None, None
    
    # Handle "(None)" selection for optional columns
    date_col_actual = None if date_col in [None, "(None)", ""] else date_col
    time_col_actual = None if time_col in [None, "(None)", ""] else time_col
    
    try:
        progress(0.1, desc="Validating data...")
        df_validated = validate_columns(
            df.copy(), lat_col, lon_col, type_col,
            date_col=date_col_actual,
            time_col=time_col_actual
        )
        
        progress(0.2, desc="Computing statistics...")
        stats = compute_crime_statistics(df_validated)
        
        progress(0.25, desc="Analyzing temporal patterns...")
        temporal = compute_temporal_patterns(df_validated)
        
        progress(0.28, desc="Analyzing crime-type patterns...")
        crime_patterns = compute_crime_type_patterns(df_validated)
        
        progress(0.3, desc="Detecting hotspots...")
        hotspots = detect_hotspots(df_validated)
        
        progress(0.4, desc="Building analysis prompt...")
        prompt = build_analysis_prompt(stats, hotspots, temporal, crime_patterns)
        
        progress(0.5, desc=f"Querying {OLLAMA_MODEL}...")
        llm_output = run_ollama_predictive_model(prompt)
        
        progress(0.7, desc="Parsing predictions...")
        insights = parse_llm_response(llm_output)
        
        if not insights:
            # Use hotspot fallback
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
        
        insights = validate_predictions(insights, stats)
        
        progress(0.8, desc="Generating map...")
        map_file = create_crime_map(df_validated, insights, stats)
        
        progress(0.9, desc="Creating report...")
        report_file = save_analysis_report(llm_output, stats, insights)
        
        # Generate stats HTML (now includes temporal data and crime patterns)
        stats_html = generate_stats_html(stats, temporal, crime_patterns)
        
        # Generate predictions HTML
        predictions_html = generate_predictions_html(insights)
        
        # Create iframe for map display (using file:// protocol or data URI)
        import base64
        with open(map_file, 'r') as f:
            map_content = f.read()
        
        # Encode map HTML as base64 for iframe src
        map_base64 = base64.b64encode(map_content.encode('utf-8')).decode('utf-8')
        map_iframe = f'''
        <div style="border-radius: 10px; overflow: hidden; border: 1px solid var(--border-color);">
            <iframe 
                src="data:text/html;base64,{map_base64}" 
                width="100%" 
                height="550" 
                style="border: none; border-radius: 10px;"
                title="Crime Analysis Map">
            </iframe>
        </div>
        '''
        
        # Generate report HTML for display
        report_html = generate_report_html(llm_output, stats, insights, temporal)
        
        progress(1.0, desc="Complete!")
        
        # Build status message with temporal info
        temporal_note = ""
        if temporal and temporal.get('has_temporal_data'):
            temporal_note = " (with temporal analysis)"
        
        status = f"""
        <div class="status-indicator status-success">
            ✓ Analysis Complete - {len(insights)} predictions generated{temporal_note}
        </div>
        """
        
        return status, stats_html, predictions_html, map_iframe, report_html, map_file, report_file
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        error_status = f"""
        <div class="status-indicator status-error">
            ✗ Analysis Failed: {str(e)}
        </div>
        """
        return error_status, "", "", "", "", None, None


def generate_report_html(llm_output: str, stats: dict, insights: list, temporal: Optional[dict] = None) -> str:
    """Generate HTML for the analysis report display."""
    bounds = stats['geographic_bounds']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    html = f'''
    <div style="padding: 1.5rem; background: var(--bg-tertiary); border-radius: 10px; max-height: 500px; overflow-y: auto;">
        <div style="margin-bottom: 1.5rem;">
            <h3 style="color: var(--text-primary); margin: 0 0 0.5rem 0; font-size: 1.1rem;">
                📋 Analysis Summary
            </h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
                <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Total Records</div>
                    <div style="color: var(--accent-primary); font-size: 1.5rem; font-weight: 700;">{stats['total_records']:,}</div>
                </div>
                <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Crime Types</div>
                    <div style="color: var(--accent-primary); font-size: 1.5rem; font-weight: 700;">{len(stats['crime_distribution'])}</div>
                </div>
                <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Predictions Made</div>
                    <div style="color: var(--accent-success); font-size: 1.5rem; font-weight: 700;">{len(insights)}</div>
                </div>
                <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Geographic Center</div>
                    <div style="color: var(--text-primary); font-size: 0.9rem; font-weight: 500;">{bounds['center_lat']:.4f}, {bounds['center_lon']:.4f}</div>
                </div>
            </div>
        </div>
    '''
    
    # Add temporal patterns section if available
    if temporal and temporal.get('has_temporal_data'):
        html += '''
        <div style="margin-bottom: 1.5rem;">
            <h3 style="color: var(--text-primary); margin: 0 0 0.75rem 0; font-size: 1.1rem;">
                ⏰ Temporal Analysis
            </h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
        '''
        
        if temporal.get('peak_hours'):
            peak_hours_str = ', '.join([f"{h}:00" for h in temporal['peak_hours'][:3]])
            html += f'''
                <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Peak Hours</div>
                    <div style="color: var(--accent-warning); font-size: 1rem; font-weight: 600;">{peak_hours_str}</div>
                </div>
            '''
        
        if temporal.get('peak_days'):
            peak_day_names = [day_names[d] for d in temporal['peak_days'][:3] if 0 <= d <= 6]
            html += f'''
                <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Peak Days</div>
                    <div style="color: var(--accent-warning); font-size: 1rem; font-weight: 600;">{', '.join(peak_day_names)}</div>
                </div>
            '''
        
        if temporal.get('recent_trend'):
            trend = temporal['recent_trend']
            trend_icon = "↑" if trend == "rising" else ("↓" if trend == "falling" else "→")
            change = temporal.get('trend_change_pct', 0)
            html += f'''
                <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Trend</div>
                    <div style="color: var(--text-primary); font-size: 1rem; font-weight: 600;">{trend_icon} {trend.capitalize()} ({change:+.1f}%)</div>
                </div>
            '''
        
        if temporal.get('dominant_time_period'):
            html += f'''
                <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                    <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase;">Most Active</div>
                    <div style="color: var(--text-primary); font-size: 1rem; font-weight: 600;">{temporal['dominant_time_period'].capitalize()}</div>
                </div>
            '''
        
        html += '''
            </div>
        </div>
        '''
    
    html += '''
        <div style="margin-bottom: 1.5rem;">
            <h3 style="color: var(--text-primary); margin: 0 0 0.75rem 0; font-size: 1.1rem;">
                📊 Crime Type Breakdown
            </h3>
            <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
    '''
    
    for crime_info in stats['top_crime_types'][:5]:
        html += f'''
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid var(--border-color);">
                    <span style="color: var(--text-secondary);">{crime_info['type']}</span>
                    <span style="color: var(--text-primary); font-weight: 600;">{crime_info['count']} ({crime_info['percentage']}%)</span>
                </div>
        '''
    
    html += '''
            </div>
        </div>
        
        <div style="margin-bottom: 1.5rem;">
            <h3 style="color: var(--text-primary); margin: 0 0 0.75rem 0; font-size: 1.1rem;">
                🤖 AI Analysis Output
            </h3>
            <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem; white-space: pre-wrap; color: var(--text-secondary); max-height: 200px; overflow-y: auto;">
    '''
    
    # Escape HTML in LLM output and add it
    import html as html_module
    escaped_output = html_module.escape(llm_output[:2000])
    if len(llm_output) > 2000:
        escaped_output += "\n\n... [truncated - download full report for complete output]"
    
    html += escaped_output
    
    html += '''
            </div>
        </div>
    </div>
    '''
    
    return html


def generate_stats_html(stats: dict, temporal: Optional[dict] = None, crime_patterns: Optional[dict] = None) -> str:
    """Generate HTML for statistics display including temporal and crime-type patterns."""
    bounds = stats['geographic_bounds']
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    html = f"""
    <div style="padding: 1rem;">
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem;">
            <div class="stat-card">
                <div class="stat-value">{stats['total_records']:,}</div>
                <div class="stat-label">Total Records</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(stats['crime_distribution'])}</div>
                <div class="stat-label">Crime Types</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{bounds['center_lat']:.2f}°</div>
                <div class="stat-label">Center Lat</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{bounds['center_lon']:.2f}°</div>
                <div class="stat-label">Center Lon</div>
            </div>
        </div>
        
        <div class="panel-header">Crime Type Distribution</div>
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
    """
    
    for crime_info in stats['top_crime_types'][:5]:
        percentage = crime_info['percentage']
        html += f"""
            <div style="display: flex; align-items: center; gap: 1rem;">
                <div style="width: 120px; color: var(--text-secondary); font-size: 0.875rem;">
                    {crime_info['type'][:15]}
                </div>
                <div style="flex: 1; background: var(--bg-elevated); border-radius: 4px; height: 24px; overflow: hidden;">
                    <div style="width: {percentage}%; height: 100%; background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));"></div>
                </div>
                <div style="width: 80px; text-align: right; color: var(--text-primary); font-weight: 600;">
                    {crime_info['count']} ({percentage}%)
                </div>
            </div>
        """
    
    html += "</div>"
    
    # Add recency section if date data available
    recency = stats.get('recency', {})
    if recency.get('has_date_data'):
        html += """
        <div class="panel-header" style="margin-top: 1.5rem;">Data Recency</div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
        """
        
        # Date range
        html += f"""
        <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
            <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;">Date Range</div>
            <div style="color: var(--text-primary); font-size: 0.9rem; font-weight: 500;">{recency.get('oldest_date', 'N/A')} to {recency.get('newest_date', 'N/A')}</div>
        </div>
        """
        
        # Last 30 days
        last_30 = recency.get('last_30_days', 0)
        recent_pct = recency.get('recent_activity_pct', 0)
        html += f"""
        <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
            <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;">Last 30 Days</div>
            <div style="color: var(--accent-warning); font-size: 1.1rem; font-weight: 600;">{last_30} incidents ({recent_pct}%)</div>
        </div>
        """
        
        # Recency score with visual indicator
        score = recency.get('recency_score')
        if score is not None:
            if score >= 70:
                score_color = "var(--accent-danger)"
                score_label = "HIGH"
            elif score >= 40:
                score_color = "var(--accent-warning)"
                score_label = "MEDIUM"
            else:
                score_color = "var(--accent-success)"
                score_label = "LOW"
            
            html += f"""
            <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px; grid-column: span 2;">
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;">Recency Score</div>
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="flex: 1; background: var(--bg-tertiary); border-radius: 4px; height: 24px; overflow: hidden;">
                        <div style="width: {score}%; height: 100%; background: linear-gradient(90deg, {score_color}, {score_color});"></div>
                    </div>
                    <div style="color: {score_color}; font-weight: 600; min-width: 100px;">{score_label} ({score}/100)</div>
                </div>
                <div style="color: var(--text-muted); font-size: 0.75rem; margin-top: 0.5rem;">
                    Higher scores mean more recent crime activity. Recent crimes are weighted more heavily in predictions.
                </div>
            </div>
            """
        
        html += "</div>"
    
    # Add temporal patterns section if available
    if temporal and temporal.get('has_temporal_data'):
        html += """
        <div class="panel-header" style="margin-top: 1.5rem;">Temporal Patterns</div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
        """
        
        # Peak hours
        if temporal.get('peak_hours'):
            peak_hours_str = ', '.join([f"{h}:00" for h in temporal['peak_hours'][:3]])
            html += f"""
            <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;">Peak Hours</div>
                <div style="color: var(--accent-primary); font-size: 1.1rem; font-weight: 600;">{peak_hours_str}</div>
            </div>
            """
        
        # Peak days
        if temporal.get('peak_days'):
            peak_day_names = [day_names[d] for d in temporal['peak_days'][:3] if 0 <= d <= 6]
            html += f"""
            <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;">Peak Days</div>
                <div style="color: var(--accent-primary); font-size: 1.1rem; font-weight: 600;">{', '.join(peak_day_names)}</div>
            </div>
            """
        
        # Dominant time period
        if temporal.get('dominant_time_period'):
            html += f"""
            <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;">Most Active Period</div>
                <div style="color: var(--accent-primary); font-size: 1.1rem; font-weight: 600;">{temporal['dominant_time_period'].capitalize()}</div>
            </div>
            """
        
        # Trend
        if temporal.get('recent_trend'):
            trend = temporal['recent_trend']
            trend_color = "var(--accent-danger)" if trend == "rising" else ("var(--accent-success)" if trend == "falling" else "var(--text-primary)")
            trend_icon = "↑" if trend == "rising" else ("↓" if trend == "falling" else "→")
            change = temporal.get('trend_change_pct', 0)
            html += f"""
            <div style="background: var(--bg-elevated); padding: 1rem; border-radius: 8px;">
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;">Recent Trend</div>
                <div style="color: {trend_color}; font-size: 1.1rem; font-weight: 600;">{trend_icon} {trend.capitalize()} ({change:+.1f}%)</div>
            </div>
            """
        
        html += "</div>"
        
        # Time period breakdown
        if temporal.get('time_periods'):
            tp = temporal['time_periods']
            total = sum(tp.values()) or 1
            html += """
            <div style="margin-top: 1rem;">
                <div style="color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.5rem;">Time of Day Breakdown</div>
                <div style="display: flex; gap: 0.5rem; height: 32px; border-radius: 6px; overflow: hidden;">
            """
            
            time_colors = {
                'morning': '#f59e0b',
                'afternoon': '#3b82f6',
                'evening': '#8b5cf6',
                'night': '#1e293b'
            }
            
            for period, count in tp.items():
                pct = (count / total) * 100
                color = time_colors.get(period, '#64748b')
                html += f"""
                    <div style="width: {pct}%; background: {color}; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.7rem; font-weight: 600;" title="{period.capitalize()}: {count} ({pct:.1f}%)">
                        {period.capitalize()}
                    </div>
                """
            
            html += """
                </div>
            </div>
            """
    
    # Add crime-type specific patterns if available
    if crime_patterns:
        html += """
        <div class="panel-header" style="margin-top: 1.5rem;">Crime-Type Timing Patterns</div>
        <div style="display: flex; flex-direction: column; gap: 0.5rem;">
        """
        
        for crime_type, pattern in list(crime_patterns.items())[:5]:
            # Build timing description
            timing_parts = []
            
            if pattern.get('peak_hours'):
                peak_hour = pattern['peak_hours'][0]
                timing_parts.append(f"peaks at {peak_hour}:00")
            
            if pattern.get('peak_period'):
                timing_parts.append(f"{pattern['peak_period']}")
            
            if pattern.get('weekend_vs_weekday') == 'weekend':
                timing_parts.append("weekends")
            elif pattern.get('weekend_vs_weekday') == 'weekday':
                timing_parts.append("weekdays")
            
            timing_str = ' | '.join(timing_parts) if timing_parts else 'No pattern detected'
            
            # Determine color based on period
            period_colors = {
                'morning': '#f59e0b',
                'afternoon': '#3b82f6',
                'evening': '#8b5cf6',
                'night': '#1e293b'
            }
            period = pattern.get('peak_period', 'afternoon')
            color = period_colors.get(period, '#64748b')
            
            html += f"""
            <div style="display: flex; align-items: center; gap: 1rem; background: var(--bg-elevated); padding: 0.75rem; border-radius: 8px; border-left: 4px solid {color};">
                <div style="min-width: 120px; color: var(--text-primary); font-weight: 600;">
                    {crime_type[:18]}
                </div>
                <div style="flex: 1; color: var(--text-secondary); font-size: 0.875rem;">
                    {timing_str}
                </div>
                <div style="color: var(--text-muted); font-size: 0.8rem;">
                    {pattern.get('count', 0)} incidents
                </div>
            </div>
            """
        
        html += "</div>"
    
    html += "</div>"
    
    return html


def generate_predictions_html(insights: list) -> str:
    """Generate HTML for predictions table."""
    html = """
    <div style="padding: 1rem;">
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: var(--bg-elevated);">
                    <th style="padding: 0.75rem; text-align: left; color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase;">#</th>
                    <th style="padding: 0.75rem; text-align: left; color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase;">Location</th>
                    <th style="padding: 0.75rem; text-align: left; color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase;">Crime Type</th>
                    <th style="padding: 0.75rem; text-align: left; color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase;">Risk Level</th>
                    <th style="padding: 0.75rem; text-align: left; color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase;">Analysis</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for i, insight in enumerate(insights, 1):
        likelihood = insight['Likelihood']
        try:
            likelihood_val = float(likelihood.strip('%'))
            if likelihood_val >= 70:
                risk_class = "risk-high"
                risk_label = "HIGH"
            elif likelihood_val >= 40:
                risk_class = "risk-medium"
                risk_label = "MEDIUM"
            else:
                risk_class = "risk-low"
                risk_label = "LOW"
        except:
            risk_class = "risk-medium"
            risk_label = "UNKNOWN"
        
        html += f"""
            <tr style="border-bottom: 1px solid var(--border-color);">
                <td style="padding: 0.75rem; color: var(--text-muted);">{i}</td>
                <td style="padding: 0.75rem; color: var(--text-primary); font-family: monospace;">
                    {insight['Latitude']:.4f}, {insight['Longitude']:.4f}
                </td>
                <td style="padding: 0.75rem; color: var(--text-primary); font-weight: 500;">
                    {insight['CrimeType']}
                </td>
                <td style="padding: 0.75rem;">
                    <span class="{risk_class}">{risk_label} ({likelihood})</span>
                </td>
                <td style="padding: 0.75rem; color: var(--text-secondary); font-size: 0.875rem; max-width: 300px;">
                    <div style="max-height: 80px; overflow-y: auto; padding-right: 0.5rem;">
                        {insight['Prediction']}
                    </div>
                </td>
            </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html


def create_app():
    """Create and configure the Gradio application."""
    
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Base(), title="Crime Analyst AI") as app:
        
        # State for storing loaded dataframe
        df_state = gr.State(None)
        
        # Header
        gr.HTML("""
            <div class="header-container">
                <div class="header-title">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M2 17L12 22L22 17" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                        <path d="M2 12L12 17L22 12" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    CRIME ANALYST AI
                    <span class="header-badge">Powered by Ollama</span>
                </div>
                <div class="header-subtitle">
                    Advanced predictive crime analysis using AI-powered pattern recognition
                </div>
            </div>
        """)
        
        with gr.Row():
            # Left Panel - Data Import
            with gr.Column(scale=1):
                gr.HTML('<div class="panel-header">📁 DATA IMPORT</div>')
                
                file_upload = gr.File(
                    label="Upload Crime Data",
                    file_types=[".csv", ".xlsx", ".xls"],
                    type="filepath",
                    elem_classes=["upload-area"]
                )
                
                status_display = gr.HTML(
                    value="<div style='color: var(--text-muted); padding: 1rem; text-align: center;'>⬆️ Upload a CSV or Excel file to begin</div>"
                )
                
                preview_display = gr.HTML()
                
                gr.HTML('<div class="panel-header" style="margin-top: 1.5rem;">🔧 COLUMN MAPPING</div>')
                
                with gr.Group():
                    lat_dropdown = gr.Dropdown(
                        label="Latitude Column",
                        choices=[],
                        interactive=True
                    )
                    lon_dropdown = gr.Dropdown(
                        label="Longitude Column",
                        choices=[],
                        interactive=True
                    )
                    type_dropdown = gr.Dropdown(
                        label="Crime Type Column",
                        choices=[],
                        interactive=True
                    )
                
                gr.HTML('<div class="panel-header" style="margin-top: 1.5rem;">⏰ TEMPORAL COLUMNS (Optional)</div>')
                
                with gr.Group():
                    date_dropdown = gr.Dropdown(
                        label="Date Column",
                        choices=[],
                        interactive=True,
                        info="Enable temporal pattern analysis"
                    )
                    time_dropdown = gr.Dropdown(
                        label="Time Column",
                        choices=[],
                        interactive=True,
                        info="Enable time-of-day analysis"
                    )
                
                analyze_btn = gr.Button(
                    "🔍 Run Predictive Analysis",
                    variant="primary",
                    elem_classes=["primary-btn"],
                    size="lg"
                )
                
                analysis_status = gr.HTML()
            
            # Right Panel - Results
            with gr.Column(scale=2):
                gr.HTML('<div class="panel-header">📊 ANALYSIS RESULTS</div>')
                
                with gr.Tabs():
                    with gr.TabItem("🗺️ Interactive Map"):
                        map_display = gr.HTML(
                            value="""
                            <div style="height: 550px; display: flex; align-items: center; justify-content: center; 
                                        background: var(--bg-tertiary); border-radius: 10px; color: var(--text-muted);">
                                <div style="text-align: center;">
                                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" style="margin-bottom: 1rem; opacity: 0.5;">
                                        <path d="M21 10C21 17 12 23 12 23C12 23 3 17 3 10C3 5.02944 7.02944 1 12 1C16.9706 1 21 5.02944 21 10Z" stroke="currentColor" stroke-width="2"/>
                                        <circle cx="12" cy="10" r="3" stroke="currentColor" stroke-width="2"/>
                                    </svg>
                                    <p>Map will appear here after analysis</p>
                                </div>
                            </div>
                            """
                        )
                    
                    with gr.TabItem("Statistics"):
                        stats_display = gr.HTML(
                            value="""
                            <div style="padding: 2rem; text-align: center; color: var(--text-muted);">
                                Statistics will appear here after analysis
                            </div>
                            """
                        )
                    
                    with gr.TabItem("Predictions"):
                        predictions_display = gr.HTML(
                            value="""
                            <div style="padding: 2rem; text-align: center; color: var(--text-muted);">
                                Predictions will appear here after analysis
                            </div>
                            """
                        )
                    
                    with gr.TabItem("Full Report"):
                        report_display = gr.HTML(
                            value="""
                            <div style="padding: 2rem; text-align: center; color: var(--text-muted);">
                                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" style="margin-bottom: 1rem; opacity: 0.5;">
                                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" stroke-width="2"/>
                                    <polyline points="14,2 14,8 20,8" stroke="currentColor" stroke-width="2"/>
                                    <line x1="16" y1="13" x2="8" y2="13" stroke="currentColor" stroke-width="2"/>
                                    <line x1="16" y1="17" x2="8" y2="17" stroke="currentColor" stroke-width="2"/>
                                    <polyline points="10,9 9,9 8,9" stroke="currentColor" stroke-width="2"/>
                                </svg>
                                <p>Full analysis report will appear here after analysis</p>
                            </div>
                            """
                        )
                
                gr.HTML('<div class="panel-header" style="margin-top: 1.5rem;">📥 EXPORT FILES</div>')
                
                with gr.Row():
                    map_download = gr.File(label="Download Map (HTML)")
                    report_download = gr.File(label="Download Report (TXT)")
        
        # Footer
        gr.HTML("""
            <div class="footer">
                Crime Analyst AI • Powered by Ollama (ministral-3:3b) • Enterprise Analytics Platform
            </div>
        """)
        
        # Event handlers
        file_upload.change(
            fn=load_file,
            inputs=[file_upload],
            outputs=[df_state, preview_display, status_display]
        ).then(
            fn=update_column_dropdowns,
            inputs=[df_state],
            outputs=[lat_dropdown, lon_dropdown, type_dropdown, date_dropdown, time_dropdown]
        )
        
        analyze_btn.click(
            fn=run_crime_analysis,
            inputs=[df_state, lat_dropdown, lon_dropdown, type_dropdown, date_dropdown, time_dropdown],
            outputs=[analysis_status, stats_display, predictions_display, map_display, report_display, map_download, report_download]
        )
    
    return app


def main():
    """Main entry point for launching the application."""
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True
    )


if __name__ == "__main__":
    main()

