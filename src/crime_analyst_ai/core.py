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
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
from prophet import Prophet

# US Federal Holidays (month, day) - used for holiday proximity analysis
US_HOLIDAYS = [
    (1, 1),    # New Year's Day
    (1, 15),   # MLK Day (approximate - 3rd Monday)
    (2, 19),   # Presidents Day (approximate - 3rd Monday)
    (5, 27),   # Memorial Day (approximate - last Monday)
    (7, 4),    # Independence Day
    (9, 2),    # Labor Day (approximate - 1st Monday)
    (10, 14),  # Columbus Day (approximate - 2nd Monday)
    (11, 11),  # Veterans Day
    (11, 28),  # Thanksgiving (approximate - 4th Thursday)
    (12, 25),  # Christmas
    (12, 31),  # New Year's Eve
]

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


def validate_columns(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    type_col: str,
    date_col: Optional[str] = None,
    time_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Validate and normalize required columns in the DataFrame.
    
    Args:
        df: The crime data DataFrame
        lat_col: Name of the latitude column
        lon_col: Name of the longitude column  
        type_col: Name of the crime type column
        date_col: Optional name of the date column for temporal analysis
        time_col: Optional name of the time column for temporal analysis
        
    Returns:
        DataFrame with standardized column names and temporal columns if provided
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
    
    # Parse temporal columns if provided
    if date_col and date_col in df.columns:
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['Month'] = df['Date'].dt.month
        valid_dates = df['Date'].notna().sum()
        logging.info(f"Parsed {valid_dates} valid dates from '{date_col}'")
    
    if time_col and time_col in df.columns:
        # Try multiple time formats
        time_parsed = pd.to_datetime(df[time_col], format='%H:%M', errors='coerce')
        if time_parsed.isna().all():
            time_parsed = pd.to_datetime(df[time_col], format='%H:%M:%S', errors='coerce')
        if time_parsed.isna().all():
            time_parsed = pd.to_datetime(df[time_col], errors='coerce')
        df['Hour'] = time_parsed.dt.hour
        valid_times = df['Hour'].notna().sum()
        logging.info(f"Parsed {valid_times} valid times from '{time_col}'")
    
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
        'top_crime_types': [],
        'recency': {}
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
    
    # Compute recency statistics
    stats['recency'] = compute_recency_stats(df)
    
    logging.info(f"Computed statistics: {stats['total_records']} records, {len(crime_counts)} crime types")
    return stats


def compute_temporal_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze temporal crime patterns from the data.
    
    Args:
        df: DataFrame with optional Date, DayOfWeek, Month, Hour columns
        
    Returns:
        Dictionary containing temporal patterns and insights
    """
    patterns = {
        'hourly_distribution': {},      # Crimes by hour (0-23)
        'daily_distribution': {},       # Crimes by day of week
        'monthly_distribution': {},     # Crimes by month
        'peak_hours': [],               # Top 3 hours
        'peak_days': [],                # Top 3 days
        'time_periods': {},             # Morning/Afternoon/Evening/Night breakdown
        'recent_trend': None,           # Rising/Falling/Stable
        'has_temporal_data': False      # Whether temporal data was available
    }
    
    # Analyze hourly patterns
    if 'Hour' in df.columns and df['Hour'].notna().any():
        patterns['has_temporal_data'] = True
        hourly = df['Hour'].value_counts().sort_index()
        patterns['hourly_distribution'] = {int(k): int(v) for k, v in hourly.items()}
        patterns['peak_hours'] = [int(h) for h in hourly.nlargest(3).index.tolist()]
        
        # Time period breakdown
        morning = len(df[(df['Hour'] >= 6) & (df['Hour'] < 12)])
        afternoon = len(df[(df['Hour'] >= 12) & (df['Hour'] < 18)])
        evening = len(df[(df['Hour'] >= 18) & (df['Hour'] < 22)])
        night = len(df[(df['Hour'] >= 22) | (df['Hour'] < 6)])
        
        patterns['time_periods'] = {
            'morning': morning,
            'afternoon': afternoon,
            'evening': evening,
            'night': night
        }
        
        # Determine dominant time period
        time_max = max(patterns['time_periods'], key=patterns['time_periods'].get)
        patterns['dominant_time_period'] = time_max
    
    # Analyze day of week patterns
    if 'DayOfWeek' in df.columns and df['DayOfWeek'].notna().any():
        patterns['has_temporal_data'] = True
        daily = df['DayOfWeek'].value_counts()
        patterns['daily_distribution'] = {int(k): int(v) for k, v in daily.items()}
        patterns['peak_days'] = [int(d) for d in daily.nlargest(3).index.tolist()]
    
    # Analyze monthly patterns
    if 'Month' in df.columns and df['Month'].notna().any():
        patterns['has_temporal_data'] = True
        monthly = df['Month'].value_counts().sort_index()
        patterns['monthly_distribution'] = {int(k): int(v) for k, v in monthly.items()}
    
    # Trend analysis (compare first half vs second half of data by date)
    if 'Date' in df.columns and df['Date'].notna().any():
        patterns['has_temporal_data'] = True
        sorted_df = df.dropna(subset=['Date']).sort_values('Date')
        
        if len(sorted_df) >= 10:  # Need enough data points for meaningful trend
            midpoint = len(sorted_df) // 2
            first_half_count = midpoint
            second_half_count = len(sorted_df) - midpoint
            
            if first_half_count > 0:
                change_pct = ((second_half_count - first_half_count) / first_half_count) * 100
                if change_pct > 10:
                    patterns['recent_trend'] = 'rising'
                elif change_pct < -10:
                    patterns['recent_trend'] = 'falling'
                else:
                    patterns['recent_trend'] = 'stable'
                patterns['trend_change_pct'] = round(change_pct, 1)
    
    if patterns['has_temporal_data']:
        logging.info(f"Computed temporal patterns: peak hours {patterns['peak_hours']}, peak days {patterns['peak_days']}, trend: {patterns['recent_trend']}")
    else:
        logging.info("No temporal data available for pattern analysis")
    
    return patterns


def compute_crime_type_patterns(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Analyze temporal patterns for each crime type separately.
    Different crimes occur at different times (e.g., burglaries during work hours,
    assaults late at night).
    
    Args:
        df: DataFrame with CrimeType and optional Hour, DayOfWeek columns
        
    Returns:
        Dictionary mapping crime types to their temporal patterns
    """
    crime_patterns = {}
    
    has_hours = 'Hour' in df.columns and df['Hour'].notna().any()
    has_days = 'DayOfWeek' in df.columns and df['DayOfWeek'].notna().any()
    
    if not has_hours and not has_days:
        logging.info("No temporal data available for crime-type pattern analysis")
        return crime_patterns
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Get top crime types (limit to top 5 for performance)
    top_crimes = df['CrimeType'].value_counts().head(5).index.tolist()
    
    for crime_type in top_crimes:
        crime_df = df[df['CrimeType'] == crime_type]
        
        pattern = {
            'count': len(crime_df),
            'peak_hours': [],
            'peak_days': [],
            'peak_period': None,
            'weekend_vs_weekday': None
        }
        
        # Analyze hours for this crime type
        if has_hours:
            crime_hours = crime_df['Hour'].dropna()
            if len(crime_hours) > 0:
                hour_counts = crime_hours.value_counts()
                pattern['peak_hours'] = [int(h) for h in hour_counts.head(2).index.tolist()]
                
                # Determine peak period
                morning = len(crime_df[(crime_df['Hour'] >= 6) & (crime_df['Hour'] < 12)])
                afternoon = len(crime_df[(crime_df['Hour'] >= 12) & (crime_df['Hour'] < 18)])
                evening = len(crime_df[(crime_df['Hour'] >= 18) & (crime_df['Hour'] < 22)])
                night = len(crime_df[(crime_df['Hour'] >= 22) | (crime_df['Hour'] < 6)])
                
                periods = {'morning': morning, 'afternoon': afternoon, 'evening': evening, 'night': night}
                pattern['peak_period'] = max(periods, key=periods.get)
        
        # Analyze days for this crime type
        if has_days:
            crime_days = crime_df['DayOfWeek'].dropna()
            if len(crime_days) > 0:
                day_counts = crime_days.value_counts()
                pattern['peak_days'] = [int(d) for d in day_counts.head(2).index.tolist()]
                
                # Weekend vs weekday comparison
                weekday_count = len(crime_df[crime_df['DayOfWeek'] < 5])
                weekend_count = len(crime_df[crime_df['DayOfWeek'] >= 5])
                
                # Normalize by number of days (5 weekdays vs 2 weekend days)
                weekday_avg = weekday_count / 5 if weekday_count > 0 else 0
                weekend_avg = weekend_count / 2 if weekend_count > 0 else 0
                
                if weekend_avg > weekday_avg * 1.2:
                    pattern['weekend_vs_weekday'] = 'weekend'
                elif weekday_avg > weekend_avg * 1.2:
                    pattern['weekend_vs_weekday'] = 'weekday'
                else:
                    pattern['weekend_vs_weekday'] = 'balanced'
        
        crime_patterns[str(crime_type)] = pattern
    
    logging.info(f"Computed temporal patterns for {len(crime_patterns)} crime types")
    return crime_patterns


def compute_recency_weights(df: pd.DataFrame, half_life_days: int = 30) -> pd.Series:
    """
    Compute recency weights for each crime record using exponential decay.
    More recent crimes get higher weights.
    
    Args:
        df: DataFrame with 'Date' column
        half_life_days: Number of days for weight to decay by half (default 30)
        
    Returns:
        Series of weights (0 to 1) indexed like the input DataFrame
    """
    if 'Date' not in df.columns or df['Date'].isna().all():
        # No date data - return uniform weights
        return pd.Series(1.0, index=df.index)
    
    # Get the most recent date in the dataset
    valid_dates = df['Date'].dropna()
    if len(valid_dates) == 0:
        return pd.Series(1.0, index=df.index)
    
    max_date = valid_dates.max()
    
    # Calculate days ago for each record
    days_ago = (max_date - df['Date']).dt.days
    
    # Apply exponential decay: weight = 0.5^(days_ago / half_life)
    # This means after half_life_days, the weight is 0.5
    # After 2*half_life_days, the weight is 0.25, etc.
    decay_rate = np.log(2) / half_life_days
    weights = np.exp(-decay_rate * days_ago.fillna(days_ago.max()))
    
    # Normalize to have a max of 1
    weights = weights / weights.max() if weights.max() > 0 else weights
    
    logging.info(f"Computed recency weights: half-life={half_life_days} days, "
                 f"date range={valid_dates.min().date()} to {max_date.date()}")
    
    return weights


def compute_recency_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute recency-related statistics for the crime data.
    
    Args:
        df: DataFrame with 'Date' column
        
    Returns:
        Dictionary containing recency statistics
    """
    recency = {
        'has_date_data': False,
        'date_range_days': None,
        'oldest_date': None,
        'newest_date': None,
        'last_7_days': 0,
        'last_30_days': 0,
        'last_90_days': 0,
        'older_than_90_days': 0,
        'recent_activity_pct': 0,  # % of crimes in last 30 days
        'recency_score': None  # Overall recency indicator
    }
    
    if 'Date' not in df.columns or df['Date'].isna().all():
        return recency
    
    valid_dates = df.dropna(subset=['Date'])
    if len(valid_dates) == 0:
        return recency
    
    recency['has_date_data'] = True
    
    max_date = valid_dates['Date'].max()
    min_date = valid_dates['Date'].min()
    
    recency['oldest_date'] = min_date.strftime('%Y-%m-%d')
    recency['newest_date'] = max_date.strftime('%Y-%m-%d')
    recency['date_range_days'] = (max_date - min_date).days
    
    # Count crimes by recency bucket
    days_ago = (max_date - valid_dates['Date']).dt.days
    
    recency['last_7_days'] = int((days_ago <= 7).sum())
    recency['last_30_days'] = int((days_ago <= 30).sum())
    recency['last_90_days'] = int((days_ago <= 90).sum())
    recency['older_than_90_days'] = int((days_ago > 90).sum())
    
    # Calculate recent activity percentage
    total = len(valid_dates)
    if total > 0:
        recency['recent_activity_pct'] = round((recency['last_30_days'] / total) * 100, 1)
    
    # Compute recency score (weighted average of how recent crimes are)
    # Higher score = more recent activity
    if len(days_ago) > 0 and recency['date_range_days'] and recency['date_range_days'] > 0:
        avg_days_ago = days_ago.mean()
        # Score: 100 if all crimes are today, 0 if all crimes are at oldest date
        recency['recency_score'] = round(100 * (1 - avg_days_ago / recency['date_range_days']), 1)
    
    logging.info(f"Recency stats: {recency['last_30_days']} crimes in last 30 days "
                 f"({recency['recent_activity_pct']}%), score={recency['recency_score']}")
    
    return recency


def compute_seasonal_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze seasonal and calendar-based crime patterns.
    Detects monthly/quarterly trends, holiday effects, and payday patterns.
    
    Args:
        df: DataFrame with 'Date' column (parsed as datetime)
        
    Returns:
        Dictionary containing seasonal patterns and insights
    """
    seasonal = {
        'has_seasonal_data': False,
        'monthly_trend': {},           # Crime counts by month
        'quarterly_trend': {},         # Crime counts by quarter
        'peak_months': [],             # Top 3 months for crime
        'peak_quarter': None,          # Highest crime quarter
        'season_comparison': {},       # Winter/Spring/Summer/Fall breakdown
        'dominant_season': None,       # Season with most crimes
        'holiday_effect': {},          # Crimes near holidays vs normal days
        'holiday_spike': None,         # Whether holidays have more crime
        'payday_effect': {},           # Crimes near 1st/15th vs other days
        'payday_spike': None,          # Whether paydays have more crime
        'year_over_year': None,        # YoY trend if multi-year data
    }
    
    if 'Date' not in df.columns or df['Date'].isna().all():
        return seasonal
    
    valid_df = df.dropna(subset=['Date']).copy()
    if len(valid_df) < 10:
        return seasonal
    
    seasonal['has_seasonal_data'] = True
    
    # Extract date components
    valid_df['Month'] = valid_df['Date'].dt.month
    valid_df['Quarter'] = valid_df['Date'].dt.quarter
    valid_df['DayOfMonth'] = valid_df['Date'].dt.day
    valid_df['Year'] = valid_df['Date'].dt.year
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Monthly trend
    monthly = valid_df['Month'].value_counts().sort_index()
    seasonal['monthly_trend'] = {month_names[int(k)-1]: int(v) for k, v in monthly.items()}
    seasonal['peak_months'] = [month_names[int(m)-1] for m in monthly.nlargest(3).index.tolist()]
    
    # Quarterly trend
    quarterly = valid_df['Quarter'].value_counts().sort_index()
    seasonal['quarterly_trend'] = {f"Q{int(k)}": int(v) for k, v in quarterly.items()}
    if len(quarterly) > 0:
        seasonal['peak_quarter'] = f"Q{int(quarterly.idxmax())}"
    
    # Season comparison (Northern Hemisphere)
    winter = len(valid_df[valid_df['Month'].isin([12, 1, 2])])
    spring = len(valid_df[valid_df['Month'].isin([3, 4, 5])])
    summer = len(valid_df[valid_df['Month'].isin([6, 7, 8])])
    fall = len(valid_df[valid_df['Month'].isin([9, 10, 11])])
    
    seasonal['season_comparison'] = {
        'winter': winter,
        'spring': spring,
        'summer': summer,
        'fall': fall
    }
    seasonal['dominant_season'] = max(seasonal['season_comparison'], 
                                       key=seasonal['season_comparison'].get)
    
    # Holiday proximity analysis
    def days_to_nearest_holiday(date):
        """Calculate days to nearest US holiday in same year."""
        if pd.isna(date):
            return None
        year = date.year
        min_diff = 365
        for month, day in US_HOLIDAYS:
            try:
                holiday = datetime(year, month, day)
                diff = abs((date - holiday).days)
                min_diff = min(min_diff, diff)
            except ValueError:
                continue
        return min_diff
    
    valid_df['days_to_holiday'] = valid_df['Date'].apply(days_to_nearest_holiday)
    
    # Compare crimes within 3 days of holiday vs other days
    near_holiday = len(valid_df[valid_df['days_to_holiday'] <= 3])
    far_from_holiday = len(valid_df[valid_df['days_to_holiday'] > 3])
    
    # Normalize by number of days (roughly 11 holidays * 7 days = 77 "holiday zone" days per year)
    total_days = (valid_df['Date'].max() - valid_df['Date'].min()).days + 1
    holiday_days = min(77, total_days)  # ~77 days within 3 days of a holiday
    non_holiday_days = max(1, total_days - holiday_days)
    
    holiday_rate = near_holiday / holiday_days if holiday_days > 0 else 0
    non_holiday_rate = far_from_holiday / non_holiday_days if non_holiday_days > 0 else 0
    
    seasonal['holiday_effect'] = {
        'near_holiday_count': near_holiday,
        'normal_days_count': far_from_holiday,
        'near_holiday_rate': round(holiday_rate, 2),
        'normal_rate': round(non_holiday_rate, 2)
    }
    
    if non_holiday_rate > 0:
        ratio = holiday_rate / non_holiday_rate
        if ratio > 1.2:
            seasonal['holiday_spike'] = 'higher'
        elif ratio < 0.8:
            seasonal['holiday_spike'] = 'lower'
        else:
            seasonal['holiday_spike'] = 'similar'
    
    # Payday analysis (1st and 15th of month)
    payday_crimes = len(valid_df[valid_df['DayOfMonth'].isin([1, 2, 15, 16])])
    non_payday_crimes = len(valid_df[~valid_df['DayOfMonth'].isin([1, 2, 15, 16])])
    
    # 4 payday days per month vs ~26 non-payday days
    payday_rate = payday_crimes / 4 if payday_crimes > 0 else 0
    non_payday_rate = non_payday_crimes / 26 if non_payday_crimes > 0 else 0
    
    seasonal['payday_effect'] = {
        'payday_count': payday_crimes,
        'non_payday_count': non_payday_crimes,
        'payday_rate': round(payday_rate, 2),
        'non_payday_rate': round(non_payday_rate, 2)
    }
    
    if non_payday_rate > 0:
        ratio = payday_rate / non_payday_rate
        if ratio > 1.2:
            seasonal['payday_spike'] = 'higher'
        elif ratio < 0.8:
            seasonal['payday_spike'] = 'lower'
        else:
            seasonal['payday_spike'] = 'similar'
    
    # Year-over-year trend (if multi-year data)
    years = valid_df['Year'].unique()
    if len(years) >= 2:
        yearly = valid_df['Year'].value_counts().sort_index()
        first_year = yearly.iloc[0]
        last_year = yearly.iloc[-1]
        
        if first_year > 0:
            yoy_change = ((last_year - first_year) / first_year) * 100
            if yoy_change > 10:
                seasonal['year_over_year'] = f"increasing (+{yoy_change:.0f}%)"
            elif yoy_change < -10:
                seasonal['year_over_year'] = f"decreasing ({yoy_change:.0f}%)"
            else:
                seasonal['year_over_year'] = "stable"
    
    logging.info(f"Seasonal patterns: peak months {seasonal['peak_months']}, "
                 f"dominant season: {seasonal['dominant_season']}, "
                 f"holiday effect: {seasonal['holiday_spike']}, "
                 f"payday effect: {seasonal['payday_spike']}")
    
    return seasonal


def compute_crime_forecast(df: pd.DataFrame, periods: int = 14) -> Dict[str, Any]:
    """
    Forecast future crime counts using Facebook Prophet.
    Provides predictions with confidence intervals for the next N days.
    
    Args:
        df: DataFrame with 'Date' column (parsed as datetime)
        periods: Number of days to forecast (default 14)
        
    Returns:
        Dictionary containing forecast data and insights
    """
    forecast_result = {
        'has_forecast': False,
        'model': 'prophet',
        'horizon_days': periods,
        'daily_forecast': [],
        'total_predicted': 0,
        'avg_daily': 0,
        'peak_day': None,
        'peak_count': 0,
        'low_day': None,
        'low_count': 0,
        'trend': None,
        'trend_pct': 0,
        'weekly_pattern': {},
        'by_crime_type': {},
        'confidence_level': 0.80
    }
    
    if 'Date' not in df.columns or df['Date'].isna().all():
        logging.info("No date data available for forecasting")
        return forecast_result
    
    valid_df = df.dropna(subset=['Date']).copy()
    
    # Need at least 14 days of data for meaningful forecast
    date_range = (valid_df['Date'].max() - valid_df['Date'].min()).days
    if len(valid_df) < 14 or date_range < 14:
        logging.info(f"Insufficient data for forecasting: {len(valid_df)} records over {date_range} days")
        return forecast_result
    
    try:
        # Aggregate crimes by day for Prophet
        daily_counts = valid_df.groupby(valid_df['Date'].dt.date).size().reset_index()
        daily_counts.columns = ['ds', 'y']
        daily_counts['ds'] = pd.to_datetime(daily_counts['ds'])
        
        # Fill missing dates with zeros
        date_range_full = pd.date_range(start=daily_counts['ds'].min(), end=daily_counts['ds'].max(), freq='D')
        daily_counts = daily_counts.set_index('ds').reindex(date_range_full, fill_value=0).reset_index()
        daily_counts.columns = ['ds', 'y']
        
        # Need at least 2 data points
        if len(daily_counts) < 2:
            logging.info("Not enough daily data points for forecasting")
            return forecast_result
        
        # Create and fit Prophet model
        model = Prophet(
            yearly_seasonality=len(daily_counts) >= 365,  # Only if we have a year of data
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            interval_width=0.80
        )
        
        # Add US holidays
        model.add_country_holidays(country_name='US')
        
        # Suppress Prophet's verbose output
        model.fit(daily_counts)
        
        # Create future dataframe and predict
        future = model.make_future_dataframe(periods=periods)
        prophet_forecast = model.predict(future)
        
        # Extract forecast for future dates only
        last_date = daily_counts['ds'].max()
        future_forecast = prophet_forecast[prophet_forecast['ds'] > last_date].copy()
        
        if len(future_forecast) == 0:
            logging.info("No future dates in forecast")
            return forecast_result
        
        forecast_result['has_forecast'] = True
        
        # Build daily forecast
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for _, row in future_forecast.iterrows():
            predicted = max(0, round(row['yhat']))  # Can't have negative crimes
            lower = max(0, round(row['yhat_lower']))
            upper = max(0, round(row['yhat_upper']))
            
            forecast_result['daily_forecast'].append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'day_name': day_names[row['ds'].weekday()],
                'predicted': predicted,
                'lower': lower,
                'upper': upper
            })
        
        # Calculate summary statistics
        predictions = [d['predicted'] for d in forecast_result['daily_forecast']]
        forecast_result['total_predicted'] = sum(predictions)
        forecast_result['avg_daily'] = round(sum(predictions) / len(predictions), 1)
        
        # Find peak and low days
        max_idx = predictions.index(max(predictions))
        min_idx = predictions.index(min(predictions))
        
        peak_day_data = forecast_result['daily_forecast'][max_idx]
        low_day_data = forecast_result['daily_forecast'][min_idx]
        
        forecast_result['peak_day'] = f"{peak_day_data['day_name']}, {peak_day_data['date']}"
        forecast_result['peak_count'] = peak_day_data['predicted']
        forecast_result['low_day'] = f"{low_day_data['day_name']}, {low_day_data['date']}"
        forecast_result['low_count'] = low_day_data['predicted']
        
        # Calculate weekly pattern from the forecast
        weekly_totals = {}
        for day_data in forecast_result['daily_forecast']:
            day_name = day_data['day_name']
            if day_name not in weekly_totals:
                weekly_totals[day_name] = []
            weekly_totals[day_name].append(day_data['predicted'])
        
        for day_name, counts in weekly_totals.items():
            forecast_result['weekly_pattern'][day_name] = round(sum(counts) / len(counts), 1)
        
        # Calculate trend (compare forecast avg to historical avg)
        historical_avg = daily_counts['y'].mean()
        forecast_avg = forecast_result['avg_daily']
        
        if historical_avg > 0:
            trend_pct = ((forecast_avg - historical_avg) / historical_avg) * 100
            forecast_result['trend_pct'] = round(trend_pct, 1)
            
            if trend_pct > 10:
                forecast_result['trend'] = 'increasing'
            elif trend_pct < -10:
                forecast_result['trend'] = 'decreasing'
            else:
                forecast_result['trend'] = 'stable'
        
        # Forecast by crime type (top 3)
        if 'CrimeType' in valid_df.columns:
            top_crimes = valid_df['CrimeType'].value_counts().head(3).index.tolist()
            total_crimes = len(valid_df)
            
            for crime_type in top_crimes:
                crime_pct = len(valid_df[valid_df['CrimeType'] == crime_type]) / total_crimes
                predicted_for_type = round(forecast_result['total_predicted'] * crime_pct)
                forecast_result['by_crime_type'][str(crime_type)] = {
                    'predicted': predicted_for_type,
                    'pct_of_total': round(crime_pct * 100, 1)
                }
        
        logging.info(f"Prophet forecast: {forecast_result['total_predicted']} crimes predicted over {periods} days, "
                     f"trend: {forecast_result['trend']} ({forecast_result['trend_pct']:+.1f}%)")
        
    except Exception as e:
        logging.warning(f"Prophet forecasting failed: {e}")
        return forecast_result
    
    return forecast_result


def detect_hotspots(df: pd.DataFrame, n_hotspots: int = 10, use_recency_weights: bool = True,
                    eps_km: float = 0.5, min_samples: int = 3) -> List[Dict[str, Any]]:
    """
    Detect crime hotspots using DBSCAN density-based clustering.
    Identifies natural crime clusters of irregular shapes rather than grid squares.
    Includes temporal context (peak hours/days) per hotspot if available.
    Optionally uses recency weighting to prioritize recent crimes.
    
    Args:
        df: DataFrame with Latitude, Longitude, CrimeType columns
            Optional: Hour, DayOfWeek, Date columns for temporal analysis
        n_hotspots: Maximum number of hotspots to return
        use_recency_weights: Whether to apply recency weighting (default True)
        eps_km: DBSCAN epsilon in kilometers (default 0.5km radius)
        min_samples: Minimum samples to form a cluster (default 3)
        
    Returns:
        List of hotspot dictionaries with location, crime info, temporal patterns, and recency
    """
    df_copy = df.copy()
    
    # Compute recency weights if date data available
    has_recency = 'Date' in df_copy.columns and df_copy['Date'].notna().any() and use_recency_weights
    if has_recency:
        df_copy['recency_weight'] = compute_recency_weights(df_copy)
    else:
        df_copy['recency_weight'] = 1.0
    
    # Prepare coordinates for DBSCAN
    # Convert lat/lon to radians for haversine metric
    coords = df_copy[['Latitude', 'Longitude']].values
    coords_radians = np.radians(coords)
    
    # DBSCAN with haversine metric
    # eps needs to be in radians: km / earth_radius_km
    earth_radius_km = 6371.0
    eps_radians = eps_km / earth_radius_km
    
    dbscan = DBSCAN(eps=eps_radians, min_samples=min_samples, metric='haversine')
    df_copy['cluster'] = dbscan.fit_predict(coords_radians)
    
    # Filter out noise points (cluster = -1)
    clustered_df = df_copy[df_copy['cluster'] != -1]
    
    if len(clustered_df) == 0:
        # Fallback: if no clusters found, use grid-based approach
        logging.warning("DBSCAN found no clusters, falling back to grid-based detection")
        df_copy['lat_grid'] = (df_copy['Latitude'] * 100).round() / 100
        df_copy['lon_grid'] = (df_copy['Longitude'] * 100).round() / 100
        grid_counts = df_copy.groupby(['lat_grid', 'lon_grid']).agg({
            'CrimeType': 'count',
            'recency_weight': 'sum'
        }).reset_index()
        grid_counts.columns = ['lat', 'lon', 'count', 'weighted_count']
        grid_counts = grid_counts.sort_values('weighted_count', ascending=False)
        
        # Create simple hotspots from grid
        hotspots = []
        for _, row in grid_counts.head(n_hotspots).iterrows():
            hotspots.append({
                'latitude': float(row['lat']),
                'longitude': float(row['lon']),
                'incident_count': int(row['count']),
                'weighted_score': round(float(row['weighted_count']), 2),
                'dominant_crime': 'Unknown',
                'crime_breakdown': {},
                'cluster_radius_km': None,
                'peak_hours': None,
                'peak_days': None,
                'temporal_pattern': None,
                'recency_score': None,
                'recent_incidents': 0,
                'older_incidents': 0,
                'trend': None,
                'is_emerging': False,
                'cluster_method': 'grid_fallback'
            })
        return hotspots
    
    # Check for temporal columns
    has_hours = 'Hour' in df_copy.columns and df_copy['Hour'].notna().any()
    has_days = 'DayOfWeek' in df_copy.columns and df_copy['DayOfWeek'].notna().any()
    
    # Aggregate cluster statistics
    cluster_stats = []
    for cluster_id in clustered_df['cluster'].unique():
        cluster_df = clustered_df[clustered_df['cluster'] == cluster_id]
        
        # Cluster centroid
        center_lat = cluster_df['Latitude'].mean()
        center_lon = cluster_df['Longitude'].mean()
        
        # Cluster radius (max distance from centroid in km)
        lat_diff = np.radians(cluster_df['Latitude'] - center_lat)
        lon_diff = np.radians(cluster_df['Longitude'] - center_lon)
        center_lat_rad = np.radians(center_lat)
        
        # Haversine formula for distances
        a = np.sin(lat_diff/2)**2 + np.cos(center_lat_rad) * np.cos(np.radians(cluster_df['Latitude'])) * np.sin(lon_diff/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distances_km = earth_radius_km * c
        cluster_radius = distances_km.max() if len(distances_km) > 0 else 0
        
        # Crime counts and types
        incident_count = len(cluster_df)
        weighted_score = cluster_df['recency_weight'].sum()
        crime_counts = cluster_df['CrimeType'].value_counts()
        dominant_crime = crime_counts.index[0] if len(crime_counts) > 0 else 'Unknown'
        crime_breakdown = {str(k): int(v) for k, v in crime_counts.head(3).items()}
        
        # Recency and trend analysis
        hotspot_recency_score = None
        recent_incident_count = 0
        older_incident_count = 0
        hotspot_trend = None
        
        if has_recency and 'Date' in cluster_df.columns:
            cluster_dates = cluster_df.dropna(subset=['Date'])
            if len(cluster_dates) > 0:
                max_date = df_copy['Date'].max()
                days_ago = (max_date - cluster_dates['Date']).dt.days
                
                recent_incident_count = int((days_ago <= 30).sum())
                older_incident_count = int(((days_ago > 30) & (days_ago <= 90)).sum())
                very_old_count = int((days_ago > 90).sum())
                
                hotspot_recency_score = round(cluster_df['recency_weight'].mean() * 100, 1)
                
                total_older = older_incident_count + very_old_count
                
                if total_older == 0 and recent_incident_count > 0:
                    hotspot_trend = 'new'
                elif recent_incident_count > 0 and older_incident_count > 0:
                    recent_rate = recent_incident_count
                    older_rate = older_incident_count / 2
                    
                    if recent_rate > older_rate * 1.5:
                        hotspot_trend = 'growing'
                    elif recent_rate < older_rate * 0.5:
                        hotspot_trend = 'shrinking'
                    else:
                        hotspot_trend = 'stable'
                elif recent_incident_count == 0 and total_older > 0:
                    hotspot_trend = 'inactive'
                else:
                    hotspot_trend = 'stable'
        
        # Temporal patterns for this cluster
        peak_hours = None
        peak_days = None
        temporal_pattern = None
        
        if has_hours:
            cluster_hours = cluster_df['Hour'].dropna()
            if len(cluster_hours) > 0:
                hour_counts = cluster_hours.value_counts()
                peak_hours = [int(h) for h in hour_counts.head(2).index.tolist()]
                
                morning = len(cluster_df[(cluster_df['Hour'] >= 6) & (cluster_df['Hour'] < 12)])
                afternoon = len(cluster_df[(cluster_df['Hour'] >= 12) & (cluster_df['Hour'] < 18)])
                evening = len(cluster_df[(cluster_df['Hour'] >= 18) & (cluster_df['Hour'] < 22)])
                night = len(cluster_df[(cluster_df['Hour'] >= 22) | (cluster_df['Hour'] < 6)])
                
                periods = {'morning': morning, 'afternoon': afternoon, 'evening': evening, 'night': night}
                temporal_pattern = max(periods, key=periods.get)
        
        if has_days:
            cluster_days = cluster_df['DayOfWeek'].dropna()
            if len(cluster_days) > 0:
                day_counts = cluster_days.value_counts()
                peak_days = [int(d) for d in day_counts.head(2).index.tolist()]
        
        cluster_stats.append({
            'cluster_id': cluster_id,
            'latitude': round(center_lat, 6),
            'longitude': round(center_lon, 6),
            'incident_count': incident_count,
            'weighted_score': round(weighted_score, 2),
            'cluster_radius_km': round(cluster_radius, 3),
            'dominant_crime': str(dominant_crime),
            'crime_breakdown': crime_breakdown,
            'peak_hours': peak_hours,
            'peak_days': peak_days,
            'temporal_pattern': temporal_pattern,
            'recency_score': hotspot_recency_score,
            'recent_incidents': recent_incident_count,
            'older_incidents': older_incident_count,
            'trend': hotspot_trend,
            'is_emerging': hotspot_trend in ['new', 'growing'],
            'cluster_method': 'dbscan'
        })
    
    # Sort by weighted score and return top n_hotspots
    cluster_stats.sort(key=lambda x: x['weighted_score'], reverse=True)
    hotspots = cluster_stats[:n_hotspots]
    
    # Remove cluster_id from output (internal use only)
    for h in hotspots:
        h.pop('cluster_id', None)
    
    total_clusters = len(cluster_stats)
    emerging_count = sum(1 for h in hotspots if h.get('is_emerging'))
    noise_count = len(df_copy[df_copy['cluster'] == -1])
    
    logging.info(f"DBSCAN detected {total_clusters} clusters from {len(df)} points "
                 f"(noise: {noise_count}, returning top {len(hotspots)}, emerging: {emerging_count})")
    
    return hotspots


def build_analysis_prompt(
    stats: Dict,
    hotspots: List[Dict],
    temporal: Optional[Dict] = None,
    crime_patterns: Optional[Dict[str, Dict]] = None,
    seasonal: Optional[Dict] = None,
    forecast: Optional[Dict] = None
) -> str:
    """
    Build a context-rich prompt for the LLM based on crime statistics.
    
    Args:
        stats: Crime statistics dictionary
        hotspots: List of detected hotspots
        temporal: Optional temporal patterns dictionary
        crime_patterns: Optional per-crime-type temporal patterns
        seasonal: Optional seasonal patterns dictionary
        forecast: Optional Prophet forecast dictionary
        
    Returns:
        Formatted prompt string for the LLM
    """
    bounds = stats['geographic_bounds']
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
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
    
    # Add recency section if date data available
    recency = stats.get('recency', {})
    if recency.get('has_date_data'):
        prompt += "\n**Data Recency:**\n"
        prompt += f"- Date range: {recency.get('oldest_date')} to {recency.get('newest_date')} ({recency.get('date_range_days')} days)\n"
        prompt += f"- Last 7 days: {recency.get('last_7_days')} incidents\n"
        prompt += f"- Last 30 days: {recency.get('last_30_days')} incidents ({recency.get('recent_activity_pct')}% of total)\n"
        prompt += f"- Older than 90 days: {recency.get('older_than_90_days')} incidents\n"
        if recency.get('recency_score') is not None:
            score = recency['recency_score']
            if score >= 70:
                prompt += f"- Overall recency: HIGH ({score}/100) - Most crimes are recent\n"
            elif score >= 40:
                prompt += f"- Overall recency: MEDIUM ({score}/100) - Mixed recent and older data\n"
            else:
                prompt += f"- Overall recency: LOW ({score}/100) - Most crimes are older\n"
    
    # Add temporal patterns section if available
    if temporal and temporal.get('has_temporal_data'):
        prompt += "\n**Temporal Patterns:**\n"
        
        if temporal.get('peak_hours'):
            peak_hours_str = ', '.join([f"{h}:00" for h in temporal['peak_hours']])
            prompt += f"- Peak crime hours: {peak_hours_str}\n"
        
        if temporal.get('peak_days'):
            peak_day_names = [day_names[d] for d in temporal['peak_days'] if 0 <= d <= 6]
            if peak_day_names:
                prompt += f"- Peak crime days: {', '.join(peak_day_names)}\n"
        
        if temporal.get('time_periods'):
            tp = temporal['time_periods']
            prompt += f"- Time breakdown: Morning {tp.get('morning', 0)}, Afternoon {tp.get('afternoon', 0)}, "
            prompt += f"Evening {tp.get('evening', 0)}, Night {tp.get('night', 0)}\n"
        
        if temporal.get('dominant_time_period'):
            prompt += f"- Most active period: {temporal['dominant_time_period'].capitalize()}\n"
        
        if temporal.get('recent_trend'):
            trend = temporal['recent_trend']
            change = temporal.get('trend_change_pct', 0)
            prompt += f"- Recent trend: {trend.capitalize()} ({change:+.1f}% change)\n"
    
    # Add per-crime-type temporal patterns if available
    if crime_patterns:
        prompt += "\n**Crime-Specific Timing Patterns:**\n"
        for crime_type, pattern in list(crime_patterns.items())[:5]:
            timing_info = []
            
            if pattern.get('peak_hours'):
                timing_info.append(f"peaks at {pattern['peak_hours'][0]}:00")
            
            if pattern.get('peak_period'):
                timing_info.append(f"{pattern['peak_period']} activity")
            
            if pattern.get('weekend_vs_weekday') == 'weekend':
                timing_info.append("more common on weekends")
            elif pattern.get('weekend_vs_weekday') == 'weekday':
                timing_info.append("more common on weekdays")
            
            if timing_info:
                prompt += f"- {crime_type}: {', '.join(timing_info)}\n"
    
    # Add seasonal patterns if available
    if seasonal and seasonal.get('has_seasonal_data'):
        prompt += "\n**Seasonal Patterns:**\n"
        
        if seasonal.get('peak_months'):
            prompt += f"- Peak crime months: {', '.join(seasonal['peak_months'])}\n"
        
        if seasonal.get('dominant_season'):
            prompt += f"- Dominant season: {seasonal['dominant_season'].capitalize()}\n"
        
        if seasonal.get('season_comparison'):
            sc = seasonal['season_comparison']
            prompt += f"- Seasonal breakdown: Winter {sc.get('winter', 0)}, Spring {sc.get('spring', 0)}, "
            prompt += f"Summer {sc.get('summer', 0)}, Fall {sc.get('fall', 0)}\n"
        
        if seasonal.get('holiday_spike'):
            effect = seasonal['holiday_spike']
            if effect == 'higher':
                prompt += "- Holiday effect: Crime is HIGHER near holidays\n"
            elif effect == 'lower':
                prompt += "- Holiday effect: Crime is LOWER near holidays\n"
            else:
                prompt += "- Holiday effect: Similar near holidays vs normal days\n"
        
        if seasonal.get('payday_spike'):
            effect = seasonal['payday_spike']
            if effect == 'higher':
                prompt += "- Payday effect: Crime is HIGHER around 1st/15th of month\n"
            elif effect == 'lower':
                prompt += "- Payday effect: Crime is LOWER around 1st/15th of month\n"
            else:
                prompt += "- Payday effect: Similar on paydays vs other days\n"
        
        if seasonal.get('year_over_year'):
            prompt += f"- Year-over-year trend: {seasonal['year_over_year']}\n"
    
    # Add Prophet forecast section if available
    if forecast and forecast.get('has_forecast'):
        prompt += f"\n**Crime Forecast (Next {forecast.get('horizon_days', 14)} Days - Prophet Model):**\n"
        prompt += f"- Total predicted: {forecast.get('total_predicted', 0)} incidents\n"
        prompt += f"- Daily average: {forecast.get('avg_daily', 0)} crimes/day\n"
        
        if forecast.get('peak_day'):
            prompt += f"- Peak day: {forecast['peak_day']} ({forecast.get('peak_count', 0)} predicted)\n"
        
        if forecast.get('low_day'):
            prompt += f"- Lowest day: {forecast['low_day']} ({forecast.get('low_count', 0)} predicted)\n"
        
        if forecast.get('trend'):
            trend = forecast['trend']
            trend_pct = forecast.get('trend_pct', 0)
            if trend == 'increasing':
                prompt += f"- Forecast trend: INCREASING ({trend_pct:+.1f}% vs historical average)\n"
            elif trend == 'decreasing':
                prompt += f"- Forecast trend: DECREASING ({trend_pct:+.1f}% vs historical average)\n"
            else:
                prompt += f"- Forecast trend: Stable ({trend_pct:+.1f}% vs historical average)\n"
        
        if forecast.get('by_crime_type'):
            prompt += "- Forecast by type: "
            type_parts = [f"{ct} ({data['predicted']})" for ct, data in list(forecast['by_crime_type'].items())[:3]]
            prompt += ", ".join(type_parts) + "\n"
    
    # Enhanced hotspot section with temporal, recency, and trend context
    prompt += "\n**Identified Hotspot Areas (DBSCAN clusters, ranked by recency-weighted activity):**\n"
    for i, hotspot in enumerate(hotspots[:5], 1):
        prompt += f"{i}. Location ({hotspot['latitude']:.4f}, {hotspot['longitude']:.4f}): "
        prompt += f"{hotspot['incident_count']} incidents, primarily {hotspot['dominant_crime']}"
        
        # Add cluster radius if available (from DBSCAN)
        if hotspot.get('cluster_radius_km'):
            prompt += f" (radius: {hotspot['cluster_radius_km']:.2f}km)"
        
        # Add trend, recency, and temporal context for this hotspot
        context_details = []
        
        # Trend info (new feature)
        trend = hotspot.get('trend')
        if trend == 'new':
            context_details.append("NEW HOTSPOT")
        elif trend == 'growing':
            context_details.append("GROWING")
        elif trend == 'shrinking':
            context_details.append("declining")
        elif trend == 'inactive':
            context_details.append("inactive recently")
        
        # Recent activity count
        if hotspot.get('recent_incidents', 0) > 0:
            context_details.append(f"{hotspot['recent_incidents']} in last 30 days")
        
        # Temporal info
        if hotspot.get('peak_hours'):
            peak_hour = hotspot['peak_hours'][0]
            context_details.append(f"peaks at {peak_hour}:00")
        
        if hotspot.get('peak_days'):
            peak_day = hotspot['peak_days'][0]
            if 0 <= peak_day <= 6:
                context_details.append(f"busiest on {day_names[peak_day]}")
        
        if hotspot.get('temporal_pattern'):
            context_details.append(f"{hotspot['temporal_pattern']} activity")
        
        if context_details:
            prompt += f" ({', '.join(context_details)})"
        
        prompt += "\n"
    
    prompt += """
## YOUR TASK

Based on this analysis, predict 10 locations where crimes are likely to occur in the future. 
Consider patterns in the data, hotspot clustering, crime type distributions, temporal patterns, and data recency.

**PRIORITIZE RECENT ACTIVITY**: Locations with recent crime activity (last 30 days) are more likely to have future crimes than areas with only old data. Emerging hotspots should be weighted heavily.

**IMPORTANT: You must respond with ONLY a valid JSON array. No other text before or after.**

Respond with this exact JSON format:
```json
[
  {
    "latitude": 33.7550,
    "longitude": -84.3900,
    "crime_type": "Theft",
    "prediction": "High-traffic commercial area with recent theft activity, most likely during afternoon hours",
    "likelihood": 85
  }
]
```

The likelihood should be a number from 0-100 representing the probability percentage.
Make sure all predicted coordinates are within the geographic bounds provided.
Include temporal context (time of day, day of week) in your predictions when relevant.
Give higher likelihood scores to areas with recent activity vs. areas with only old data.
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
    type_col: str = 'CrimeType',
    date_col: Optional[str] = None,
    time_col: Optional[str] = None
) -> Tuple[Dict, List[Dict], str, str, Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Run the complete crime analysis pipeline.
    
    Args:
        df: Raw crime data DataFrame
        lat_col: Name of latitude column
        lon_col: Name of longitude column
        type_col: Name of crime type column
        date_col: Optional name of date column for temporal analysis
        time_col: Optional name of time column for temporal analysis
        
    Returns:
        Tuple of (statistics, predictions, map_path, report_path, temporal_patterns, seasonal_patterns, forecast)
    """
    # Validate and normalize data (including temporal columns if provided)
    df = validate_columns(df, lat_col, lon_col, type_col, date_col, time_col)
    
    # Compute statistics
    stats = compute_crime_statistics(df)
    
    # Compute temporal patterns if temporal data is available
    temporal = compute_temporal_patterns(df)
    
    # Compute per-crime-type temporal patterns
    crime_patterns = compute_crime_type_patterns(df)
    
    # Compute seasonal patterns (holidays, paydays, etc.)
    seasonal = compute_seasonal_patterns(df)
    
    # Compute crime forecast using Prophet
    forecast = compute_crime_forecast(df)
    
    # Detect hotspots using DBSCAN (includes temporal context and trends per hotspot)
    hotspots = detect_hotspots(df)
    
    # Build prompt with temporal context, crime patterns, seasonal, forecast, and query LLM
    prompt = build_analysis_prompt(stats, hotspots, temporal, crime_patterns, seasonal, forecast)
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
    
    return stats, insights, map_path, report_path, temporal, seasonal, forecast


def main():
    """Main entry point for command-line usage."""
    sample_file = DATA_DIR / 'sample_crime_data.csv'
    
    if not sample_file.exists():
        logging.error(f"Data file not found: {sample_file}")
        logging.info("Please provide a crime data file with columns: Latitude, Longitude, CrimeType")
        return
    
    try:
        df = read_crime_data(str(sample_file))
        # Use temporal columns if available in sample data
        stats, insights, map_path, report_path, temporal, seasonal, forecast = run_analysis(
            df,
            date_col='Date' if 'Date' in df.columns else None,
            time_col='Time' if 'Time' in df.columns else None
        )
        
        print("\n" + "=" * 50)
        print("CRIME ANALYST AI - ANALYSIS COMPLETE")
        print("=" * 50)
        print(f"\nRecords analyzed: {stats['total_records']}")
        print(f"Predictions generated: {len(insights)}")
        
        if temporal and temporal.get('has_temporal_data'):
            print(f"\nTemporal Analysis:")
            if temporal.get('peak_hours'):
                print(f"  - Peak hours: {temporal['peak_hours']}")
            if temporal.get('recent_trend'):
                print(f"  - Trend: {temporal['recent_trend']}")
        
        if seasonal and seasonal.get('has_seasonal_data'):
            print(f"\nSeasonal Analysis:")
            if seasonal.get('peak_months'):
                print(f"  - Peak months: {seasonal['peak_months']}")
            if seasonal.get('dominant_season'):
                print(f"  - Dominant season: {seasonal['dominant_season']}")
            if seasonal.get('holiday_spike'):
                print(f"  - Holiday effect: {seasonal['holiday_spike']}")
        
        if forecast and forecast.get('has_forecast'):
            print(f"\nCrime Forecast (Next {forecast.get('horizon_days', 14)} Days):")
            print(f"  - Total predicted: {forecast.get('total_predicted', 0)} crimes")
            print(f"  - Daily average: {forecast.get('avg_daily', 0)}")
            if forecast.get('peak_day'):
                print(f"  - Peak day: {forecast['peak_day']} ({forecast.get('peak_count', 0)} predicted)")
            if forecast.get('trend'):
                print(f"  - Trend: {forecast['trend']} ({forecast.get('trend_pct', 0):+.1f}%)")
        
        print(f"\nOutputs:")
        print(f"  - Map: {map_path}")
        print(f"  - Report: {report_path}")
        
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        raise


if __name__ == '__main__':
    main()

