

import pandas as pd
import requests
from metar import Metar
import numpy as np
from datetime import datetime, timedelta
import pytz
from zoneinfo import ZoneInfo # Python 3.9+
import re

def get_metar_data(station='KLAX', start_dt=None, end_dt=None):
    """Fetches METAR data for a given station and time range."""
    if start_dt is None:
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(hours=12) # Default to last 12 hours if no range specified

    start_str = start_dt.strftime('%Y%m%d%H%M')
    end_str = end_dt.strftime('%Y%m%d%H%M')
    
    url = f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={station}&data=metar&year1={start_dt.year}&month1={start_dt.month}&day1={start_dt.day}&hour1={start_dt.hour}&minute1={start_dt.minute}&year2={end_dt.year}&month2={end_dt.month}&day2={end_dt.day}&hour2={end_dt.hour}&minute2={end_dt.minute}&tz=Etc/UTC&format=onlycomma&latlon=no&direct=no&report_type=1&report_type=2"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching METAR data: {e}")
        return None

def parse_metar_data(metar_text):
    """Parses raw METAR text into a list of Metar objects."""
    lines = metar_text.strip().split('\n')
    metar_reports = []
    for line in lines:
        if line.strip() and "station,valid,metar" not in line:
            try:
                parts = line.split(',')
                if len(parts) >= 3:
                    timestamp_str = parts[1].strip()
                    context_dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M')
                    context_year = context_dt.year
                    context_month = context_dt.month

                    metar_pattern = re.compile(r'(K?LAX\s\d{6}Z.*?)(?:\sRMK|\sIEM_GHCNH|\sKRSA-SRUS56-RR5RSA|$)')
                    search_string = ','.join(parts[2:]).strip()
                    match = metar_pattern.search(search_string)

                    if match:
                        core_metar_string = match.group(1).strip()

                        if core_metar_string.startswith('LAX'):
                            core_metar_string = 'K' + core_metar_string
                        elif not core_metar_string.startswith('KLAX'):
                            core_metar_string = 'KLAX ' + core_metar_string

                        metar_reports.append(Metar.Metar(core_metar_string, year=context_year, month=context_month))
                    else:
                        pass 
                else:
                    pass 
            except Exception as e:
                print(f"Could not parse METAR string: {line.strip()} - {e}")
    return metar_reports

def utc_to_la_time(dt_object):
    """Converts a UTC datetime object to America/Los_Angeles timezone."""
    utc_zone = pytz.utc
    la_zone = pytz.timezone('America/Los_Angeles')
    return utc_zone.localize(dt_object).astimezone(la_zone)

def get_ceiling_data_for_time_range(metars, start_hour_la, end_hour_la):
    """Extracts ceiling data for a specific time range in LA time."""
    ceilings = []
    for m in metars:
        la_time = utc_to_la_time(m.time)
        if start_hour_la <= la_time.hour < end_hour_la:
            if m.sky and m.sky[0][0] in ['BKN', 'OVC']:
                 if m.sky[0][1]:
                    ceilings.append(m.sky[0][1].value())
    return ceilings

def calculate_ceiling_slope(metars):
    """
    Calculates the ceiling slope: difference between average ceiling height
    at 3-6 AM PST and 6-9 AM PST, in feet/hour. Missing values default to 0.
    """
    ceilings_3_6am = get_ceiling_data_for_time_range(metars, 3, 6)
    ceilings_6_9am = get_ceiling_data_for_time_range(metars, 6, 9)

    avg_ceiling_3_6am = np.mean(ceilings_3_6am) if ceilings_3_6am else np.nan
    avg_ceiling_6_9am = np.mean(ceilings_6_9am) if ceilings_6_9am else np.nan

    if np.isnan(avg_ceiling_3_6am) or np.isnan(avg_ceiling_6_9am):
        return 0.0 # As per instructions, missing values = 0
    
    # The time difference is 3 hours (from mid-point of 3-6 to mid-point of 6-9)
    # (4.5 - 7.5) = -3 hours
    # Slope = (change in ceiling) / (change in time)
    # Here, we are looking for the difference between the two averages,
    # which implicitly assumes a 3-hour interval between the midpoints of the ranges.
    # So, the slope is (avg_6_9am - avg_3_6am) / 3 hours
    return (avg_ceiling_6_9am - avg_ceiling_3_6am) / 3.0

def get_temp_slope_7_10am(metars):
    """
    Calculates the rate of temperature rise from 7AM to 10AM PST, in °F/hour.
    Uses closest observations if exact times are not available.
    """
    temps_with_time = []
    for m in metars:
        la_time = utc_to_la_time(m.time)
        if 7 <= la_time.hour < 10 and m.temp:
            # Convert temperature from Celsius to Fahrenheit
            temp_f = (m.temp.value() * 9/5) + 32
            temps_with_time.append((la_time, temp_f))
    
    if len(temps_with_time) < 2:
        return 0.0 # Not enough data points for a slope

    # Sort by time to ensure correct order for slope calculation
    temps_with_time.sort(key=lambda x: x[0])

    # Extract times and temperatures
    times = np.array([(t.hour * 60 + t.minute) for t, _ in temps_with_time]) # minutes from start of day
    temps = np.array([t for _, t in temps_with_time])

    # Calculate slope using linear regression
    # Convert minutes to hours for slope calculation
    slope, _ = np.polyfit(times / 60.0, temps, 1)
    return slope

def get_sun_flag_8am(metars):
    """
    Binary value: 1 if sky condition at 8AM PST is FEW or SKC, 0 otherwise.
    Uses closest observation if exact 8AM is not available.
    """
    target_hour_la = 8
    
    # Find the METAR report closest to 8 AM PST
    closest_metar = None
    min_time_diff = timedelta.max

    for m in metars:
        la_time = utc_to_la_time(m.time)
        time_diff = abs(la_time - la_time.replace(hour=target_hour_la, minute=0, second=0, microsecond=0))
        
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_metar = m
            
    if closest_metar and closest_metar.sky:
        # Check sky condition of the closest METAR
        if closest_metar.sky[0][0] in ['FEW', 'SKC']:
            return 1
    return 0

def get_dewpoint_spread_5am(metars):
    """
    Calculates temp - dewpoint at or closest to 5AM PST, in °F.
    Uses closest observation if exact 5AM is not available.
    """
    target_hour_la = 5
    
    # Find the METAR report closest to 5 AM PST
    closest_metar = None
    min_time_diff = timedelta.max

    for m in metars:
        la_time = utc_to_la_time(m.time)
        time_diff = abs(la_time - la_time.replace(hour=target_hour_la, minute=0, second=0, microsecond=0))
        
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_metar = m
            
    if closest_metar and closest_metar.temp and closest_metar.dewpt:
        # Convert to Fahrenheit before calculating spread
        temp_f = (closest_metar.temp.value() * 9/5) + 32
        dewpt_f = (closest_metar.dewpt.value() * 9/5) + 32
        return temp_f - dewpt_f
    return np.nan

def create_nowcast_features(target_date=None):
    """
    Creates and saves enriched nowcast features to data/klax_nowcast_features_enriched.csv.
    This function is based on the original purpose of the file.
    """
    if target_date is None:
        start_dt = datetime.utcnow() - timedelta(hours=12)
        end_dt = datetime.utcnow()
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    else:
        start_dt = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=pytz.utc)
        end_dt = start_dt + timedelta(days=1) - timedelta(minutes=1)
        date_str = target_date.strftime('%Y-%m-%d')

    metar_text = get_metar_data(start_dt=start_dt, end_dt=end_dt)
    if not metar_text:
        print(f"Could not fetch METAR data for nowcast for {date_str}. Skipping.")
        return None

    metars = parse_metar_data(metar_text)
    if not metars:
        print(f"Could not parse METAR data for nowcast for {date_str}. Skipping.")
        return None
        
    wind_speeds = []
    wind_directions = []
    cloud_covers = []

    for m in metars:
        if m.wind_speed:
            wind_speeds.append(m.wind_speed.value())
        if m.wind_dir:
            wind_directions.append(m.wind_dir.value())
        
        if m.sky and m.sky[0][0] in ['FEW', 'SCT', 'BKN', 'OVC']:
            if m.sky[0][0] == 'FEW':
                cloud_covers.append(25)
            elif m.sky[0][0] == 'SCT':
                cloud_covers.append(50)
            elif m.sky[0][0] == 'BKN':
                cloud_covers.append(75)
            elif m.sky[0][0] == 'OVC':
                cloud_covers.append(100)
        else:
            cloud_covers.append(0) # Clear sky

    dewpoint_spread = get_dewpoint_spread_5am(metars) # This is already in F

    avg_wind_speed_6h = np.mean(wind_speeds) if wind_speeds else np.nan
    avg_wind_direction_6h = np.mean(wind_directions) if wind_directions else np.nan
    morning_cloud_cover_avg = np.mean(cloud_covers) if cloud_covers else np.nan

    enriched_features = {
        'avg_wind_speed_6h': [avg_wind_speed_6h],
        'avg_wind_direction_6h': [avg_wind_direction_6h],
        'morning_cloud_cover_avg': [morning_cloud_cover_avg],
        'dewpoint_spread_5am': [dewpoint_spread],
        'DATE': [date_str]
    }
    
    df = pd.DataFrame(enriched_features)
    return df

def create_new_marine_features(target_date=None):
    """
    Creates and returns marine-relevant features for KLAX.
    This function uses the newly defined marine feature calculations.
    """
    if target_date is None:
        start_dt = datetime.utcnow() - timedelta(hours=12)
        end_dt = datetime.utcnow()
        date_str = datetime.utcnow().strftime('%Y-%m-%d')
    else:
        start_dt = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=pytz.utc)
        end_dt = start_dt + timedelta(days=1) - timedelta(minutes=1)
        date_str = target_date.strftime('%Y-%m-%d')

    metar_text = get_metar_data(start_dt=start_dt, end_dt=end_dt)
    if not metar_text:
        print(f"Could not fetch METAR data for marine features for {date_str}. Skipping.")
        return None

    metars = parse_metar_data(metar_text)
    if not metars:
        print(f"Could not parse METAR data for marine features for {date_str}. Skipping.")
        return None
        
    ceiling_slope = calculate_ceiling_slope(metars)
    temp_slope_7_10am = get_temp_slope_7_10am(metars)
    sun_flag_8am = get_sun_flag_8am(metars)
    dewpoint_spread_5am = get_dewpoint_spread_5am(metars)
    
    marine_features = {
        'ceiling_slope': [ceiling_slope],
        'temp_slope_7_10am': [temp_slope_7_10am],
        'sun_flag_8am': [sun_flag_8am],
        'dewpoint_spread_5am': [dewpoint_spread_5am],
        'DATE': [date_str]
    }
    
    df = pd.DataFrame(marine_features)
    return df

if __name__ == '__main__':
    # When marine_features.py is run directly, it should generate both files for today
    nowcast_df = create_nowcast_features()
    if nowcast_df is not None:
        nowcast_df.to_csv('data/klax_nowcast_features_enriched.csv', index=False)
        print(f"Nowcast features created and saved to data/klax_nowcast_features_enriched.csv:")
        print(nowcast_df)

    marine_df = create_new_marine_features()
    if marine_df is not None:
        marine_df.to_csv('data/klax_marine_features.csv', index=False)
        print(f"Marine features created and saved to data/klax_marine_features.csv:")
        print(marine_df)
