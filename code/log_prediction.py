
import pandas as pd
from datetime import datetime
import subprocess
import os

def log_prediction():
    """Logs the daily TMAX prediction and its accuracy."""
    
    # Construct the path to the python executable in the virtual environment
    python_executable = os.path.join(os.getcwd(), '.venv', 'bin', 'python3')

    # Run prediction script and capture output
    process = subprocess.run(
        [python_executable, 'code/run_prediction.py'], 
        capture_output=True, 
        text=True,
        env={'PYTHONPATH': '.'}
    )
    
    if process.returncode != 0:
        print(f"Error running prediction script: {process.stderr}")
        return
        
    predicted_tmax_str = process.stdout.strip().split(': ')[1].replace('°F', '')
    predicted_tmax = float(predicted_tmax_str)
    
    # Get actual TMAX
    daily_temp = pd.read_csv('data/klax_daily.csv')
    today_str = datetime.utcnow().strftime('%Y-%m-%d')
    
    actual_tmax = None
    if not daily_temp[daily_temp['DATE'] == today_str].empty:
        actual_tmax = daily_temp[daily_temp['DATE'] == today_str]['TMAX'].iloc[0]
    else:
        # Fetch from METAR as a fallback
        try:
            metar_process = subprocess.run(
                [python_executable, '-c', "from code.marine_features import get_metar_data, parse_metar_data; metars = parse_metar_data(get_metar_data(hours=1)); print(metars[-1].temp.value('C'))"],
                capture_output=True,
                text=True,
                env={'PYTHONPATH': '.'}
            )
            if metar_process.returncode == 0 and metar_process.stdout.strip():
                temp_c = float(metar_process.stdout.strip())
                actual_tmax = (temp_c * 9/5) + 32
        except Exception as e:
            print(f"Could not get METAR data: {e}")

    # Compute delta
    delta = None
    if actual_tmax is not None:
        delta = actual_tmax - predicted_tmax
        
    # Log to CSV
    log_file = 'data/model_performance_log.csv'
    log_entry = {
        'date': [today_str],
        'predicted': [predicted_tmax],
        'actual': [actual_tmax],
        'delta': [delta],
        'notes': ['']
    }
    
    try:
        log_df = pd.read_csv(log_file)
        log_df = pd.concat([log_df, pd.DataFrame(log_entry)], ignore_index=True)
    except FileNotFoundError:
        log_df = pd.DataFrame(log_entry)
        
    log_df.to_csv(log_file, index=False)
    
    print("Prediction logged:")
    print(log_df.tail(1))

if __name__ == '__main__':
    log_prediction()
