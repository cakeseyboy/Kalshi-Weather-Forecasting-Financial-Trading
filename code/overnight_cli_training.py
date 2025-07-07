#!/usr/bin/env python3
"""
Overnight CLI Training Script for MARLIN v3
Runs multiple training passes with different random seeds to find the best CLI model.
"""

import os
import sys
import time
import subprocess
import signal
import pandas as pd
import numpy as np
import torch
import pickle
import asyncio
import libsql_client
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
TRAINING_SCRIPT = 'code/train_contextual_transformer_cli.py'
RESULTS_LOG = 'data/overnight_training_log.csv'
BEST_MODEL_DIR = 'model/best_cli'
DB_URL = os.getenv("TURSO_DATABASE_URL")
AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")

class OvernightTrainer:
    def __init__(self):
        self.results = []
        self.best_mae = float('inf')
        self.best_accuracy = 0.0
        self.current_run = 0
        self.start_time = datetime.now()
        
        # Create best model directory
        os.makedirs(BEST_MODEL_DIR, exist_ok=True)
        
        # Initialize results log
        self.init_results_log()
        
    def init_results_log(self):
        """Initialize the results CSV file"""
        if not os.path.exists(RESULTS_LOG):
            df = pd.DataFrame(columns=[
                'run_number', 'start_time', 'end_time', 'duration_minutes',
                'final_mae', 'final_accuracy', 'best_mae', 'epochs_completed',
                'random_seed', 'model_path', 'scaler_path', 'notes'
            ])
            df.to_csv(RESULTS_LOG, index=False)
    
    def wait_for_current_training(self):
        """Wait for the currently running training to finish"""
        print("=== Monitoring Current Training Process ===")
        
        while True:
            # Use pgrep to check for training process (simpler than psutil)
            try:
                result = subprocess.run(['pgrep', '-f', 'train_contextual_transformer_cli.py'], 
                                      capture_output=True, text=True)
                
                if result.returncode != 0:
                    print("✅ No active training processes found. Ready to start overnight training!")
                    break
                
                pids = result.stdout.strip().split('\n')
                print(f"⏳ Found {len(pids)} training process(es) still running...")
                for pid in pids:
                    print(f"   PID: {pid}")
                
                print("   Waiting 30 seconds before checking again...")
                time.sleep(30)
                
            except Exception as e:
                print(f"   Could not check processes: {e}")
                print("   Assuming no processes running...")
                break
    
    def run_single_training(self, run_number, random_seed=None):
        """Run a single training pass"""
        print(f"{'='*60}")
        print(f"🚀 STARTING TRAINING RUN #{run_number}")
        print(f"{'='*60}")
        
        start_time = datetime.now()
        
        if random_seed is None:
            random_seed = int(time.time()) % 10000  # Use timestamp-based seed
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        print(f"Random seed: {random_seed}")
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Training script: {TRAINING_SCRIPT}")
        print(f"\n🔥 REAL-TIME TRAINING OUTPUT:")
        print("=" * 60)
        
        # Run training script with real-time output
        try:
            # Activate virtual environment and run training
            cmd = ['source', '.venv/bin/activate', '&&', 'python', TRAINING_SCRIPT]
            
            # Use Popen for real-time output
            process = subprocess.Popen(
                ' '.join(cmd), 
                shell=True, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Capture output while showing real-time
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())  # Show real-time
                    output_lines.append(output)
            
            # Wait for completion
            process.wait()
            full_output = ''.join(output_lines)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            
            print(f"\n{'='*60}")
            print(f"📊 Training Run #{run_number} Results:")
            print(f"   Duration: {duration:.1f} minutes")
            print(f"   Exit code: {process.returncode}")
            
            # Parse output for metrics
            mae, accuracy, epochs = self.parse_training_output(full_output)
            
            print(f"   Final MAE: {mae:.3f}°F")
            print(f"   Rounding Accuracy: {accuracy:.1f}%")
            print(f"   Epochs Completed: {epochs}")
            
            # Check if this is the best model
            is_best = self.check_and_save_best_model(mae, accuracy, run_number, random_seed)
            
            # Log results
            self.log_results(run_number, start_time, end_time, duration, mae, accuracy, epochs, random_seed, is_best)
            
            if process.returncode != 0:
                print(f"⚠️  Training failed with exit code {process.returncode}")
            
        except Exception as e:
            print(f"❌ Error in training run #{run_number}: {e}")
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds() / 60
            self.log_results(run_number, start_time, end_time, duration, float('inf'), 0.0, 0, random_seed, False, f"ERROR: {str(e)}")
    
    def parse_training_output(self, output):
        """Parse training output to extract final metrics"""
        lines = output.split('\n')
        mae = float('inf')
        accuracy = 0.0
        epochs = 0
        
        for line in lines:
            # Look for final performance line
            if "Final Performance:" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "°F" in part:
                        try:
                            mae = float(part.replace('°F', ''))
                        except:
                            pass
                    if "%" in part and "Rounding" in line:
                        try:
                            accuracy = float(part.replace('%', ''))
                        except:
                            pass
            
            # Count epochs
            if "Epoch" in line and "/" in line:
                try:
                    epoch_part = line.split("Epoch")[1].split("/")[0].strip()
                    epochs = max(epochs, int(epoch_part))
                except:
                    pass
        
        return mae, accuracy, epochs
    
    def check_and_save_best_model(self, mae, accuracy, run_number, random_seed):
        """Check if this is the best model and save it"""
        is_best = False
        
        # Primary: lowest MAE, Secondary: highest accuracy
        if mae < self.best_mae or (mae == self.best_mae and accuracy > self.best_accuracy):
            print(f"🏆 NEW BEST MODEL! (Previous: {self.best_mae:.3f}°F, {self.best_accuracy:.1f}%)")
            self.best_mae = mae
            self.best_accuracy = accuracy
            is_best = True
            
            # Copy current model files to best directory
            try:
                import shutil
                
                # Source files (current model)
                model_src = 'model/klax_contextual_transformer_cli.pth'
                scaler_src = 'model/contextual_cli_scaler.pkl'
                
                # Destination files (best model)
                model_dst = f'{BEST_MODEL_DIR}/klax_contextual_transformer_cli_best.pth'
                scaler_dst = f'{BEST_MODEL_DIR}/contextual_cli_scaler_best.pkl'
                
                if os.path.exists(model_src):
                    shutil.copy2(model_src, model_dst)
                if os.path.exists(scaler_src):
                    shutil.copy2(scaler_src, scaler_dst)
                
                # Save metadata
                metadata = {
                    'run_number': run_number,
                    'mae': mae,
                    'accuracy': accuracy,
                    'random_seed': random_seed,
                    'timestamp': datetime.now().isoformat(),
                    'model_path': model_dst,
                    'scaler_path': scaler_dst
                }
                
                with open(f'{BEST_MODEL_DIR}/best_model_metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"   ✅ Best model saved to {BEST_MODEL_DIR}/")
                
            except Exception as e:
                print(f"   ⚠️  Error saving best model: {e}")
        
        return is_best
    
    def log_results(self, run_number, start_time, end_time, duration, mae, accuracy, epochs, random_seed, is_best, notes=""):
        """Log results to CSV file"""
        new_row = {
            'run_number': run_number,
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_minutes': round(duration, 1),
            'final_mae': mae if mae != float('inf') else 'FAILED',
            'final_accuracy': accuracy,
            'best_mae': 'YES' if is_best else 'NO',
            'epochs_completed': epochs,
            'random_seed': random_seed,
            'model_path': 'model/klax_contextual_transformer_cli.pth',
            'scaler_path': 'model/contextual_cli_scaler.pkl',
            'notes': notes
        }
        
        # Append to CSV
        df = pd.read_csv(RESULTS_LOG)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(RESULTS_LOG, index=False)
    
    def print_status_summary(self):
        """Print current training status"""
        elapsed = datetime.now() - self.start_time
        
        print(f"\n{'='*60}")
        print(f"📈 OVERNIGHT TRAINING STATUS")
        print(f"{'='*60}")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed: {elapsed}")
        print(f"Completed Runs: {self.current_run}")
        print(f"Best MAE: {self.best_mae:.3f}°F")
        print(f"Best Accuracy: {self.best_accuracy:.1f}%")
        
        if os.path.exists(RESULTS_LOG):
            df = pd.read_csv(RESULTS_LOG)
            if len(df) > 0:
                print(f"Average Duration: {df['duration_minutes'].mean():.1f} min")
                print(f"Success Rate: {(df['final_mae'] != 'FAILED').mean()*100:.1f}%")
        
        print(f"{'='*60}")
    
    def run_overnight_training(self, max_runs=20, max_hours=8):
        """Run multiple training passes overnight"""
        print("🌙 Starting Overnight CLI Training Marathon!")
        print(f"Max runs: {max_runs}")
        print(f"Max duration: {max_hours} hours")
        
        # Wait for current training to finish
        self.wait_for_current_training()
        
        end_time = self.start_time + timedelta(hours=max_hours)
        
        while self.current_run < max_runs and datetime.now() < end_time:
            self.current_run += 1
            
            # Run training
            self.run_single_training(self.current_run)
            
            # Print status
            self.print_status_summary()
            
            # Short break between runs
            if self.current_run < max_runs and datetime.now() < end_time:
                print(f"\n⏸️  Resting for 1 minute before next run...")
                time.sleep(60)
        
        # Final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final training summary"""
        total_time = datetime.now() - self.start_time
        
        print(f"\n{'='*60}")
        print(f"🏁 OVERNIGHT TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Total Time: {total_time}")
        print(f"Total Runs: {self.current_run}")
        print(f"Best MAE: {self.best_mae:.3f}°F")
        print(f"Best Accuracy: {self.best_accuracy:.1f}%")
        print(f"Results Log: {RESULTS_LOG}")
        print(f"Best Model: {BEST_MODEL_DIR}/")
        
        if os.path.exists(RESULTS_LOG):
            print(f"\n📊 Final Statistics:")
            df = pd.read_csv(RESULTS_LOG)
            successful_runs = df[df['final_mae'] != 'FAILED']
            
            if len(successful_runs) > 0:
                print(f"   Success Rate: {len(successful_runs)/len(df)*100:.1f}%")
                print(f"   MAE Range: {successful_runs['final_mae'].min():.3f} - {successful_runs['final_mae'].max():.3f}°F")
                print(f"   Accuracy Range: {successful_runs['final_accuracy'].min():.1f} - {successful_runs['final_accuracy'].max():.1f}%")
                print(f"   Avg Duration: {successful_runs['duration_minutes'].mean():.1f} minutes")
        
        print(f"\n🎯 Best model ready for Kalshi trading!")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        max_runs = int(sys.argv[1])
    else:
        max_runs = 20
    
    if len(sys.argv) > 2:
        max_hours = int(sys.argv[2])
    else:
        max_hours = 8
    
    trainer = OvernightTrainer()
    
    try:
        trainer.run_overnight_training(max_runs=max_runs, max_hours=max_hours)
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Training interrupted by user")
        trainer.print_final_summary()
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        trainer.print_final_summary()
        raise

if __name__ == '__main__':
    main() 