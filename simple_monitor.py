#!/usr/bin/env python3
"""
MARLIN v3 Simple Training Monitor
=================================
Basic monitoring script for CLI training sessions using shell commands
"""

import time
import subprocess
import pandas as pd
import os
from datetime import datetime, timedelta
import signal
import sys

class SimpleTrainingMonitor:
    def __init__(self):
        self.results_log = "data/overnight_training_log.csv"
        self.best_model_dir = "model/best_cli"
        self.training_script = "train_contextual_transformer_cli.py"
        self.overnight_script = "overnight_cli_training.py"
        
        # Monitoring settings
        self.check_interval = 30  # seconds
        self.start_time = datetime.now()
        
        print("🔍 MARLIN v3 Simple Training Monitor Initialized")
        print("=" * 50)
    
    def get_training_processes(self):
        """Find all training-related processes using ps"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            processes = []
            
            for line in result.stdout.split('\n'):
                if self.training_script in line or self.overnight_script in line:
                    parts = line.split()
                    if len(parts) >= 11:
                        processes.append({
                            'pid': parts[1],
                            'cpu_percent': parts[2],
                            'memory_percent': parts[3],
                            'command': ' '.join(parts[10:])
                        })
            return processes
        except Exception as e:
            print(f"Error getting processes: {e}")
            return []
    
    def get_training_progress(self):
        """Read current training progress from logs"""
        if not os.path.exists(self.results_log):
            return {"completed_runs": 0, "best_mae": float('inf'), "best_accuracy": 0.0, "success_rate": 0.0, "avg_duration": 0.0}
        
        try:
            df = pd.read_csv(self.results_log)
            if len(df) == 0:
                return {"completed_runs": 0, "best_mae": float('inf'), "best_accuracy": 0.0, "success_rate": 0.0, "avg_duration": 0.0}
            
            completed_runs = len(df)
            
            # Find best performance
            valid_runs = df[df['final_mae'] != 'FAILED']
            if len(valid_runs) > 0:
                best_mae = float(valid_runs['final_mae'].min())
                best_row = valid_runs.loc[valid_runs['final_mae'].idxmin()]
                best_accuracy = float(best_row['final_accuracy'])
            else:
                best_mae = float('inf')
                best_accuracy = 0.0
            
            return {
                "completed_runs": completed_runs,
                "best_mae": best_mae, 
                "best_accuracy": best_accuracy,
                "success_rate": len(valid_runs) / len(df) * 100 if len(df) > 0 else 0,
                "avg_duration": float(valid_runs['duration_minutes'].mean()) if len(valid_runs) > 0 else 0
            }
        except Exception as e:
            print(f"⚠️  Error reading progress: {e}")
            return {"completed_runs": 0, "best_mae": float('inf'), "best_accuracy": 0.0, "success_rate": 0.0, "avg_duration": 0.0}
    
    def get_system_stats(self):
        """Get basic system stats using shell commands"""
        try:
            # Get CPU usage
            cpu_result = subprocess.run(['top', '-l', '1', '-n', '0'], capture_output=True, text=True)
            cpu_percent = "N/A"
            for line in cpu_result.stdout.split('\n'):
                if 'CPU usage:' in line:
                    cpu_percent = line.split('CPU usage:')[1].split()[0]
                    break
            
            # Get memory usage
            mem_result = subprocess.run(['vm_stat'], capture_output=True, text=True)
            memory_percent = "N/A"
            
            return {
                "cpu_percent": cpu_percent,
                "memory_info": "Available"
            }
        except Exception as e:
            return {"cpu_percent": "N/A", "memory_info": "N/A"}
    
    def check_model_files(self):
        """Check if model files exist and get their info"""
        cli_model = "model/klax_contextual_transformer_cli.pth"
        cli_scaler = "model/contextual_cli_scaler.pkl"
        
        model_info = {}
        for filepath in [cli_model, cli_scaler]:
            if os.path.exists(filepath):
                stat = os.stat(filepath)
                model_info[os.path.basename(filepath)] = {
                    "size_mb": stat.st_size / (1024*1024),
                    "modified": datetime.fromtimestamp(stat.st_mtime)
                }
        
        # Check best model directory
        best_models = {}
        if os.path.exists(self.best_model_dir):
            for file in os.listdir(self.best_model_dir):
                if file.endswith(('.pth', '.pkl', '.json')):
                    filepath = os.path.join(self.best_model_dir, file)
                    stat = os.stat(filepath)
                    best_models[file] = {
                        "size_mb": stat.st_size / (1024*1024),
                        "modified": datetime.fromtimestamp(stat.st_mtime)
                    }
        
        return {"current": model_info, "best": best_models}
    
    def estimate_completion(self, progress):
        """Estimate when training will complete"""
        if progress["completed_runs"] == 0 or progress["avg_duration"] == 0:
            return "Unknown"
        
        remaining_runs = 20 - progress["completed_runs"]
        estimated_time = remaining_runs * progress["avg_duration"]
        completion_time = datetime.now() + timedelta(minutes=estimated_time)
        
        return f"{remaining_runs} runs left (~{estimated_time:.0f}min) - ETA: {completion_time.strftime('%H:%M:%S')}"
    
    def print_status(self):
        """Print comprehensive status update"""
        print(f"\n{'='*60}")
        print(f"🔍 MARLIN v3 Training Monitor - {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")
        
        # Get current data
        processes = self.get_training_processes()
        progress = self.get_training_progress()
        system_stats = self.get_system_stats()
        model_info = self.check_model_files()
        
        # Training Status
        print(f"\n🚀 TRAINING STATUS:")
        if processes:
            for proc in processes:
                script_name = "OVERNIGHT" if self.overnight_script in proc['command'] else "TRAINING"
                print(f"   {script_name} (PID {proc['pid']}): CPU {proc['cpu_percent']}% | Memory {proc['memory_percent']}%")
        else:
            print("   ❌ No active training processes")
        
        # Progress Metrics
        print(f"\n📊 PROGRESS METRICS:")
        print(f"   Completed Runs: {progress['completed_runs']}/20")
        print(f"   Success Rate: {progress['success_rate']:.1f}%")
        if progress['best_mae'] != float('inf'):
            print(f"   Best Performance: {progress['best_mae']:.3f}°F MAE | {progress['best_accuracy']:.1f}% Accuracy")
        else:
            print("   Best Performance: No successful runs yet")
        
        if progress['avg_duration'] > 0:
            print(f"   Avg Duration: {progress['avg_duration']:.1f} min/run")
            print(f"   ETA: {self.estimate_completion(progress)}")
        
        # System Resources
        print(f"\n💻 SYSTEM RESOURCES:")
        print(f"   CPU: {system_stats['cpu_percent']} | Memory: {system_stats['memory_info']}")
        
        # Model Files
        print(f"\n📁 MODEL FILES:")
        if model_info["current"]:
            for filename, info in model_info["current"].items():
                print(f"   {filename}: {info['size_mb']:.1f}MB (modified: {info['modified'].strftime('%H:%M:%S')})")
        else:
            print("   ❌ No current model files found")
        
        if model_info["best"]:
            print("   Best Models:")
            for filename, info in model_info["best"].items():
                print(f"     {filename}: {info['size_mb']:.1f}MB (saved: {info['modified'].strftime('%H:%M:%S')})")
        
        # Check if training is active
        if processes:
            print(f"\n✅ Training is active and running!")
        else:
            print(f"\n⚠️  No training processes detected")
        
        # Runtime
        total_runtime = datetime.now() - self.start_time
        print(f"\n⏱️  Monitor Runtime: {total_runtime}")
        print(f"{'='*60}")
    
    def run_monitor(self, duration_hours=8):
        """Run continuous monitoring"""
        end_time = datetime.now() + timedelta(hours=duration_hours)
        
        print(f"🔍 Starting {duration_hours}-hour monitoring session...")
        print(f"Will monitor until: {end_time.strftime('%H:%M:%S')}")
        
        try:
            while datetime.now() < end_time:
                self.print_status()
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
        
        print(f"\n🏁 Monitoring session complete!")
        self.print_status()  # Final status

def signal_handler(sig, frame):
    print('\n⏹️  Monitoring stopped gracefully')
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command line arguments
    duration = 8  # Default 8 hours
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            print("Usage: python simple_monitor.py [hours]")
            sys.exit(1)
    
    monitor = SimpleTrainingMonitor()
    monitor.run_monitor(duration_hours=duration) 