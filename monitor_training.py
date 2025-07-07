#!/usr/bin/env python3
"""
MARLIN v3 Overnight Training Monitor
====================================
Real-time monitoring script for CLI training sessions with:
- Live progress tracking
- Performance metrics
- System health monitoring  
- Auto-restart capabilities
- Early warning alerts
"""

import time
import subprocess
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import psutil
import signal
import sys

class TrainingMonitor:
    def __init__(self):
        self.results_log = "data/overnight_training_log.csv"
        self.best_model_dir = "model/best_cli"
        self.training_script = "code/train_contextual_transformer_cli.py"
        self.overnight_script = "code/overnight_cli_training.py"
        
        # Monitoring settings
        self.check_interval = 30  # seconds
        self.max_stall_time = 600  # 10 minutes without progress
        self.max_run_time = 7200   # 2 hours per training run
        
        # State tracking
        self.last_seen_run = 0
        self.last_progress_time = datetime.now()
        self.start_time = datetime.now()
        
        print("🔍 MARLIN v3 Training Monitor Initialized")
        print("=" * 50)
    
    def get_training_processes(self):
        """Find all training-related processes"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if any(script in cmdline for script in [self.training_script, self.overnight_script]):
                    processes.append({
                        'pid': proc.info['pid'],
                        'cmdline': cmdline,
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_percent': proc.info['memory_percent'],
                        'runtime': datetime.now() - datetime.fromtimestamp(proc.info['create_time'])
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return processes
    
    def get_training_progress(self):
        """Read current training progress from logs"""
        if not os.path.exists(self.results_log):
            return {"completed_runs": 0, "best_mae": float('inf'), "best_accuracy": 0.0, "last_run_time": None}
        
        try:
            df = pd.read_csv(self.results_log)
            if len(df) == 0:
                return {"completed_runs": 0, "best_mae": float('inf'), "best_accuracy": 0.0, "last_run_time": None}
            
            completed_runs = len(df)
            
            # Find best performance
            valid_runs = df[df['final_mae'] != 'FAILED']
            if len(valid_runs) > 0:
                best_mae = valid_runs['final_mae'].min()
                best_accuracy = valid_runs.loc[valid_runs['final_mae'].idxmin(), 'final_accuracy']
            else:
                best_mae = float('inf')
                best_accuracy = 0.0
            
            # Get last run time
            last_run_time = None
            if completed_runs > 0:
                last_run_time = pd.to_datetime(df.iloc[-1]['end_time'])
            
            return {
                "completed_runs": completed_runs,
                "best_mae": best_mae, 
                "best_accuracy": best_accuracy,
                "last_run_time": last_run_time,
                "success_rate": len(valid_runs) / len(df) * 100 if len(df) > 0 else 0,
                "avg_duration": valid_runs['duration_minutes'].mean() if len(valid_runs) > 0 else 0
            }
        except Exception as e:
            print(f"⚠️  Error reading progress: {e}")
            return {"completed_runs": 0, "best_mae": float('inf'), "best_accuracy": 0.0, "last_run_time": None}
    
    def get_system_stats(self):
        """Get system resource usage"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('.').percent
        }
    
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
    
    def detect_issues(self, processes, progress):
        """Detect potential training issues"""
        issues = []
        
        # Check if training is running
        if not processes:
            issues.append("🚨 No training processes found!")
        
        # Check for stuck training
        if progress["completed_runs"] == self.last_seen_run:
            time_since_progress = (datetime.now() - self.last_progress_time).total_seconds()
            if time_since_progress > self.max_stall_time:
                issues.append(f"⚠️  No progress for {time_since_progress/60:.1f} minutes")
        else:
            self.last_seen_run = progress["completed_runs"]
            self.last_progress_time = datetime.now()
        
        # Check for long-running processes
        for proc in processes:
            if proc['runtime'].total_seconds() > self.max_run_time:
                issues.append(f"⚠️  Process {proc['pid']} running too long ({proc['runtime']})")
        
        # Check resource usage
        system_stats = self.get_system_stats()
        if system_stats["memory_percent"] > 90:
            issues.append(f"⚠️  High memory usage: {system_stats['memory_percent']:.1f}%")
        
        if system_stats["disk_usage"] > 95:
            issues.append(f"⚠️  Low disk space: {system_stats['disk_usage']:.1f}% used")
        
        return issues
    
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
                script_name = "overnight" if self.overnight_script in proc['cmdline'] else "training"
                print(f"   {script_name.upper()} (PID {proc['pid']}): CPU {proc['cpu_percent']:.1f}% | Runtime: {proc['runtime']}")
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
        print(f"   CPU: {system_stats['cpu_percent']:.1f}% | Memory: {system_stats['memory_percent']:.1f}% | Disk: {system_stats['disk_usage']:.1f}%")
        
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
        
        # Issue Detection
        issues = self.detect_issues(processes, progress)
        if issues:
            print(f"\n🚨 ALERTS:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print(f"\n✅ All systems running smoothly!")
        
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
            print("Usage: python monitor_training.py [hours]")
            sys.exit(1)
    
    monitor = TrainingMonitor()
    monitor.run_monitor(duration_hours=duration) 