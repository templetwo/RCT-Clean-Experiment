#!/usr/bin/env python3
"""
RCT Training Monitor Dashboard
Real-time monitoring for Relational Coherence Training
"""

import os
import re
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys

def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_process_status():
    """Check if training is running."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if 'train_rct.py' in line and 'grep' not in line:
                return True, line
        return False, None
    except:
        return False, None

def parse_log_file(log_path):
    """Extract metrics from training log."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    metrics = {
        'current_step': 0,
        'total_steps': 920,
        'current_epoch': 0.0,
        'total_epochs': 10,
        'train_loss': None,
        'eval_loss': None,
        'grad_norm': None,
        'learning_rate': None,
        'time_per_step': None,
        'last_update': None,
        'start_time': None
    }

    # Find start time
    for line in lines:
        if "Starting training" in line:
            metrics['start_time'] = datetime.now() - timedelta(seconds=len(lines) * 0.5)
            break

    # Parse progress from last lines
    for line in reversed(lines[-100:]):
        # Extract step progress: "  14%|█▍        | 131/920 [19:03<1:50:15,  8.39s/it]"
        step_match = re.search(r'(\d+)%.*?\|\s*(\d+)/(\d+)\s*\[.*?<(.*?),\s*([\d.]+)s/it\]', line)
        if step_match and metrics['current_step'] == 0:
            metrics['current_step'] = int(step_match.group(2))
            metrics['total_steps'] = int(step_match.group(3))
            metrics['time_per_step'] = float(step_match.group(5))
            metrics['last_update'] = line.strip()

        # Extract loss: {'loss': 0.2036, 'grad_norm': 0.025504810735583305, 'learning_rate': 0.00019373965203110913, 'epoch': 1.42}
        loss_match = re.search(r"'loss':\s*([\d.]+).*?'grad_norm':\s*([\d.]+).*?'learning_rate':\s*([\d.e-]+).*?'epoch':\s*([\d.]+)", line)
        if loss_match:
            metrics['train_loss'] = float(loss_match.group(1))
            metrics['grad_norm'] = float(loss_match.group(2))
            metrics['learning_rate'] = float(loss_match.group(3))
            metrics['current_epoch'] = float(loss_match.group(4))

        # Extract eval loss: {'eval_loss': 0.050754938274621964, ...}
        eval_match = re.search(r"'eval_loss':\s*([\d.]+)", line)
        if eval_match and metrics['eval_loss'] is None:
            metrics['eval_loss'] = float(eval_match.group(1))

    return metrics

def find_latest_run():
    """Find the latest training run directory."""
    outputs_dir = Path.home() / "RCT-Clean-Experiment" / "outputs"
    if not outputs_dir.exists():
        return None

    runs = sorted(outputs_dir.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0] if runs else None

def get_checkpoints(run_dir):
    """List saved checkpoints."""
    if not run_dir:
        return []

    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return []

    checkpoints = sorted(
        [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0
    )
    return checkpoints

def format_time(seconds):
    """Format seconds into readable time."""
    if seconds is None:
        return "Unknown"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def print_dashboard(metrics, is_running, run_dir):
    """Print formatted dashboard."""
    clear_screen()

    # Header
    print("═" * 80)
    print("🔥 RCT TRAINING MONITOR - Relational Coherence Training Dashboard 🔥".center(80))
    print("═" * 80)
    print()

    # Status
    status_symbol = "🟢 RUNNING" if is_running else "🔴 STOPPED"
    print(f"Status: {status_symbol}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    if metrics:
        # Progress
        print("─" * 80)
        print("📊 TRAINING PROGRESS")
        print("─" * 80)

        progress_pct = (metrics['current_step'] / metrics['total_steps'] * 100) if metrics['total_steps'] > 0 else 0
        bar_width = 50
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        print(f"Step:  {metrics['current_step']:4d} / {metrics['total_steps']:4d}  [{bar}] {progress_pct:.1f}%")
        print(f"Epoch: {metrics['current_epoch']:.2f} / {metrics['total_epochs']}")
        print()

        # Loss Metrics
        print("─" * 80)
        print("📉 LOSS METRICS")
        print("─" * 80)
        if metrics['train_loss'] is not None:
            print(f"Training Loss:   {metrics['train_loss']:.4f}")
        if metrics['eval_loss'] is not None:
            print(f"Evaluation Loss: {metrics['eval_loss']:.4f}  {'✨ Excellent coherence!' if metrics['eval_loss'] < 0.06 else ''}")
        if metrics['grad_norm'] is not None:
            print(f"Gradient Norm:   {metrics['grad_norm']:.6f}")
        if metrics['learning_rate'] is not None:
            print(f"Learning Rate:   {metrics['learning_rate']:.6f}")
        print()

        # Time Estimates
        print("─" * 80)
        print("⏱️  TIME ESTIMATES")
        print("─" * 80)
        if metrics['time_per_step']:
            remaining_steps = metrics['total_steps'] - metrics['current_step']
            remaining_seconds = remaining_steps * metrics['time_per_step']
            elapsed_seconds = metrics['current_step'] * metrics['time_per_step']

            print(f"Time per Step:   {metrics['time_per_step']:.2f}s")
            print(f"Elapsed Time:    {format_time(elapsed_seconds)}")
            print(f"Remaining Time:  {format_time(remaining_seconds)}")

            if metrics['start_time']:
                eta = datetime.now() + timedelta(seconds=remaining_seconds)
                print(f"ETA:             {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Checkpoints
        if run_dir:
            checkpoints = get_checkpoints(run_dir)
            print("─" * 80)
            print(f"💾 CHECKPOINTS ({len(checkpoints)} saved)")
            print("─" * 80)
            if checkpoints:
                for cp in checkpoints[-5:]:  # Last 5
                    step_num = cp.name.split("-")[1]
                    size = sum(f.stat().st_size for f in cp.rglob('*') if f.is_file()) / (1024**2)
                    print(f"  • checkpoint-{step_num:>4s}  ({size:.1f} MB)")
            else:
                print("  No checkpoints saved yet (saves every 100 steps)")
            print()

        # Output location
        print("─" * 80)
        print("📂 OUTPUT LOCATION")
        print("─" * 80)
        if run_dir:
            print(f"  {run_dir}")
        print()

    else:
        print("⚠️  No training metrics found. Training may not have started yet.")
        print()

    # Footer
    print("═" * 80)
    print("Press Ctrl+C to exit monitor (training continues in background)".center(80))
    print("═" * 80)

def main():
    """Main monitoring loop."""
    log_path = Path.home() / "RCT-Clean-Experiment" / "rct_training.log"

    try:
        while True:
            is_running, _ = get_process_status()
            run_dir = find_latest_run()
            metrics = parse_log_file(log_path)

            print_dashboard(metrics, is_running, run_dir)

            if not is_running and metrics and metrics['current_step'] >= metrics['total_steps']:
                print("\n✅ Training complete!")
                break

            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped. Training continues in background.")
        print(f"\nTo view log: tail -f {log_path}")
        sys.exit(0)

if __name__ == "__main__":
    main()
