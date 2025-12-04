#!/usr/bin/env python3
"""
Enhanced RCT Training Monitor Dashboard
Real-time monitoring with live feed and advanced metrics
"""

import os
import re
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys
from collections import deque

# ANSI color codes
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

def clear_screen():
    """Clear terminal screen (only used on startup)."""
    print('\033[2J\033[H', end='')

def get_terminal_size():
    """Get terminal width and height."""
    try:
        size = os.get_terminal_size()
        return size.columns, size.lines
    except:
        return 120, 40

def get_process_status():
    """Check if training is running."""
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'train_rct.py' in line and 'grep' not in line:
                return True, line
        return False, None
    except:
        return False, None

def parse_log_file(log_path):
    """Extract metrics from training log with history."""
    if not os.path.exists(log_path):
        return None, []

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
        'samples_per_second': None,
        'last_update': None,
        'start_time': None,
        'trainable_params': None,
        'total_params': None
    }

    loss_history = []

    # Find trainable params
    for line in lines:
        if "Trainable parameters:" in line:
            match = re.search(r'([\d,]+)\s*/\s*([\d,]+)', line)
            if match:
                metrics['trainable_params'] = match.group(1)
                metrics['total_params'] = match.group(2)
        if "Starting training" in line:
            metrics['start_time'] = datetime.now() - timedelta(seconds=len(lines) * 0.5)
            break

    # Parse all loss values for history
    for line in lines:
        loss_match = re.search(r"'loss':\s*([\d.]+).*?'grad_norm':\s*([\d.]+).*?'learning_rate':\s*([\d.e-]+).*?'epoch':\s*([\d.]+)", line)
        if loss_match:
            loss_history.append({
                'loss': float(loss_match.group(1)),
                'grad_norm': float(loss_match.group(2)),
                'lr': float(loss_match.group(3)),
                'epoch': float(loss_match.group(4))
            })

    # Parse latest metrics from recent lines
    for line in reversed(lines[-100:]):
        step_match = re.search(r'(\d+)%.*?\|\s*(\d+)/(\d+)\s*\[.*?<(.*?),\s*([\d.]+)s/it\]', line)
        if step_match and metrics['current_step'] == 0:
            metrics['current_step'] = int(step_match.group(2))
            metrics['total_steps'] = int(step_match.group(3))
            metrics['time_per_step'] = float(step_match.group(5))

        # Latest loss
        if loss_history and metrics['train_loss'] is None:
            metrics['train_loss'] = loss_history[-1]['loss']
            metrics['grad_norm'] = loss_history[-1]['grad_norm']
            metrics['learning_rate'] = loss_history[-1]['lr']
            metrics['current_epoch'] = loss_history[-1]['epoch']

        # Eval loss
        eval_match = re.search(r"'eval_loss':\s*([\d.]+).*?'eval_samples_per_second':\s*([\d.]+)", line)
        if eval_match and metrics['eval_loss'] is None:
            metrics['eval_loss'] = float(eval_match.group(1))
            metrics['samples_per_second'] = float(eval_match.group(2))

    return metrics, loss_history

def get_recent_logs(log_path, num_lines=10):
    """Get recent log lines."""
    if not os.path.exists(log_path):
        return []

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()

        # Filter out progress bars and get meaningful lines
        recent = []
        for line in reversed(lines[-50:]):
            if any(x in line for x in ['loss', 'eval', 'Step', 'Epoch', 'Saving', 'checkpoint']):
                recent.append(line.strip())
            if len(recent) >= num_lines:
                break

        return list(reversed(recent))
    except:
        return []

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
    if seconds is None or seconds < 0:
        return "Unknown"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def draw_ascii_chart(values, width=60, height=10):
    """Draw ASCII chart of values."""
    if not values:
        return ["No data"]

    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val if max_val > min_val else 1

    # Normalize values to chart height
    normalized = [int((v - min_val) / range_val * (height - 1)) for v in values]

    # Create chart
    chart = []
    for y in range(height - 1, -1, -1):
        line = ""
        for x, norm_val in enumerate(normalized):
            if norm_val >= y:
                line += "█"
            elif norm_val == y - 1:
                line += "▄"
            else:
                line += " "

        # Add y-axis label
        val = min_val + (y / (height - 1)) * range_val
        chart.append(f"{val:6.4f} │{line}")

    # Add x-axis
    chart.append("       └" + "─" * len(normalized))

    return chart

def print_dashboard(metrics, loss_history, is_running, run_dir, recent_logs, first_run=False):
    """Print enhanced dashboard."""
    width, height = get_terminal_size()
    c = Colors

    if first_run:
        clear_screen()
        # Print header only on first run
        print(f"{c.BOLD}{c.BRIGHT_CYAN}╔{'═' * (width - 2)}╗{c.RESET}")
        print(f"{c.BOLD}{c.BRIGHT_CYAN}║{c.BRIGHT_YELLOW}{'🔥 RCT TRAINING MONITOR - Relational Coherence Training 🔥'.center(width - 2)}{c.BRIGHT_CYAN}║{c.RESET}")
        print(f"{c.BOLD}{c.BRIGHT_CYAN}╚{'═' * (width - 2)}╝{c.RESET}")
        print()
        print(f"{c.DIM}Updates scroll below (Ctrl+C to exit){c.RESET}")
        print(f"{c.BRIGHT_CYAN}{'─' * width}{c.RESET}\n")

    # Scrolling update format
    timestamp = datetime.now().strftime('%H:%M:%S')

    if not metrics:
        print(f"{c.BRIGHT_RED}[{timestamp}] ⚠️  No training data yet{c.RESET}")
        return

    # Compact scrolling update
    status_color = c.BRIGHT_GREEN if is_running else c.BRIGHT_RED
    status_symbol = "●" if is_running else "○"

    # Progress bar (compact)
    progress_pct = (metrics['current_step'] / metrics['total_steps'] * 100) if metrics['total_steps'] > 0 else 0
    bar_width = 30
    filled = int(bar_width * progress_pct / 100)
    bar = f"{c.BRIGHT_GREEN}{'█' * filled}{c.DIM}{'░' * (bar_width - filled)}{c.RESET}"

    # Main update line
    print(f"{c.BRIGHT_WHITE}[{timestamp}]{c.RESET} "
          f"{status_color}{status_symbol}{c.RESET} "
          f"{c.BRIGHT_YELLOW}Step {metrics['current_step']:4d}/{metrics['total_steps']}{c.RESET} "
          f"{bar} "
          f"{c.BRIGHT_CYAN}{progress_pct:5.1f}%{c.RESET} "
          f"{c.DIM}│{c.RESET} "
          f"{c.BRIGHT_YELLOW}Epoch {metrics['current_epoch']:.2f}{c.RESET}")

    # Metrics line (compact, single line)
    metrics_parts = []

    if metrics['train_loss'] is not None:
        # Loss trend
        if len(loss_history) > 1:
            recent_losses = [h['loss'] for h in loss_history[-10:]]
            trend = "↓" if recent_losses[-1] < recent_losses[0] else "↑"
            trend_color = c.BRIGHT_GREEN if trend == "↓" else c.BRIGHT_RED
        else:
            trend = "─"
            trend_color = c.YELLOW

        metrics_parts.append(f"{c.BRIGHT_WHITE}Loss:{c.RESET} {c.BRIGHT_YELLOW}{metrics['train_loss']:.4f}{c.RESET}{trend_color}{trend}{c.RESET}")

        # Progress bar
        progress_pct = (metrics['current_step'] / metrics['total_steps'] * 100) if metrics['total_steps'] > 0 else 0
        bar_width = col_width - 20
        filled = int(bar_width * progress_pct / 100)
        bar = f"{c.BRIGHT_GREEN}{'█' * filled}{c.DIM}{'░' * (bar_width - filled)}{c.RESET}"

        print(f"{c.BRIGHT_WHITE}Step:  {c.BRIGHT_YELLOW}{metrics['current_step']:4d}{c.RESET} / "
              f"{metrics['total_steps']:4d}  {bar} {c.BRIGHT_CYAN}{progress_pct:5.1f}%{c.RESET}")
        print(f"{c.BRIGHT_WHITE}Epoch: {c.BRIGHT_YELLOW}{metrics['current_epoch']:5.2f}{c.RESET} / "
              f"{metrics['total_epochs']}")

        if metrics['trainable_params']:
            print(f"{c.BRIGHT_WHITE}Params:{c.BRIGHT_MAGENTA} {metrics['trainable_params']}{c.RESET} / "
                  f"{metrics['total_params']} trainable")
        print()

        # Loss metrics with trend
        print(f"{c.BOLD}{c.BRIGHT_BLUE}┌─ LOSS METRICS {'─' * (col_width - 16)}┐{c.RESET}")

        if metrics['train_loss'] is not None:
            # Show trend
            if len(loss_history) > 1:
                recent_losses = [h['loss'] for h in loss_history[-10:]]
                trend = "↓" if recent_losses[-1] < recent_losses[0] else "↑"
                trend_color = c.BRIGHT_GREEN if trend == "↓" else c.BRIGHT_RED
            else:
                trend = "─"
                trend_color = c.YELLOW

            print(f"{c.BRIGHT_WHITE}Training Loss:   {c.BRIGHT_YELLOW}{metrics['train_loss']:.6f} "
                  f"{trend_color}{trend}{c.RESET}")

        if metrics['eval_loss'] is not None:
            eval_status = "✨ Excellent!" if metrics['eval_loss'] < 0.06 else "Good" if metrics['eval_loss'] < 0.1 else "Training..."
            eval_color = c.BRIGHT_GREEN if metrics['eval_loss'] < 0.06 else c.BRIGHT_CYAN
            print(f"{c.BRIGHT_WHITE}Eval Loss:       {eval_color}{metrics['eval_loss']:.6f} {c.DIM}{eval_status}{c.RESET}")

        if metrics['grad_norm'] is not None:
            print(f"{c.BRIGHT_WHITE}Gradient Norm:   {c.BRIGHT_MAGENTA}{metrics['grad_norm']:.6f}{c.RESET}")

        if metrics['learning_rate'] is not None:
            print(f"{c.BRIGHT_WHITE}Learning Rate:   {c.BRIGHT_CYAN}{metrics['learning_rate']:.8f}{c.RESET}")

        if metrics['samples_per_second'] is not None:
            print(f"{c.BRIGHT_WHITE}Throughput:      {c.BRIGHT_GREEN}{metrics['samples_per_second']:.2f} samples/sec{c.RESET}")
        print()

        # Time estimates
        print(f"{c.BOLD}{c.BRIGHT_BLUE}┌─ TIME ANALYSIS {'─' * (col_width - 17)}┐{c.RESET}")

        if metrics['time_per_step']:
            remaining_steps = metrics['total_steps'] - metrics['current_step']
            remaining_seconds = remaining_steps * metrics['time_per_step']
            elapsed_seconds = metrics['current_step'] * metrics['time_per_step']

            print(f"{c.BRIGHT_WHITE}Step Duration:   {c.BRIGHT_YELLOW}{metrics['time_per_step']:.2f}s{c.RESET}")
            print(f"{c.BRIGHT_WHITE}Elapsed:         {c.BRIGHT_CYAN}{format_time(elapsed_seconds)}{c.RESET}")
            print(f"{c.BRIGHT_WHITE}Remaining:       {c.BRIGHT_MAGENTA}{format_time(remaining_seconds)}{c.RESET}")

            if metrics['start_time']:
                eta = datetime.now() + timedelta(seconds=remaining_seconds)
                print(f"{c.BRIGHT_WHITE}ETA:             {c.BRIGHT_GREEN}{eta.strftime('%H:%M:%S')}{c.RESET} "
                      f"{c.DIM}({eta.strftime('%b %d')}){c.RESET}")
        print()

        # Loss history chart
        if len(loss_history) > 5:
            print(f"{c.BOLD}{c.BRIGHT_BLUE}┌─ TRAINING LOSS HISTORY {'─' * (col_width - 25)}┐{c.RESET}")
            recent_losses = [h['loss'] for h in loss_history[-60:]]  # Last 60 data points
            chart = draw_ascii_chart(recent_losses, width=min(60, col_width - 10), height=8)
            for line in chart:
                print(f"{c.BRIGHT_YELLOW}{line}{c.RESET}")
            print()

        # Checkpoints
        if run_dir:
            checkpoints = get_checkpoints(run_dir)
            print(f"{c.BOLD}{c.BRIGHT_BLUE}┌─ CHECKPOINTS ({len(checkpoints)} saved) {'─' * (col_width - 26)}┐{c.RESET}")

            if checkpoints:
                for cp in checkpoints[-4:]:
                    step_num = cp.name.split("-")[1]
                    size = sum(f.stat().st_size for f in cp.rglob('*') if f.is_file()) / (1024**2)
                    mtime = datetime.fromtimestamp(cp.stat().st_mtime)
                    print(f"  {c.BRIGHT_GREEN}●{c.RESET} checkpoint-{c.BRIGHT_YELLOW}{step_num:>4s}{c.RESET}  "
                          f"{c.DIM}{size:6.1f} MB  {mtime.strftime('%H:%M:%S')}{c.RESET}")

                next_checkpoint = ((metrics['current_step'] // 100) + 1) * 100
                if next_checkpoint <= metrics['total_steps']:
                    steps_until = next_checkpoint - metrics['current_step']
                    time_until = steps_until * metrics['time_per_step'] if metrics['time_per_step'] else 0
                    print(f"\n  {c.DIM}Next: checkpoint-{next_checkpoint} in ~{format_time(time_until)}{c.RESET}")
            else:
                print(f"  {c.DIM}Saving every 100 steps...{c.RESET}")
            print()

        # Live feed
        print(f"{c.BOLD}{c.BRIGHT_BLUE}┌─ LIVE TRAINING FEED {'─' * (width - 23)}┐{c.RESET}")

        if recent_logs:
            for log in recent_logs[-8:]:  # Last 8 lines
                # Color code different types of logs
                if 'eval' in log.lower():
                    print(f"  {c.BRIGHT_CYAN}▶{c.RESET} {c.DIM}{log[:width-5]}{c.RESET}")
                elif 'loss' in log.lower():
                    print(f"  {c.BRIGHT_YELLOW}▶{c.RESET} {c.DIM}{log[:width-5]}{c.RESET}")
                elif 'saving' in log.lower() or 'checkpoint' in log.lower():
                    print(f"  {c.BRIGHT_GREEN}▶{c.RESET} {c.DIM}{log[:width-5]}{c.RESET}")
                else:
                    print(f"  {c.BRIGHT_WHITE}▶{c.RESET} {c.DIM}{log[:width-5]}{c.RESET}")
        else:
            print(f"  {c.DIM}Waiting for log updates...{c.RESET}")
        print()

        # Output location
        print(f"{c.BOLD}{c.BRIGHT_BLUE}┌─ OUTPUT {'─' * (width - 10)}┐{c.RESET}")
        if run_dir:
            print(f"  {c.DIM}{str(run_dir)[:width-4]}{c.RESET}")
        print()

    else:
        print(f"{c.BRIGHT_RED}⚠️  No training metrics found.{c.RESET}")
        print(f"{c.DIM}Training may not have started yet.{c.RESET}")
        print()

    # Footer
    print(f"{c.BOLD}{c.BRIGHT_CYAN}{'─' * width}{c.RESET}")
    print(f"{c.DIM}Press Ctrl+C to exit monitor (training continues in background){c.RESET}".center(width))
    print(f"{c.DIM}Refreshing every 3 seconds...{c.RESET}".center(width))
    print(f"{c.BOLD}{c.BRIGHT_CYAN}{'─' * width}{c.RESET}")

def main():
    """Main monitoring loop."""
    log_path = Path.home() / "RCT-Clean-Experiment" / "rct_training.log"

    try:
        while True:
            is_running, _ = get_process_status()
            run_dir = find_latest_run()
            metrics, loss_history = parse_log_file(log_path)
            recent_logs = get_recent_logs(log_path, num_lines=12)

            print_dashboard(metrics, loss_history, is_running, run_dir, recent_logs)

            if not is_running and metrics and metrics['current_step'] >= metrics['total_steps']:
                print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✅ TRAINING COMPLETE!{Colors.RESET}\n")
                break

            time.sleep(3)  # Update every 3 seconds

    except KeyboardInterrupt:
        print(f"\n\n{Colors.BRIGHT_YELLOW}👋 Monitor stopped. Training continues in background.{Colors.RESET}")
        print(f"\n{Colors.DIM}To view log: tail -f {log_path}{Colors.RESET}\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
