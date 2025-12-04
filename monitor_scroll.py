#!/usr/bin/env python3
"""
Scrolling RCT Training Monitor
Real-time scrolling updates for Relational Coherence Training
"""

import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import time
import sys

# ANSI color codes
class C:
    R = '\033[0m'  # Reset
    B = '\033[1m'  # Bold
    D = '\033[2m'  # Dim

    # Colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'

def clear():
    """Clear screen."""
    print('\033[2J\033[H', end='')

def get_process_status():
    """Check if training is running."""
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'train_rct.py' in line and 'grep' not in line:
                return True
        return False
    except:
        return False

def parse_log(log_path):
    """Extract latest metrics from log."""
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    m = {
        'step': 0, 'total_steps': 920, 'epoch': 0.0,
        'train_loss': None, 'eval_loss': None,
        'grad_norm': None, 'lr': None, 'time_per_step': None,
        'loss_history': []
    }

    # Parse all loss values for trend
    for line in lines:
        loss_match = re.search(r"'loss':\s*([\d.]+).*?'grad_norm':\s*([\d.]+).*?'learning_rate':\s*([\d.e-]+).*?'epoch':\s*([\d.]+)", line)
        if loss_match:
            m['loss_history'].append(float(loss_match.group(1)))

    # Parse latest from recent lines
    for line in reversed(lines[-100:]):
        step_match = re.search(r'(\d+)%.*?\|\s*(\d+)/(\d+)\s*\[.*?<(.*?),\s*([\d.]+)s/it\]', line)
        if step_match and m['step'] == 0:
            m['step'] = int(step_match.group(2))
            m['total_steps'] = int(step_match.group(3))
            m['time_per_step'] = float(step_match.group(5))

        if m['loss_history'] and m['train_loss'] is None:
            m['train_loss'] = m['loss_history'][-1]

        loss_match2 = re.search(r"'loss':\s*([\d.]+).*?'grad_norm':\s*([\d.]+).*?'learning_rate':\s*([\d.e-]+).*?'epoch':\s*([\d.]+)", line)
        if loss_match2 and m['grad_norm'] is None:
            m['grad_norm'] = float(loss_match2.group(2))
            m['lr'] = float(loss_match2.group(3))
            m['epoch'] = float(loss_match2.group(4))

        eval_match = re.search(r"'eval_loss':\s*([\d.]+)", line)
        if eval_match and m['eval_loss'] is None:
            m['eval_loss'] = float(eval_match.group(1))

    return m

def get_checkpoints(run_dir):
    """Get checkpoint count."""
    if not run_dir:
        return 0
    cp_dir = run_dir / "checkpoints"
    if not cp_dir.exists():
        return 0
    return len([d for d in cp_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])

def find_latest_run():
    """Find latest run dir."""
    outputs = Path.home() / "RCT-Clean-Experiment" / "outputs"
    if not outputs.exists():
        return None
    runs = sorted(outputs.glob("run_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    return runs[0] if runs else None

def format_time(secs):
    """Format seconds."""
    if not secs or secs < 0:
        return "?"
    h, m, s = int(secs // 3600), int((secs % 3600) // 60), int(secs % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def print_update(m, running, run_dir, first=False):
    """Print scrolling update."""
    c = C
    ts = datetime.now().strftime('%H:%M:%S')

    if first:
        clear()
        print(f"{c.B}{c.CYAN}{'═' * 80}{c.R}")
        print(f"{c.B}{c.YELLOW}{'🔥 RCT TRAINING MONITOR - Live Scrolling Feed 🔥'.center(80)}{c.R}")
        print(f"{c.B}{c.CYAN}{'═' * 80}{c.R}")
        print(f"{c.D}Training updates scroll below • Ctrl+C to exit (training continues){c.R}\n")

    if not m:
        print(f"{c.RED}[{ts}] ⚠  Waiting for training data...{c.R}")
        return

    # Status symbol
    status = f"{c.GREEN}●{c.R}" if running else f"{c.RED}○{c.R}"

    # Progress bar (mini)
    pct = (m['step'] / m['total_steps'] * 100) if m['total_steps'] > 0 else 0
    bar_w = 25
    filled = int(bar_w * pct / 100)
    bar = f"{c.GREEN}{'█' * filled}{c.GRAY}{'░' * (bar_w - filled)}{c.R}"

    # Loss trend
    if len(m['loss_history']) > 5:
        recent = m['loss_history'][-5:]
        trend = "↓" if recent[-1] < recent[0] else "↑" if recent[-1] > recent[0] else "─"
        t_color = c.GREEN if trend == "↓" else c.RED if trend == "↑" else c.YELLOW
    else:
        trend, t_color = "─", c.YELLOW

    # Main update line
    print(f"{c.WHITE}[{ts}]{c.R} {status} "
          f"{c.YELLOW}Step {m['step']:4d}/{m['total_steps']}{c.R} "
          f"{bar} {c.CYAN}{pct:5.1f}%{c.R} "
          f"{c.D}│{c.R} "
          f"{c.YELLOW}E{m['epoch']:.2f}{c.R}")

    # Metrics line
    parts = []

    if m['train_loss']:
        parts.append(f"{c.WHITE}Loss:{c.R} {c.YELLOW}{m['train_loss']:.4f}{c.R} {t_color}{trend}{c.R}")

    if m['eval_loss']:
        eval_icon = "✨" if m['eval_loss'] < 0.06 else "•"
        parts.append(f"{c.WHITE}Eval:{c.R} {c.CYAN}{m['eval_loss']:.4f}{c.R} {eval_icon}")

    if m['grad_norm']:
        parts.append(f"{c.WHITE}Grad:{c.R} {c.MAGENTA}{m['grad_norm']:.4f}{c.R}")

    if m['lr']:
        parts.append(f"{c.WHITE}LR:{c.R} {c.CYAN}{m['lr']:.6f}{c.R}")

    if parts:
        print(f"         {c.D}│{c.R} " + f" {c.D}•{c.R} ".join(parts))

    # Time estimate line
    if m['time_per_step']:
        remaining_steps = m['total_steps'] - m['step']
        remaining_secs = remaining_steps * m['time_per_step']
        elapsed_secs = m['step'] * m['time_per_step']
        eta = datetime.now() + timedelta(seconds=remaining_secs)

        print(f"         {c.D}│ "
              f"⏱ {m['time_per_step']:.1f}s/step • "
              f"Elapsed: {format_time(elapsed_secs)} • "
              f"Remaining: {format_time(remaining_secs)} • "
              f"ETA: {c.GREEN}{eta.strftime('%H:%M')}{c.D}{c.R}")

    # Checkpoint info (occasionally)
    cp_count = get_checkpoints(run_dir)
    if cp_count > 0 and m['step'] % 10 == 0:  # Show every 10 updates
        next_cp = ((m['step'] // 100) + 1) * 100
        if next_cp <= m['total_steps']:
            steps_to_cp = next_cp - m['step']
            time_to_cp = steps_to_cp * m['time_per_step'] if m['time_per_step'] else 0
            print(f"         {c.D}│ 💾 {cp_count} checkpoints saved • "
                  f"Next at step {next_cp} (~{format_time(time_to_cp)}){c.R}")

    print()  # Blank line between updates

def main():
    """Main loop."""
    log_path = Path.home() / "RCT-Clean-Experiment" / "rct_training.log"
    first = True

    try:
        while True:
            running = get_process_status()
            run_dir = find_latest_run()
            metrics = parse_log(log_path)

            print_update(metrics, running, run_dir, first=first)
            first = False

            if not running and metrics and metrics['step'] >= metrics['total_steps']:
                print(f"\n{C.B}{C.GREEN}✅ TRAINING COMPLETE!{C.R}\n")
                break

            time.sleep(5)  # Update every 5 seconds

    except KeyboardInterrupt:
        print(f"\n\n{C.YELLOW}👋 Monitor stopped. Training continues in background.{C.R}")
        print(f"{C.D}Log: tail -f {log_path}{C.R}\n")
        sys.exit(0)

if __name__ == "__main__":
    main()
