#!/usr/bin/env python3
"""
Analyze training logs to extract timing information per epoch.

Usage:
    python analyze_training_time.py <log_file>
"""

import re
import sys
from datetime import datetime
from collections import defaultdict


def parse_log_file(log_file):
    """Parse training log file and extract timing information."""
    
    epoch_data = defaultdict(lambda: {
        'iters': [],
        'times': [],
        'data_times': [],
        'start_time': None,
        'workflow_start': None,
        'end_time': None,
        'checkpoint_time': None,
        'is_complete': False
    })
    
    # Patterns to match
    epoch_pattern = r'Epoch \[(\d+)\]\[(\d+)/(\d+)\]'
    time_pattern = r'time: ([\d.]+)'
    data_time_pattern = r'data_time: ([\d.]+)'
    timestamp_pattern = r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})'
    workflow_pattern = r'workflow: \[\(\'train\', 1\)\]'
    checkpoint_pattern = r'Saving checkpoint at (\d+) epochs'
    
    total_start_time = None
    total_end_time = None
    workflow_start_times = {}  # Track workflow start time for each epoch
    current_epoch = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Extract timestamp
            ts_match = re.match(timestamp_pattern, line)
            if ts_match:
                current_time = datetime.strptime(ts_match.group(1), '%Y-%m-%d %H:%M:%S')
                if total_start_time is None:
                    total_start_time = current_time
                total_end_time = current_time
            
            # Detect workflow start (epoch start)
            if workflow_pattern in line and ts_match:
                # Increment epoch counter (workflow marks the beginning of next epoch)
                if current_epoch is None:
                    current_epoch = 1
                else:
                    current_epoch += 1
                workflow_start_times[current_epoch] = current_time
                epoch_data[current_epoch]['workflow_start'] = current_time
            
            # Detect checkpoint save (epoch end)
            checkpoint_match = re.search(checkpoint_pattern, line)
            if checkpoint_match and ts_match:
                epoch = int(checkpoint_match.group(1))
                epoch_data[epoch]['checkpoint_time'] = current_time
                epoch_data[epoch]['is_complete'] = True
                
                # Set start time from workflow
                if epoch in workflow_start_times:
                    epoch_data[epoch]['start_time'] = workflow_start_times[epoch]
            
            # Extract epoch, iteration, and timing info
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                iteration = int(epoch_match.group(2))
                total_iters = int(epoch_match.group(3))
                
                # Extract time
                time_match = re.search(time_pattern, line)
                if time_match:
                    iter_time = float(time_match.group(1))
                    epoch_data[epoch]['times'].append(iter_time)
                    epoch_data[epoch]['iters'].append(iteration)
                
                # Extract data_time
                data_time_match = re.search(data_time_pattern, line)
                if data_time_match:
                    data_time = float(data_time_match.group(1))
                    epoch_data[epoch]['data_times'].append(data_time)
                
                # Update end time for epoch (last iteration timestamp)
                if ts_match:
                    epoch_data[epoch]['end_time'] = current_time
                    
                # Store total iterations
                epoch_data[epoch]['total_iters'] = total_iters
    
    return epoch_data, total_start_time, total_end_time


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{seconds/60:.2f} minutes ({hours:.2f} hours)"


def analyze_epochs(epoch_data, total_start_time, total_end_time):
    """Analyze and print epoch timing statistics."""
    
    if not epoch_data:
        print("No epoch data found in log file.")
        return
    
    print("\n" + "="*80)
    print("TRAINING TIME ANALYSIS")
    print("="*80)
    
    # Separate complete and incomplete epochs
    complete_epochs = {k: v for k, v in epoch_data.items() if v['is_complete']}
    incomplete_epochs = {k: v for k, v in epoch_data.items() if not v['is_complete']}
    
    # Per-epoch analysis (complete epochs only)
    if complete_epochs:
        print("\nComplete Epoch Statistics:")
        print("-" * 80)
    
    all_iter_times = []
    all_data_times = []
    epoch_durations = []
    
    for epoch in sorted(complete_epochs.keys()):
        data = complete_epochs[epoch]
        
        if not data['times']:
            continue
        
        avg_time = sum(data['times']) / len(data['times'])
        avg_data_time = sum(data['data_times']) / len(data['data_times']) if data['data_times'] else 0
        compute_time = avg_time - avg_data_time
        
        all_iter_times.extend(data['times'])
        all_data_times.extend(data['data_times'])
        
        # Calculate actual epoch duration: workflow start → checkpoint save
        if data['start_time'] and data['checkpoint_time']:
            epoch_duration = (data['checkpoint_time'] - data['start_time']).total_seconds()
            epoch_durations.append(epoch_duration)
        else:
            # Fallback: estimate from average iter time
            total_iters = data.get('total_iters', len(data['times']))
            epoch_duration = avg_time * total_iters
            epoch_durations.append(epoch_duration)
        
        print(f"\nEpoch {epoch}:")
        print(f"  Status: ✓ Complete")
        print(f"  Iterations logged: {len(data['times'])}/{data.get('total_iters', 'unknown')}")
        print(f"  Avg iter time: {avg_time:.4f} s/iter")
        print(f"  Avg data time: {avg_data_time:.4f} s/iter")
        print(f"  Compute time: {compute_time:.4f} s/iter")
        print(f"  Epoch duration: {format_time(epoch_duration)}")
        
        if data['start_time']:
            print(f"  Start: {data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        if data['checkpoint_time']:
            print(f"  Checkpoint saved: {data['checkpoint_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show incomplete epochs separately
    if incomplete_epochs:
        print("\n" + "-" * 80)
        print("Incomplete Epoch Statistics (not included in averages):")
        print("-" * 80)
        
        for epoch in sorted(incomplete_epochs.keys()):
            data = incomplete_epochs[epoch]
            
            if not data['times']:
                continue
            
            avg_time = sum(data['times']) / len(data['times'])
            avg_data_time = sum(data['data_times']) / len(data['data_times']) if data['data_times'] else 0
            
            print(f"\nEpoch {epoch}:")
            print(f"  Status: ⚠ Incomplete")
            print(f"  Iterations logged: {len(data['times'])}/{data.get('total_iters', 'unknown')}")
            print(f"  Avg iter time: {avg_time:.4f} s/iter")
            print(f"  Avg data time: {avg_data_time:.4f} s/iter")
            
            if data['start_time']:
                print(f"  Start: {data['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            if data['end_time']:
                print(f"  Last iteration: {data['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overall summary (complete epochs only)
    print("\n" + "="*80)
    print("SUMMARY (Complete Epochs Only)")
    print("="*80)
    
    if not complete_epochs:
        print("\n⚠️  No complete epochs found in log file.")
        print("Training may still be in progress or was interrupted.")
        print()
        return
    
    if epoch_durations:
        total_duration = sum(epoch_durations)
        avg_epoch_time = total_duration / len(epoch_durations)
        
        print(f"Total training time: {format_time(total_duration)}")
        print(f"Average time per epoch: {format_time(avg_epoch_time)}")
    
    completed_epochs = len(complete_epochs)
    total_epochs = max(epoch_data.keys()) if epoch_data else 0
    print(f"Completed epochs: {completed_epochs}/{total_epochs}")
    
    if all_iter_times:
        overall_avg_iter = sum(all_iter_times) / len(all_iter_times)
        overall_avg_data = sum(all_data_times) / len(all_data_times) if all_data_times else 0
        overall_compute = overall_avg_iter - overall_avg_data
        
        print(f"Overall avg iter time: {overall_avg_iter:.4f} s/iter")
        print(f"Overall avg data time: {overall_avg_data:.4f} s/iter")
        print(f"Compute time (iter - data): {overall_compute:.4f} s/iter")
    
    print("\n" + "="*80)
    
    # Efficiency metrics
    if all_iter_times and all_data_times:
        data_loading_ratio = (overall_avg_data / overall_avg_iter) * 100
        compute_ratio = (overall_compute / overall_avg_iter) * 100
        
        print("\nEfficiency Metrics:")
        print(f"  Data loading: {data_loading_ratio:.1f}% of iteration time")
        print(f"  Computation: {compute_ratio:.1f}% of iteration time")
        
        if data_loading_ratio > 20:
            print("\n⚠️  Warning: Data loading takes >20% of iteration time.")
            print("   Consider using more data workers or optimizing data pipeline.")
    
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_time.py <log_file>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    try:
        epoch_data, total_start_time, total_end_time = parse_log_file(log_file)
        analyze_epochs(epoch_data, total_start_time, total_end_time)
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
