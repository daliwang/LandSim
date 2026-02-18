#!/usr/bin/env python3
"""
PyTorch Profiler wrapper for train_model.py
Run this script without modifying train_model.py source code
"""

import os
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_profiler_output_dir() -> Path:
    """Create profiler output directory"""
    out_dir = Path("pytorch_profiler_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def main():
    """Main function: run train_model.py under profiler environment"""
    
    # Detect available device activities
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
        print(f"Detected CUDA/ROCm device, will analyze both CPU and GPU performance")
    else:
        print("Only CPU detected, will analyze CPU performance only")
    
    # Prepare output directory
    output_dir = setup_profiler_output_dir()
    trace_path = output_dir / "train_model_trace.json"
    stats_path = output_dir / "train_model_stats.txt"
    
    print(f"Starting PyTorch Profiler analysis of train_model.py...")
    print(f"Output directory: {output_dir}")
    
    try:
        # Check if TensorBoard logs already exist
        tensorboard_logs_dir = output_dir / "tensorboard_logs"
        existing_traces = list(tensorboard_logs_dir.glob("*.pt.trace.json")) if tensorboard_logs_dir.exists() else []
        
        if existing_traces:
            print(f"✓ Found existing TensorBoard trace files:")
            for trace_file in existing_traces:
                print(f"  - {trace_file.name} ({trace_file.stat().st_size / (1024**3):.2f} GB)")
            print(f"✓ Skipping training and profiler generation (using existing traces)")
        else:
            print(f"✓ No existing traces found, running training with profiler...")
            # Import train_model main function (avoid importing inside profiler context)
            from train_model import main as train_main
            
            # Run training under profiler context
            with profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,  # Record call stack
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(tensorboard_logs_dir)),
            ) as prof:
                train_main()
            
            # Note: Chrome trace export is disabled when using tensorboard_trace_handler
            # because the trace is already saved by the handler
            print(f"✓ TensorBoard logs saved to: {tensorboard_logs_dir}")
            print(f"✓ Chrome trace export skipped (already saved by tensorboard handler)")
        
        # Initialize memory_path variable
        memory_path = None
        
        # Generate performance statistics summary (only if we have profiler data)
        if not existing_traces:
            if torch.cuda.is_available():
                sort_key = "cuda_time_total"
            else:
                sort_key = "cpu_time_total"
                
            stats_table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
            
            # Save statistics summary to file
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write("PyTorch Profiler Performance Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Analysis Device: {'CPU + GPU' if torch.cuda.is_available() else 'CPU'}\n")
                f.write(f"Sort By: {sort_key}\n\n")
                f.write(stats_table)
            
            print(f"✓ Performance statistics summary saved to: {stats_path}")
            
            # Display top 10 most time-consuming operations in console
            print("\nTop 10 Most Time-Consuming Operations:")
            print("-" * 80)
            print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
            
            # Memory usage statistics
            if torch.cuda.is_available():
                memory_stats = prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10)
                print("\nTop 10 Memory-Intensive Operations:")
                print("-" * 80)
                print(memory_stats)
                
                memory_path = output_dir / "train_model_memory_stats.txt"
                with open(memory_path, 'w', encoding='utf-8') as f:
                    f.write("PyTorch Profiler Memory Usage Analysis Report\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(memory_stats)
                print(f"✓ Memory usage statistics saved to: {memory_path}")
        else:
            print(f"✓ Using existing trace files - statistics already available in TensorBoard")
            print(f"✓ Check existing statistics files: {stats_path}")
            # Check if memory stats file exists from previous run
            memory_path = output_dir / "train_model_memory_stats.txt"
            if not memory_path.exists():
                memory_path = None
        
        print(f"\nAnalysis completed! All files saved in: {output_dir}")
        print(f"\nView Results:")
        print(f"1. Performance summary: Check {stats_path}")
        if torch.cuda.is_available():
            print(f"2. Memory statistics: Check {memory_path}")
        print(f"3. TensorBoard visualization: Run 'tensorboard --logdir={output_dir / 'tensorboard_logs'}' then open http://localhost:6006")
        print(f"   - Click on 'PyTorch Profiler' tab")
        print(f"   - View operator shapes, memory usage, and detailed timeline")
        print(f"   - Chrome trace is available in TensorBoard interface")
        
    except Exception as e:
        print(f"❌ Profiler execution failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
