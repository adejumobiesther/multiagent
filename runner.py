"""
run.py - Simulation Runner
===========================

Simple script to run the multi-agent simulation and save logs.

Usage:
    python run.py                    # Run with defaults (5 rounds)
    python run.py --rounds 10        # Run for 10 rounds
    python run.py --output my_sim    # Custom output filename

Output:
    Creates a JSON log file in ./logs/ folder
    Example: ./logs/sim_20240115_143022.json
"""

import argparse
import os
from datetime import datetime

# Import our simulation
from simulation import OrganizationSimulation, NUM_ROUNDS


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the multi-agent organization simulation"
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int,
        default=NUM_ROUNDS,
        help=f"Number of rounds to run (default: {NUM_ROUNDS})"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output filename (without extension). Default: auto-generated timestamp"
    )
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate output filename
    if args.output:
        filename = f"logs/{args.output}.json"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"logs/sim_{timestamp}.json"
    
    print("\n" + "="*60)
    print("STARTING SIMULATION")
    print("="*60)
    print(f"Rounds: {args.rounds}")
    print(f"Output: {filename}")
    print("="*60)
    
    # Create and run simulation
    sim = OrganizationSimulation()
    logs = sim.run(num_rounds=args.rounds)
    
    # Save logs
    logs.save(filename)
    
    print("\n" + "="*60)
    print("SIMULATION SAVED")
    print("="*60)
    print(f"Log file: {filename}")
    print(f"\nTo analyze results, run:")
    print(f"    python analysis.py {filename}")
    print("="*60 + "\n")
    
    return filename


if __name__ == "__main__":
    main()