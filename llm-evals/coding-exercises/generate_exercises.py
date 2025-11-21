#!/usr/bin/env python3
"""
Script to generate exercise files from solutions.py

This script:
1. Generates solutions.ipynb from solutions.py using jupytext
2. Generates exercises.py by removing solution blocks and uncommenting NotImplementedError
3. Generates exercises.ipynb from exercises.py using jupytext
"""

import re
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Command: {cmd}")
        print(f"  Error: {e.stderr}")
        sys.exit(1)

def generate_exercises_py(solutions_file, exercises_file):
    """Generate exercises.py from solutions.py by removing solution blocks"""
    print(f"Generating {exercises_file} from {solutions_file}")
    
    with open(solutions_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    output_lines = []
    in_solution_block = False
    
    for line in lines:
        # Check for solution block markers
        if line.strip() == "#BEGIN SOLUTION":
            in_solution_block = True
            continue
        elif line.strip() == "#END SOLUTION":
            in_solution_block = False
            continue
        
        # Skip lines that are inside solution blocks
        if in_solution_block:
            continue
        
        # Uncomment NotImplementedError lines
        if line.strip().startswith("# raise NotImplementedError"):
            # Remove the "# " at the beginning to uncomment
            line = line.replace("# raise NotImplementedError", "raise NotImplementedError", 1)
        
        output_lines.append(line)
    
    # Write the exercises file
    with open(exercises_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"✓ Generated {exercises_file}")

def main():
    # Set up file paths
    base_dir = Path(__file__).parent
    solutions_py = base_dir / "solutions.py"
    exercises_py = base_dir / "exercises.py"
    solutions_ipynb = base_dir / "solutions.ipynb"
    exercises_ipynb = base_dir / "exercises.ipynb"
    
    # Check that solutions.py exists
    if not solutions_py.exists():
        print(f"Error: {solutions_py} does not exist")
        sys.exit(1)
    
    print("=== Generating exercise files from solutions.py ===")
    
    # Step 1: Generate solutions.ipynb from solutions.py
    run_command(
        f"jupytext --to notebook {solutions_py}",
        f"Converting {solutions_py} to {solutions_ipynb}"
    )
    
    # Step 2: Generate exercises.py by processing solutions.py
    generate_exercises_py(solutions_py, exercises_py)
    
    # Step 3: Generate exercises.ipynb from exercises.py
    run_command(
        f"jupytext --to notebook {exercises_py}",
        f"Converting {exercises_py} to {exercises_ipynb}"
    )
    
    print("\n=== Summary ===")
    print(f"✓ Generated {solutions_ipynb}")
    print(f"✓ Generated {exercises_py}")  
    print(f"✓ Generated {exercises_ipynb}")
    print("\nAll files generated successfully!")

if __name__ == "__main__":
    main()