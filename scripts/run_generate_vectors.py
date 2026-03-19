#!/usr/bin/env python3
"""Generate persona steering vectors.

Usage:
  python scripts/run_generate_vectors.py                    # Generate for all personas
  python scripts/run_generate_vectors.py <persona_uuid>    # Generate for one persona
"""

import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.synth_persona_io import SynthPersonaDataset
from src.generate_persona_vector import main as generate_vector

def list_personas():
    """List all available personas."""
    try:
        dataset = SynthPersonaDataset()
        print(f"Available personas ({len(dataset)} total):\n")
        for i, persona in enumerate(dataset):
            qa_count = len(dataset.get_qa(persona.id))
            print(f"  [{i}] {persona.name}")
            print(f"      ID: {persona.id}")
            print(f"      QA pairs: {qa_count}")
            print()
        return dataset
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("\nMake sure your HuggingFace token is configured:")
        print("  huggingface-cli login")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate persona steering vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all personas
  python scripts/run_generate_vectors.py --list
  
  # Generate for one persona
  python scripts/run_generate_vectors.py 0023952f-142e-434b-82e2-7a7451b7c55f
  
  # Generate for all personas
  python scripts/run_generate_vectors.py --all
        """
    )
    parser.add_argument("persona_id", nargs="?", help="Persona UUID to generate for")
    parser.add_argument("--list", action="store_true", help="List all available personas")
    parser.add_argument("--all", action="store_true", help="Generate for all personas")
    parser.add_argument("--model", default="google/gemma-2-9b-it", help="HF Model ID")
    parser.add_argument("--layer", type=int, default=20, help="Layer to extract from")
    
    args = parser.parse_args()
    
    # Load dataset for any operation
    dataset = SynthPersonaDataset()
    
    if args.list:
        list_personas()
        return
    
    if args.all:
        print(f"Generating vectors for all {len(dataset)} personas...\n")
        for i, persona in enumerate(dataset):
            print(f"[{i+1}/{len(dataset)}] {persona.name}... ", end="", flush=True)
            try:
                # Call the generate script with this persona_id
                sys.argv = ["generate_persona_vector.py", "--persona_id", persona.id, "--model", args.model, "--layer", str(args.layer)]
                generate_vector()
                print("✓")
            except SystemExit:
                # generate_vector calls sys.exit or returns
                print("✓")
            except Exception as e:
                print(f"✗ ({e})")
        return
    
    if args.persona_id:
        # Check if persona exists
        persona = next((p for p in dataset if p.id == args.persona_id), None)
        if not persona:
            print(f"Persona {args.persona_id} not found.")
            print("\nAvailable personas:")
            list_personas()
            sys.exit(1)
        
        print(f"Generating vector for: {persona.name}")
        sys.argv = ["generate_persona_vector.py", "--persona_id", args.persona_id, "--model", args.model, "--layer", str(args.layer)]
        generate_vector()
        return
    
    # No argument provided
    parser.print_help()
    print("\nAvailable personas:")
    list_personas()

if __name__ == "__main__":
    main()
