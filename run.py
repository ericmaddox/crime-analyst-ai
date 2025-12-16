#!/usr/bin/env python3
"""
Crime Analyst AI - Launcher Script

This script provides a convenient way to launch the Crime Analyst AI application.
It handles path setup and imports from the src directory.

Usage:
    python run.py          # Launch the web UI
    python run.py --cli    # Run command-line analysis
"""

import sys
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))


def main():
    """Main entry point for the launcher."""
    parser = argparse.ArgumentParser(
        description="Crime Analyst AI - Predictive Crime Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py              # Launch the web interface
    python run.py --cli        # Run command-line analysis
    python run.py --port 8080  # Use a different port
        """
    )
    
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in command-line mode instead of web UI"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port for the web server (default: 7860)"
    )
    
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open the browser"
    )
    
    args = parser.parse_args()
    
    if args.cli:
        # Run command-line mode
        from crime_analyst_ai.core import main as cli_main
        cli_main()
    else:
        # Run web UI
        from crime_analyst_ai.app import create_app
        
        print("\n" + "=" * 50)
        print("   CRIME ANALYST AI")
        print("   Predictive Crime Analysis Tool")
        print("=" * 50)
        print(f"\nStarting server on port {args.port}...")
        print(f"Open http://localhost:{args.port} in your browser\n")
        
        app = create_app()
        app.launch(
            server_name="0.0.0.0",
            server_port=args.port,
            share=False,
            show_error=True,
            inbrowser=not args.no_browser
        )


if __name__ == "__main__":
    main()

