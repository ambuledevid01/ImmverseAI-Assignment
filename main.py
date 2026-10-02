# Main entry point for manuscript layout region detection verification.
import sys
import os
from config import load_config
from src.utils.logger import setup_logger
logger = setup_logger("Main")
def print_project_summary():
    try:
        config = load_config()
        print("\n" + "=" * 65)
        print(f" Project: {config.get('project_name', 'manuscript-layout-detector')}")
        print(f" Version: {config.get('version', '0.1.0')}")
        print("=" * 65)
        print("\nTarget Manuscript Layout Region Classes:")
        for idx, cls in enumerate(config.get("target_classes", []), start=1):
            print(f"  {idx}. {cls}")
        print("\nDefault Paths:")
        for key, path in config.get("paths", {}).items():
            print(f"  - {key}: {path}")
        print("=" * 65 + "\n")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
def main():
    logger.info("Initializing Manuscript Specific Layout Region Detector...")
    print_project_summary()
    logger.info("Project initialized successfully.")
if __name__ == "__main__":
    main()
