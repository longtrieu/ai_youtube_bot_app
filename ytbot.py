#!/usr/bin/env python3
"""
YouTube Bot Application Launcher
This is the main entry point for the YouTube Bot application.
The application has been modularized for better organization and maintainability.
"""

from main_app import create_interface

if __name__ == "__main__":
  # Create and launch the interface
  interface = create_interface()
  interface.launch(server_name="0.0.0.0", server_port=7860)