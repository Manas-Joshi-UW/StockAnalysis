#!/usr/bin/env python3
"""
Local deployment script for Stock Analysis Dashboard
Run this to start your web app accessible from any device on your network
"""

import subprocess
import sys
import os
import socket

def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Connect to a remote address to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def main():
    print("🚀 Starting Stock Analysis Dashboard...")
    print(f"📱 Local URL: http://localhost:8050")
    
    local_ip = get_local_ip()
    if local_ip != "127.0.0.1":
        print(f"📱 Network URL: http://{local_ip}:8050")
        print(f"📱 Access from your phone using: http://{local_ip}:8050")
    
    print("\n💡 Tips:")
    print("• Make sure your phone is on the same WiFi network")
    print("• If you can't connect, check your firewall settings")
    print("• Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Run the Dash app
    try:
        subprocess.run([sys.executable, "interface.py"], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main()

