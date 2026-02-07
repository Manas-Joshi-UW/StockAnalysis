"""
Production-ready Stock Analysis Dashboard
This file is optimized for cloud deployment
"""

from interface import app
import os

# For production deployment
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))

