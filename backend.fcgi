#!/usr/bin/python3
import sys
import os

# Add your Python path
sys.path.insert(0, '/domains/actionrecognition.id.vn/backend')

# Set environment variables
os.environ['PYTHONPATH'] = '/domains/actionrecognition.id.vn/backend'

# Import your application
from test import main
import asyncio

# Run the application
if __name__ == '__main__':
    asyncio.run(main()) 