import os
import sys

# Add your application directory to Python path
INTERP = os.path.expanduser("/home/username/venv/bin/python")
if sys.executable != INTERP:
    os.execl(INTERP, INTERP, *sys.argv)

sys.path.append(os.getcwd())

# Import your application
from test import main
import asyncio

# Run the application
if __name__ == '__main__':
    asyncio.run(main()) 