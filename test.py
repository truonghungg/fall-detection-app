#!/usr/bin/python3
import asyncio
import websockets
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from aiohttp import web
import os
import time
from collections import deque
import pickle
import aiohttp_cors
from sklearn.metrics import accuracy_score, classification_report
import ssl
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ... rest of your code ... 