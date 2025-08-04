#!/usr/bin/env python3
"""
Simple cron script for Railway
"""
from app import fetch_and_process_news

if __name__ == "__main__":
    fetch_and_process_news()
