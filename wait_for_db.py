#!/usr/bin/env python
"""
Database Connection Verification Script for AlphaVox

This script tries to connect to the configured database and waits until the
connection is successful or a timeout is reached. It's used during deployment
to ensure that database migrations only run after the database is available.

Usage:
    python scripts/wait_for_db.py [--timeout SECONDS]

Options:
    --timeout SECONDS    Maximum time to wait in seconds (default: 60)
"""

import argparse
import logging
import os
import sys
import time
from urllib.parse import urlparse

import psycopg2

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("wait_for_db")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AlphaVox Database Connection Verification")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Maximum time to wait in seconds (default: 60)",
    )
    return parser.parse_args()

def get_database_url():
    """Get the database URL from environment variables."""
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        # Try to construct from individual components
        db_user = os.environ.get("POSTGRES_USER")
        db_password = os.environ.get("POSTGRES_PASSWORD")
        db_host = os.environ.get("POSTGRES_HOST")
        db_port = os.environ.get("POSTGRES_PORT", "5432")
        db_name = os.environ.get("POSTGRES_DB", "alphavox")
        
        if db_user and db_password and db_host:
            db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        else:
            logger.error("Database connection information not found.")
            sys.exit(1)
    
    return db_url

def parse_db_url(db_url):
    """Parse the database URL into components."""
    try:
        result = urlparse(db_url)
        
        # Extract username and password
        if '@' in result.netloc:
            userpass, hostport = result.netloc.split('@', 1)
        else:
            userpass, hostport = '', result.netloc
            
        if ':' in userpass:
            username, password = userpass.split(':', 1)
        else:
            username, password = userpass, ''
        
        # Extract host and port
        if ':' in hostport:
            host, port = hostport.split(':', 1)
            port = int(port)
        else:
            host, port = hostport, 5432
        
        # Extract database name
        database = result.path.lstrip('/')
        
        return {
            'user': username,
            'password': password,
            'host': host,
            'port': port,
            'database': database,
        }
    except Exception as e:
        logger.error(f"Error parsing database URL: {e}")
        sys.exit(1)

def try_connect(db_params):
    """Try to connect to the database."""
    try:
        conn = psycopg2.connect(**db_params)
        conn.close()
        return True
    except Exception as e:
        logger.debug(f"Connection failed: {e}")
        return False

def wait_for_database(db_params, timeout=60):
    """Wait for the database to become available."""
    logger.info(f"Waiting for database connection at {db_params['host']}:{db_params['port']}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if try_connect(db_params):
            elapsed = time.time() - start_time
            logger.info(f"Database connection successful after {elapsed:.2f} seconds.")
            return True
        
        # Wait before trying again
        time.sleep(1)
    
    logger.error(f"Timeout after {timeout} seconds waiting for database connection.")
    return False

def main():
    """Main function."""
    args = parse_args()
    
    # Get and parse database URL
    db_url = get_database_url()
    db_params = parse_db_url(db_url)
    
    # Wait for database connection
    if wait_for_database(db_params, args.timeout):
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())