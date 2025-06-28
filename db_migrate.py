#!/usr/bin/env python
"""
Database Migration Script for AlphaVox

This script handles database migrations for AlphaVox, ensuring that the database schema
is up-to-date with the latest application model definitions. It creates tables if they
don't exist and applies any necessary schema changes.

Usage:
    python scripts/db_migrate.py [--force]

Options:
    --force    Force migration even if it would result in data loss
"""

import argparse
import logging
import os
import sys
from urllib.parse import urlparse

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("db_migrate")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="AlphaVox Database Migration Script")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force migration even if it would result in data loss",
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

def validate_db_url(db_url):
    """Validate the database URL."""
    try:
        result = urlparse(db_url)
        if not all([result.scheme, result.netloc]):
            logger.error("Invalid database URL format.")
            return False
        return True
    except Exception as e:
        logger.error(f"Error parsing database URL: {e}")
        return False

def create_tables():
    """Create all tables in the database."""
    from app import app
    from app import db
    
    logger.info("Creating database tables...")
    
    with app.app_context():
        # Import all models to ensure they're registered with SQLAlchemy
        import models
        
        # Create tables
        try:
            db.create_all()
            logger.info("Database tables created successfully.")
            return True
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            return False

def add_initial_data():
    """Add initial seed data to the database if needed."""
    from app import app
    
    logger.info("Adding initial data...")
    
    with app.app_context():
        # Import necessary models
        from models import User
        from db import db
        # Not using werkzeug security for our simplified User model
    # from werkzeug.security import generate_password_hash
        
        # Check if admin user exists
        admin = User.query.filter_by(name="admin").first()
        if not admin:
            # Create admin user
            admin = User(name="admin")
            # Our current User model is simple and doesn't have username/email/password fields
            db.session.add(admin)
            db.session.commit()
            logger.info("Added admin user.")
        else:
            logger.info("Admin user already exists.")
        
        return True

def run_migrations():
    """Run all database migrations."""
    db_url = get_database_url()
    if not validate_db_url(db_url):
        return False
    
    # Set the environment variable for the app to use
    os.environ["DATABASE_URL"] = db_url
    
    # Create tables
    if not create_tables():
        return False
    
    # Add initial data if needed
    if not add_initial_data():
        logger.warning("Failed to add initial data.")
    
    return True

def main():
    """Main function."""
    args = parse_args()
    
    # Set force flag if provided
    if args.force:
        os.environ["DB_MIGRATION_FORCE"] = "1"
    
    logger.info("Starting database migration...")
    
    success = run_migrations()
    
    if success:
        logger.info("Database migration completed successfully.")
        return 0
    else:
        logger.error("Database migration failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())