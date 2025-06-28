import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Initialize SQLAlchemy base class
class Base(DeclarativeBase):
    pass

# Initialize database
db = SQLAlchemy(model_class=Base)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "alphavox_dev_secret")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure database
database_url = os.environ.get("DATABASE_URL", "sqlite:///alphavox.db")
# Ensure proper PostgreSQL URL format
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

# Set database configuration
app.config["SQLALCHEMY_DATABASE_URI"] = database_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True
}

# Log database connection (hide credentials)
logging.info(f"Connecting to database: {database_url.split('@')[0].split('://')[-1]}@[hidden]")

# Initialize database with app
db.init_app(app)

# Create database tables within app context
with app.app_context():
    import models  # Import models here to avoid circular imports
    db.create_all()