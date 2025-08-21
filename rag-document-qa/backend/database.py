import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
engine = None
SessionLocal = None

if DATABASE_URL and DATABASE_URL != "your_supabase_connection_string_here":
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        print("✅ Database connection established successfully")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        engine = None
        SessionLocal = None
else:
    print("⚠️  DATABASE_URL not configured - running in demo mode")

def get_db():
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL is not configured or invalid")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create database tables if they don't exist"""
    if engine is not None:
        try:
            from models import Base
            Base.metadata.create_all(bind=engine)
            print("✅ Database tables created successfully")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
