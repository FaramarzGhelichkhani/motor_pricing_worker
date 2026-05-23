from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from core.config import DATABASE_URL, DJANGO_DB_URL

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
        )
else:
    engine = create_engine(
        DATABASE_URL, 
        pool_size=10,       
        max_overflow=20,    
        pool_pre_ping=True  
    )
    django_engine = create_engine(DJANGO_DB_URL,pool_size=10, max_overflow=20, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
DjangoSession = sessionmaker(autocommit=False, autoflush=False, bind=django_engine)
Base = declarative_base()

def get_db():
    """
    Dependency Injection Generator
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
