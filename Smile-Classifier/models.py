from sqlalchemy import Column, Integer, String, DateTime
from database import Base
from datetime import datetime

class Classification(Base):
    """
    Database model for storing classification results
    """
    __tablename__ = "classifications"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(255), nullable=False)
    classification = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.now)