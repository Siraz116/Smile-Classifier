from pydantic import BaseModel
from datetime import datetime

class ImageBase(BaseModel):
    image_path : str
    classification : str


class ImageCreate(ImageBase):
    pass


class Image(ImageBase):
    id : int
    timestamp : datetime

    class Config:
        orm_mode = True