from sqlalchemy.orm import Session
import models,schemas

def create_image(db: Session, image:schemas.ImageCreate):
    db_image = models.Classification(image_path=image.image_path,
                          classification=image.classification)
    print(f"Image Added")
    db.add(db_image)
    db.commit()
    db.refresh(db_image)
    return db_image


def get_images(db: Session):
    return db.query(models.Classification).all()