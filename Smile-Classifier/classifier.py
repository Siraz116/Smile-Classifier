import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
from PIL import Image

class SmileClassifier:
    def __init__(self, dataset_path):
        """
        Initialize the smile classifier
        
        Args:
            dataset_path (str): Path to the dataset directory
        """
        self.dataset_path = dataset_path
        self.model = None
        self.scaler = StandardScaler()

    def load_dataset(self):
        """
        Load and preprocess image dataset
        
        Returns:
            tuple: X (images), y (labels)
        """
        images = []
        labels = []
        
        for label_folder in ['smiling', 'not_smiling']:
            folder_path = os.path.join(self.dataset_path, label_folder)
            label = 1 if label_folder == 'smiling' else 0
            
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize((64, 64))  # Resize to a standard size
                    img_array = np.array(img).flatten()  # Flatten image
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        return np.array(images), np.array(labels)

    def train_model(self):
        """
        Train Random Forest Classifier for smile detection
        
        Why Random Forest?
        - Handles non-linear relationships
        - Robust to overfitting
        - Works well with image classification
        - Can handle complex feature interactions
        """
        # Load dataset
        X, y = self.load_dataset()
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        print(classification_report(y_test, y_pred))
        
        return self.model

    def save_model(self, filepath='smile_classifier.pkl'):
        """
        Save trained model and scaler
        
        Args:
            filepath (str): Path to save model
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler
            }, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath='smile_classifier.pkl'):
        """
        Load saved model and scaler
        
        Args:
            filepath (str): Path to load model from
        
        Returns:
            dict: Containing model and scaler
        """
        with open(filepath, 'rb') as f:
            saved_data = pickle.load(f)
        return saved_data

def main():
    # Example usage
    classifier = SmileClassifier(dataset_path='./dataset')
    classifier.train_model()
    classifier.save_model()

if __name__ == "__main__":
    main()