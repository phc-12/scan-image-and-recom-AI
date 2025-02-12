import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from skimage import transform
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib  # For saving the model

class StyleRecommender:
    def __init__(self, image_folder, style_labels):
        self.image_folder = image_folder
        self.style_labels = style_labels
        for label in style_labels:
            if not os.path.exists(os.path.join(image_folder, label)):
                raise ValueError(f"Folder for label {label} does not exist.")
            label = ["casual", "elegant", "fancy", "formal", "minimal"]
        self.image_folder = image_folder
        self.style_labels = style_labels
        self.label_encoder = LabelEncoder()
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
    def load_and_preprocess_data(self):
        X = []
        y = []
        
        for label in self.style_labels:
            label_path = os.path.join(self.image_folder, label)
            if not os.path.exists(label_path):
                print(f"Warning: {label_path} does not exist. Skipping this label.")
                continue
                
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # Convert color space and resize
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = transform.resize(img, (128, 128))
                    
                    # Option to extract additional features:
                    features = self.extract_features(img)
                    # Flatten the image pixels and concatenate with extracted features:
                    flat_pixels = img.flatten()
                    combined_features = np.concatenate([flat_pixels, np.array(features)])
                    X.append(combined_features)
                    
                    y.append(label)
                else:
                    print(f"Warning: Could not read image at {img_path}")
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y = self.label_encoder.fit_transform(y)
        
        # Scale the features (recommended if features vary in scale)
        X = self.scaler.fit_transform(X)
        
        return X, y
    
    def train(self):
        X, y = self.load_and_preprocess_data()
        if len(X) == 0:
            raise ValueError("No images found. Please check your dataset path and image files.")
            
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.clf.fit(self.X_train, self.y_train)
        
    def evaluate(self):
        y_pred = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        target_names = self.label_encoder.classes_
        report = classification_report(self.y_test, y_pred, target_names=target_names)
        
        return accuracy, report
    
    def predict_style(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return "Error: Could not load image"
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform.resize(img, (128, 128))
        
        # Extract features as done in training
        features = self.extract_features(img)
        flat_pixels = img.flatten()
        combined_features = np.concatenate([flat_pixels, np.array(features)])
        
        # Scale features using the same scaler as training
        combined_features = self.scaler.transform([combined_features])
        
        # Get prediction probabilities
        probs = self.clf.predict_proba(combined_features)[0]
        
        # Use label encoder's classes for correct ordering
        style_names = self.label_encoder.classes_
        style_probs = list(zip(style_names, probs))
        
        return sorted(style_probs, key=lambda x: x[1], reverse=True)
    
    def extract_features(self, img):
        features = []
        # Average color across channels (R, G, B)
        avg_color = np.mean(img, axis=(0, 1))
        features.extend(avg_color)
        
        # Overall brightness (using grayscale conversion)
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        features.append(brightness)
        
        return features
    
    def save_model(self, model_path):
        # Save the classifier, scaler, and label encoder
        joblib.dump({
            'classifier': self.clf,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, model_path)
    
    def load_model(self, model_path):
        # Load a saved model
        data = joblib.load(model_path)
        self.clf = data['classifier']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']

