import os
import pickle
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_gesture_model(model_path='models/gesture_model.pkl'):
    """Load the trained gesture recognition model."""
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            print(f"ERROR: Model file not found at {model_path}")
            print("Please run train_nonverbal_models.py first to generate the model.")
            return None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Loaded gesture model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading gesture model: {str(e)}")
        print(f"ERROR: Failed to load model: {str(e)}")
        return None

def test_gestures(model):
    """Test the model with sample gesture inputs."""
    if model is None:
        return
        
    # Sample gestures to test
    test_gestures = {
        "Hand Up": [0.5, 0.8, 160, 45],
        "Wave Left": [0.3, 0.6, 120, 30],
        "Wave Right": [0.7, 0.6, 120, 30],
        "Stimming": [0.5, 0.5, 90, 90]
    }
    
    print("\n===== GESTURE MODEL TEST =====")
    print("Testing standard gestures:")
    
    for name, features in test_gestures.items():
        try:
            # Convert to numpy array and reshape for model
            features_array = np.array(features).reshape(1, -1)
            
            # Predict the gesture
            prediction = model.predict(features_array)[0]
            confidence = max(model.predict_proba(features_array)[0]) * 100
            
            # Print the result with color coding
            result = "✓ MATCH" if prediction == name else "✗ MISMATCH"
            color = "\033[92m" if prediction == name else "\033[91m"  # Green or red
            
            print(f"{color}Testing '{name}' gesture: Predicted '{prediction}' with {confidence:.1f}% confidence - {result}\033[0m")
            
        except Exception as e:
            logger.error(f"Error predicting gesture {name}: {str(e)}")
            print(f"Error testing '{name}' gesture: {str(e)}")
    
    # Test a custom gesture
    print("\nTesting a custom gesture input:")
    custom_features = [0.6, 0.7, 140, 50]  # Something between Hand Up and Wave Right
    try:
        features_array = np.array(custom_features).reshape(1, -1)
        prediction = model.predict(features_array)[0]
        confidence = max(model.predict_proba(features_array)[0]) * 100
        print(f"Custom gesture [0.6, 0.7, 140, 50]: Predicted '{prediction}' with {confidence:.1f}% confidence")
    except Exception as e:
        logger.error(f"Error predicting custom gesture: {str(e)}")
        print(f"Error testing custom gesture: {str(e)}")
        
    print("\n===== MODEL INFORMATION =====")
    try:
        print(f"Model type: {type(model).__name__}")
        if hasattr(model, 'n_estimators'):
            print(f"Number of trees: {model.n_estimators}")
        if hasattr(model, 'feature_importances_'):
            print("Feature importance ranking:")
            features = ['wrist_x', 'wrist_y', 'elbow_angle', 'shoulder_angle']
            importances = model.feature_importances_
            for feature, importance in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
                print(f"  - {feature}: {importance:.4f}")
        print(f"Classes recognized: {model.classes_}")
    except Exception as e:
        logger.error(f"Error displaying model information: {str(e)}")
        print(f"Error displaying model information: {str(e)}")
        
    print("\n===== USAGE IN ALPHAVOX =====")
    print("To use this model in the AlphaVox system:")
    print("1. Ensure alphavox_input_nlu.py correctly references this model")
    print("2. Test gesture processing with an interaction like:")
    print('   {"type": "gesture", "input": [0.5, 0.8, 160, 45]}')
    print("3. The model should return a predicted gesture name with confidence score")

def main():
    """Test the trained gesture model."""
    model = load_gesture_model()
    if model:
        test_gestures(model)
    else:
        print("\nPlease train the model first by running:")
        print("python train_nonverbal_models.py")

if __name__ == "__main__":
    main()