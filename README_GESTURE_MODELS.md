# AlphaVox Nonverbal Gesture Models

This documentation explains how to use the gesture model training system in AlphaVox, particularly for nonverbal autism and neurodivergent communication support.

## Training the Gesture Model

The `train_nonverbal_models.py` script creates a machine learning model that recognizes different gestures based on body/hand positions.

### How to Run

```bash
python train_nonverbal_models.py
```

This will train the model and save `gesture_model.pkl` to the `models/` directory.

### Default Gestures

The model is pre-configured with simulated data for these autism-relevant gestures:
- **Hand Up** (request attention)
- **Wave Left** (greeting)
- **Wave Right** (greeting)
- **Stimming** (self-regulation, common in autism)

## Using Real Data (Optional)

The script uses simulated data by default, but you can use real data for more accurate results:

1. Create a CSV file (e.g., `gesture_data.csv`) with these columns:
   ```
   wrist_x,wrist_y,elbow_angle,shoulder_angle,label
   0.5,0.8,160,45,Hand Up
   0.3,0.6,120,30,Wave Left
   0.5,0.5,90,90,Stimming
   ```

2. Update the `data_path` in the `main()` function to point to your CSV file.

3. Run the script as usual.

## Recommended Data Sources

To make the model more relevant to nonverbal autism and neurodivergent users, consider these data sources:

### Public Datasets

- **HaGRID Dataset**: Contains hand gesture images for autism-relevant gestures like pointing or stimming. Requires preprocessing to extract features.
  
- **DHG-14/28 Dataset**: Includes skeletal joint data for hand gestures, adaptable for AlphaVox's feature set.
  
- **Jester Dataset**: Video-based hand gestures useful for dynamic gestures like waving or stimming. Requires video-to-feature extraction.

### Custom Collection

You can collect your own data using:

```python
import cv2
import mediapipe as mp

def extract_features(image_path):
    mp_hands = mp.solutions.hands.Hands()
    image = cv2.imread(image_path)
    results = mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        wrist = results.multi_hand_landmarks[0].landmark[0]  # Wrist landmark
        elbow_angle = 160  # Placeholder (calculate from joints)
        shoulder_angle = 45  # Placeholder
        return [wrist.x, wrist.y, elbow_angle, shoulder_angle]
    return None
```

Record gestures common in autism (e.g., hand-flapping, pointing, rocking) from consenting participants or simulations, ensuring ethical data collection.

## Research-Informed Gestures

The script uses the Research Module to add gestures like "Pointing" based on autism research (e.g., PECS-inspired communication). The `update_gestures_from_research()` function can be expanded to include more gestures from relevant studies.

## Integration with AlphaVox

Once `gesture_model.pkl` is generated:

1. It's automatically used by `alphavox_input_nlu.py` for gesture processing.

2. The Flask app (`app.py`) routes like `/speak/<gesture>` will process gestures correctly.

3. Test gesture processing by running `alphavox_input_nlu.py` or the Flask app and sending a gesture interaction:
   ```
   {"type": "gesture", "input": [0.5, 0.8, 160, 45]}
   ```

The Research Module ensures gestures reflect autism-relevant communication strategies (e.g., stimming, pointing), enhancing AlphaVox's neurodiversity-affirming approach.

## Troubleshooting

If you encounter issues:

1. Verify the model file exists: `ls models/gesture_model.pkl`

2. Check the logs for accuracy and classification reports

3. Ensure `alphavox_input_nlu.py` correctly references the model file path