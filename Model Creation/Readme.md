# Model Creation
Preprocess the dataset, train a PyTorch model, and predict on new unseen data.


## Dataset
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge/data)

## Preprocessing
1. Load the dataset
2. Split the video into frames
3. Crop the face from each frame
4. Save the face-cropped video

## Model & Training
1. Load preprocessed video and labels from a CSV file
2. Create a PyTorch model using transfer learning with ResNext50 and LSTM
3. Split data into train and test sets
4. Train and test the model
5. Save the model as a `.pt` file

## Prediction
- Load the saved PyTorch model
- Predict output based on trained weights

## Helpers
Utility scripts for:
- Converting JSON label files to CSV
- Copying files between directories
- Removing audio-altered files from the DFDC dataset
