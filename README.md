# Dermoscopy Image Classification for Melanoma Detection

This project implements a deep learning approach for classifying dermoscopy images, potentially aiding in the diagnosis of melanoma. The project builds upon the master thesis titled ["Combining deep learning, handcrafted features, and metadata for the classification of dermoscopy images"](https://dione.lib.unipi.gr/xmlui/handle/unipi/15875) by Dimitris Tsel.

## Project Structure

The project is organized into the following directories:

- **models**: This directory stores the trained model weights generated during training.
- **outputs**: This directory stores any output files generated during training or validation, such as loss curves, visualizations, or predictions.
- **scripts**: This directory contains Python modules (.py files) that are imported and used by other scripts for various functionalities:

  - Training and validation (`train-val.ipynb`)
  - Data handling (`dataio.py`, `prepareDataset.ipynb`)
  - Preprocessing (`preprocess.py`)
  - Feature extraction (`colorfeatures.py`, `feature_extraction.py`)
  - Model prediction (`classify.py`, `predict.ipynb`)
  - Data augmentation (`dataAugmentation.ipynb`)
  - Metadata analysis (`metadataAnalysis.ipynb`)
  - Segmentation (`segment.py`)
  - Testing (`test.ipynb`)

## Dependencies

Python libraries:

- TensorFlow
- Keras
- Scikit-learn
- OpenCV
- NumPy
- matplotlib
- seaborn
  
## Data

The project uses a dermoscopy image dataset of [The ISIC 2020 Challenge](https://challenge2020.isic-archive.com/) for Skin Lesion Analysis Towards Melanoma Detection.

## Functionality

### Data Preparation:

- Load the dermoscopy image dataset (imports `dataio.py`).
- Preprocess the images (e.g., resizing, normalization) (imports `preprocess.py`).
- Split the data into training, validation, and testing sets (refer to `prepareDataset.ipynb`).
- Implement data augmentation techniques to balance Benign and Malignant classes and help a model improve the accuracy of predictions (refer to `dataAugmentation.ipynb`).

### Feature Extraction:

- Extract color features from images (potentially using `colorfeatures.py`).
- Extract handcrafted features, such as Grey Level Co-occurrence Matrix (GLCM) and Local Binary Pattern (LBP) features (imports `feature_extraction.py`).
- Pre-trained deep learning models like DenseNet, VGG19, and EfficientNet for feature extraction.
- Combine these features (imports `feature_extraction.py`).

### Model Training:

- Define a convolutional neural network (CNN) architecture suitable for image classification (refer to `train-val.ipynb`).
- Integrate the extracted features, including deep learning features from DenseNet, VGG19, and EfficientNet, handcrafted features like GLCM and LBP, and patients' metadata features, into the model.
- Train the model on the training set, monitoring progress using metrics like accuracy and loss.
- Implement techniques like early stopping to prevent overfitting.

### Validation and Testing:

- Evaluate the trained model's performance on the validation set (refer to `train-val.ipynb`).
- Optionally, evaluate the model on a separate test set (refer to `test.ipynb`) for a more unbiased assessment.

### Evaluation:
Calculates and display the following metrics:
- Accuracy: Proportion of correctly classified samples.
- Precision: Ratio of true positives to all predicted positives.
- Recall: Ratio of true positives to all actual positives.
- F1 Score: Harmonic mean of precision and recall, useful when dealing with imbalanced class distributions.
- Specificity: Ratio of true negatives to all predicted negatives.
- AUC (Area Under the ROC Curve): Represents the probability that the model ranks a randomly chosen positive instance higher than a randomly chosen negative instance.
The notebook provides visualizations of these metrics, such as confusion matrices or ROC curves, to aid in model evaluation.

### Prediction:

- Use the trained model to classify new dermoscopy images.
- The model outputs the predicted class (e.g., melanoma or benign).

## Using Jupyter Notebooks for Dermoscopy Image Classification

### Option 1: Using a Jupyter Notebook server

1. **Install a Jupyter Notebook server** following the instructions for your operating system.
2. **Navigate to the project directory** in your terminal:
    ```bash
    cd Dermoscopy-Image-Classification
    ```
3. **Start the Jupyter Notebook server**:
    ```bash
    jupyter notebook
    ```
   This will open a web interface in your browser, typically at [http://localhost:8888](http://localhost:8888).
4. From the list of notebooks, open the script`train-val.ipynb`.

### Option 2: Using a local kernel (command line)

1. Ensure you have a Python kernel installed for Jupyter Notebooks (e.g., ipykernel).
2. **Activate the project's virtual environment** (if applicable).
3. **Navigate to the project directory** in your terminal.
4. **Run the desired script directly**:
    ```bash
    jupyter notebook train-val.ipynb 
    ```

## Further Development

Here are some suggestions for further development:

- Experiment with different CNN architectures and more powerful models with more parameters.
- Explore advanced training techniques like transfer learning.
- Explore explainability in Deep Learning and analyze the decisions behind the model's predictions.


