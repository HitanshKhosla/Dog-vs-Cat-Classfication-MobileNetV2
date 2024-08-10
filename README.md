

# Dog vs. Cat Classification with MobileNetV2

This project focuses on classifying images of dogs and cats using MobileNetV2. The dataset used is from the Kaggle competition "Dogs vs. Cats," and the model achieves an accuracy of 97.25%.

## Project Overview

The objective is to create a model capable of distinguishing between images of dogs and cats. MobileNetV2 is utilized as the feature extractor, with a custom classification head for binary classification.

## Dataset

The dataset is obtained from the Kaggle competition [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats). It includes labeled images of dogs and cats for training and evaluation.

To download the dataset, use the following Kaggle API command:

```bash
!kaggle competitions download -c dogs-vs-cats
```

## Installation

Make sure to install the required libraries:

```bash
pip install tensorflow tensorflow_hub numpy opencv-python
```

## Model Architecture

The model is built using MobileNetV2 as the feature extractor with the following components:

1. **MobileNetV2**: Pre-trained on ImageNet, used for feature extraction.
2. **Dense Layer**: A fully connected layer with 2 units and a softmax activation function for binary classification.

## Training

The model uses the following configuration:

- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Batch Size**: 32
- **Epochs**: 5
- **No Validation Split**: The entire dataset is used for training without validation.

### Example Code

Here's the implementation of the model:

```python
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Load pre-trained MobileNetV2 module
hub_module = hub.KerasLayer('https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4', trainable=False)

# Define the model
model = Sequential([
    hub_module,
    Dense(2, activation='softmax')  # Use softmax for SparseCategoricalCrossentropy
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(
    X_train_scaled, 
    Y_train, 
    epochs=5, 
    verbose=1
)
```

## Results

The model achieves an accuracy of **97.25%** on the training set, demonstrating its capability to classify dog and cat images effectively.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- **Kaggle**: For providing the dataset.
- **TensorFlow Hub**: For the pre-trained MobileNetV2 model.

For more information on TensorFlow Hub and MobileNetV2, refer to the [official TensorFlow Hub documentation](https://www.tensorflow.org/hub).
