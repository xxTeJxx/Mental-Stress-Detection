
# Mental Stress Detection Using Deep Learning Models

This project focuses on detecting mental stress from textual data using various deep learning models, including Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Long Short-Term Memory (LSTM) networks. The textual data can be sourced from various inputs such as social media posts, academic writings, and other text-based interactions.

## Overview

The aim of this project is to develop a machine learning model that can accurately classify whether an individual is experiencing stress based on their textual data. The project integrates CNN, RNN, and LSTM architectures to extract meaningful insights and achieve high accuracy.

## Dataset

For training and evaluation, the ISEAR dataset is used, which contains labeled emotional responses. The data includes various emotions such as anger, fear, sadness, and joy, which are mapped to 'stressed' or 'not stressed' categories.

### Data Preprocessing

- The dataset is loaded and preprocessed by mapping emotions to stress levels.
- Textual data is tokenized and converted into sequences suitable for input to the models.
- Labels are encoded for classification purposes.

## Models Implemented

The following deep learning models are used in this project:

1. **LSTM Model:** Utilizes LSTM layers to capture long-term dependencies in the text data.
2. **CNN Model:** Employs convolutional layers to detect local patterns in the text sequences.
3. **RNN Model:** Uses simple RNN layers to process sequential data and extract patterns.

## Implementation

### Requirements

- Python 3.x
- TensorFlow
- Pandas
- Scikit-learn

Install the required packages using pip:

```bash
pip install tensorflow pandas scikit-learn
```

### Code Structure

- **`load_isear_dataset()`:** Loads and preprocesses the ISEAR dataset, mapping emotions to stress categories.
- **`preprocess_data()`:** Tokenizes and pads text sequences, encodes labels.
- **`build_lstm_model()`:** Defines the architecture for the LSTM model.
- **`build_cnn_model()`:** Defines the architecture for the CNN model.
- **`build_rnn_model()`:** Defines the architecture for the RNN model.
- **`train_and_evaluate_model()`:** Trains and evaluates each model, returning accuracy and other metrics.

### Running the Models

To train and evaluate the models, run the script as follows:

```python
python stress_detection.py
```

This will:

1. Load the dataset.
2. Split the data into training and testing sets.
3. Preprocess the text data and labels.
4. Train the LSTM, CNN, and RNN models.
5. Evaluate each model and print classification reports.

### Performance Metrics

- Each model's performance is evaluated using accuracy, precision, recall, and F1-score.
- A majority voting mechanism is used to combine predictions from all three models to provide a final decision on whether the individual is stressed or not.

## Results

The results include the accuracy and classification reports for each model. The final prediction is made based on majority voting from the three models.

- **LSTM Model Accuracy:** Achieved a high level of accuracy by capturing sequential patterns.
- **CNN Model Accuracy:** Showed robust performance in identifying local patterns in text.
- **RNN Model Accuracy:** Performed well in sequential data analysis but may be prone to vanishing gradients over long sequences.

## Conclusion

This project successfully demonstrates the application of deep learning models in detecting mental stress from textual data. By leveraging the strengths of CNN, RNN, and LSTM models, the approach provides a comprehensive solution for stress detection.

## References

- For more details on the ISEAR dataset, refer to the [official documentation](https://www.unige.ch/cisa/research/materials-and-online-research/research-material/).
- For implementation details, refer to the source code in this repository.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Special thanks to Moulya A S for being my partner in this project.
- Inspired by various works in the field of AI-driven mental health assessment.

---
