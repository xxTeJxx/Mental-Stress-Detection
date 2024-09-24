import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Conv1D, GlobalMaxPooling1D, SimpleRNN
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Function to load and preprocess the ISEAR dataset
def load_isear_dataset():
    isear_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/eng_dataset.csv')

    stress_mapping = {
        'anger': 'stressed',
        'fear': 'stressed',
        'sadness': 'stressed',
        'joy': 'not stressed',
    }

    # Apply the mapping to create 'label' column
    isear_df['label'] = isear_df['label'].apply(lambda x: stress_mapping.get(x, 'unknown'))

    # Filter out rows where label is 'unknown'
    isear_df = isear_df[isear_df['label'] != 'unknown']

    # Define X and y
    X = isear_df['text'].astype(str)
    y = isear_df['label']

    return X, y

# Common function to preprocess text and labels
def preprocess_data(X_train, X_test, y_train, y_test, max_words=5000, maxlen=100):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_test_enc = label_encoder.transform(y_test)

    return X_train_pad, X_test_pad, y_train_enc, y_test_enc, label_encoder

# Function to build and evaluate the LSTM model
def build_lstm_model(max_words=5000, maxlen=100, embedding_dim=100):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to build and evaluate the CNN model
def build_cnn_model(max_words=5000, maxlen=100, embedding_dim=100):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to build and evaluate the RNN model
def build_rnn_model(max_words=5000, maxlen=100, embedding_dim=100):
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
    model.add(SimpleRNN(100))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train_pad, y_train_enc, X_test_pad, y_test_enc, epochs=10, batch_size=64):
    model.fit(X_train_pad, y_train_enc, epochs=epochs, batch_size=batch_size, validation_data=(X_test_pad, y_test_enc), verbose=1)
    scores = model.evaluate(X_test_pad, y_test_enc, verbose=0)
    print(f"\nEvaluation on Test Data:\nLoss: {scores[0]}\nAccuracy: {scores[1]}")

    y_pred_prob = model.predict(X_test_pad)
    y_pred = (y_pred_prob > 0.5).astype(int)

    return y_pred, scores[1]

# Main execution
if __name__ == "__main__":
    # Load the ISEAR dataset
    X, y = load_isear_dataset()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the data
    X_train_pad, X_test_pad, y_train_enc, y_test_enc, label_encoder = preprocess_data(X_train, X_test, y_train, y_test)

    # LSTM Model
    print("\nTraining LSTM Model...")
    y_pred_lstm, lstm_accuracy = train_and_evaluate_model(build_lstm_model(), X_train_pad, y_train_enc, X_test_pad, y_test_enc)

    # CNN Model
    print("\nTraining CNN Model...")
    y_pred_cnn, cnn_accuracy = train_and_evaluate_model(build_cnn_model(), X_train_pad, y_train_enc, X_test_pad, y_test_enc)

    # RNN Model
    print("\nTraining RNN Model...")
    y_pred_rnn, rnn_accuracy = train_and_evaluate_model(build_rnn_model(), X_train_pad, y_train_enc, X_test_pad, y_test_enc)

    # Decode predictions
    y_pred_lstm_labels = label_encoder.inverse_transform(y_pred_lstm.flatten())
    y_pred_cnn_labels = label_encoder.inverse_transform(y_pred_cnn.flatten())
    y_pred_rnn_labels = label_encoder.inverse_transform(y_pred_rnn.flatten())

    # Calculate majority prediction
    majority_vote = pd.DataFrame({
        'LSTM': y_pred_lstm_labels,
        'CNN': y_pred_cnn_labels,
        'RNN': y_pred_rnn_labels
    }).mode(axis=1)[0]

    majority_prediction = 'stressed' if (majority_vote == 'stressed').sum() > (majority_vote == 'not stressed').sum() else 'not stressed'

    # Print performance metrics
    print("\nPerformance Metrics:")
    print(f"LSTM Accuracy: {lstm_accuracy:.4f}")
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
    print(f"RNN Accuracy: {rnn_accuracy:.4f}")

    print("\nFinal Prediction:")
    print(f"Based on the majority of models, the user is predicted to be: {majority_prediction.upper()}")

    print("\nClassification Reports:")
    print("\nLSTM Model Classification Report:")
    print(classification_report(y_test, y_pred_lstm_labels))

    print("\nCNN Model Classification Report:")
    print(classification_report(y_test, y_pred_cnn_labels))

    print("\nRNN Model Classification Report:")
    print(classification_report(y_test, y_pred_rnn_labels))
