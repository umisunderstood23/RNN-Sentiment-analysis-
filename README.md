# RNN Sentiment Analysis

## Overview

This project demonstrates the implementation of a simple Recurrent Neural Network (RNN) for binary sentiment analysis using TensorFlow/Keras.

The model is trained on a small manually created dataset consisting of 10 sentences, where each sentence is labeled as either **Positive (1)** or **Negative (0)**.

Since the dataset is manually created and already clean, common NLP preprocessing steps such as punctuation removal, stop-word removal, stemming, and lemmatization have been intentionally skipped. The project focuses on understanding the complete deep learning pipeline involving tokenization, embeddings, RNNs, model training, and inference.

---

## Dataset

The dataset consists of:

* A Python list containing 10 sentences.
* A corresponding label list where:

  * **1** → Positive Sentiment
  * **0** → Negative Sentiment

Each sentence has one corresponding sentiment label.

---

# Project Workflow

## Step 1: Tokenization

The first step is converting textual data into numerical form.

A TensorFlow `Tokenizer` is created and fitted on the complete dataset. During this process, the tokenizer scans every sentence and builds a vocabulary by assigning a unique integer ID to every unique word.

Example:

```text
good  -> 1
movie -> 2
bad   -> 3
love  -> 4
```

The `texts_to_sequences()` function then replaces every word in each sentence with its corresponding integer ID.

Example:

```text
"I love this movie"

↓

[12, 5, 18, 2]
```

Neural networks cannot process raw text directly, therefore tokenization converts words into numerical representations that can be processed by the model.

---

## Step 2: Padding

Different sentences naturally have different lengths.

Example:

```text
I love AI
↓

[5, 7, 2]

This movie was absolutely amazing
↓

[8, 4, 10, 13, 21]
```

Neural networks process data in batches, where every sequence in a batch must have the same length. Padding standardizes the sequence lengths by adding zeros to shorter sequences.

Example:

```text
[5, 7, 2]

↓

[5, 7, 2, 0, 0]
```

This project uses **post-padding**, meaning zeros are added at the end of shorter sequences.

---

## Step 3: Model Architecture

The model is built using TensorFlow's Sequential API.

### Embedding Layer

The first layer is an Embedding layer.

Instead of sending integer tokens directly into the RNN, the embedding layer converts every token into a dense vector representation.

Example:

```text
Token

15

↓

Embedding Vector

[0.23, -0.61, 0.84, ...]
```

Internally, the embedding layer maintains a trainable embedding matrix. During training, gradient descent updates these vectors so that words with similar meanings gradually obtain similar vector representations.

---

### Simple RNN Layer

The embedding vectors are then passed into a Simple RNN layer.

Unlike a traditional Dense network, an RNN processes one word at a time while maintaining a hidden state that carries information from previously seen words. This allows the network to learn sequential patterns and understand word order within a sentence.

The Simple RNN layer uses the **tanh** activation function to update its hidden state.

---

### Output Layer

The final layer is a Dense layer containing a single neuron with a **sigmoid** activation function.

Since this is a binary classification problem, sigmoid converts the output into a probability between **0 and 1**.

Example:

```text
0.93 → Positive

0.12 → Negative
```

---

## Step 4: Model Training

The padded token sequences are used as the input features, while the sentiment labels serve as the target values.

During training:

1. The input sentences pass through the Embedding layer.
2. The embeddings are processed by the Simple RNN.
3. The Dense layer predicts the sentiment probability.
4. The prediction is compared with the actual label to compute the loss.
5. Backpropagation calculates gradients.
6. Gradient descent updates the trainable parameters of:

   * The Embedding layer
   * The RNN layer
   * The Dense layer

As training progresses, the model gradually learns better word representations and improves its prediction accuracy.

---

## Step 5: Making Predictions

After training, the user can enter a custom sentence.

The prediction pipeline follows the exact same preprocessing steps used during training:

1. Tokenize the input sentence using the same tokenizer.
2. Apply the same padding length.
3. Pass the padded sequence to `model.predict()`.
4. The sigmoid output produces a probability between 0 and 1.

A threshold of **0.5** is used for classification:

* Probability ≥ 0.5 → Positive Sentiment
* Probability < 0.5 → Negative Sentiment

Using the same tokenizer during inference is important because it preserves the vocabulary mapping learned during training.

---

# Important Notes

* This project is intended for educational purposes to understand how an RNN processes text.
* The dataset is intentionally small and is not meant to produce a production-quality sentiment analysis model.
* The model is trained entirely in memory whenever the Python script is executed.
* Since the model is not saved using `model.save()`, its learned weights are discarded once the program terminates. Training is therefore repeated every time the script runs.

---

# Technologies Used

* Python
* TensorFlow / Keras
* NumPy

---

# Learning Outcomes

This project demonstrates practical understanding of:

* Tokenization
* Sequence Padding
* Word Embeddings
* Recurrent Neural Networks (Simple RNN)
* Binary Classification
* Forward Propagation
* Backpropagation
* Gradient Descent
* Model Training and Inference using TensorFlow
