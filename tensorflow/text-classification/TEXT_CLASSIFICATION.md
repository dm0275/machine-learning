# Simple Text Classification
## Pre-reqs
* `TensorFlow`
* `Keras`
* `Matplotlib`

## Steps
1. Use `tf.keras.utils.get_file ` to download and extract the IMDB dataset which contains 50,000 movie reviews
   * ```python
     dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='datasets')
     ```
2. Generate the training, validation, and testing datasets
    * ```python
      batch_size = 32
      seed = 42
     
      raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        directory=dataset_dir + "/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=seed)

      raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        directory=dataset_dir + "/train",
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=seed)

      raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        directory=dataset_dir + "/test",
        batch_size=batch_size)
      ```
    * `tf.keras.utils.text_dataset_from_directory` is used to generate a `tf.data.Dataset` object from text files in a directory.
3. Create a custom text standardization function
    * ```python
      def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(input=stripped_html,
                                        pattern='[%s]' % re.escape(string.punctuation),
                                        rewrite='')
      ```
    * This function is used to pre-process the text data, converts text to lowercase, removes HTML tags ('\<br \/>'), and remove punctuation.
4. A pre-processing layer is created using `layers.TextVectorization` to convert text into sequences of integers.
    * ```python
      max_features = 10000
      sequence_length = 250

      vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)
      ```
    * It uses the `custom_standardization` function created in step 3 to standardize the input text and limits the vocabulary to the top 10,000 words.
    * The output sequences are padded or truncated to have a fixed length of 250.
5. Use a mapping function to extract the text samples and create a new dataset 
    * ```python
      train_text = raw_train_ds.map(lambda x, y: x)
      ```
    * In this mapping function, the lambda function (lambda x, y: x) is used to extract only the text samples (movie reviews) from each pair, ignoring the labels (y).
6. Adapt the `vectorize_layer` using th training text
   * ```python
      vectorize_layer.adapt(train_text)
     ```
   * `vectorize_layer` is an instance of `layers.TextVectorization` (created in step 4), it is used to convert text data into sequences of integers.
   * `adapt` is a method of `TextVectorization` that is used to adapt the vectorization layer to the training data. It computes vocabulary statistics and tokenizes the text data based on the training dataset it receives.
   * It is essentially telling the vectorization layer to analyze the text in `train_text` and build its vocabulary and tokenization rules based on this training data.
7. Define a function to pre-process a single text sample
   * ```python
      def vectorize_text(text, label):
         text = tf.expand_dims(text, -1)
         return vectorize_layer(text), label
     ```
   * Reshape the input text using `tf.expand_dims(text, -1)`, this function adds a new dimension to the `text` tensor along its last axis (axis -1). Essentially, it converts a 1D tensor (representing a single text sample) into a 2D tensor (representing a batch of one text sample).
     * `layers.TextVectorization`, expects input data to be in the form of a batch of sequences. It assumes that you are providing multiple text sequences as input, and it processes them in a batched fashion.
     * Before: `text` is a 1D tensor with shape `(sequence_length,)`, where `sequence_length` is the length of the text.
     * After: `tf.expand_dims(text, -1)` converts it into a 2D tensor with shape `(sequence_length, 1)`. This 2D tensor now represents a batch of one sequence.
     * This is needed to adapt the shape of the input `text` tensor to match the batching expectations of `vectorize_layer`, even when you are processing a single text sample. It ensures consistency and compatibility in how the text is processed by the vectorization layer.
   * `vectorize_layer(text)` is invoking the `call` method of the `TextVectorization` layer.
     * When you call an instance of a layer (ex. `vectorize_layer`) with an input tensor (ex `text`), it implicitly calls the `call` method of that layer to process the input and produce the output.
     * The `call` method of the `TextVectorization` layer performs the vectorization process, which includes tokenization, mapping words to integers based on its vocabulary, and padding/truncating sequences to the specified length. It essentially applies the layer's transformations to the input data.
8. Apply the `vectorize_text` function to the raw datasets to pre-process and vectorize the data
    * ```python
      train_ds = raw_train_ds.map(vectorize_text)
      val_ds = raw_val_ds.map(vectorize_text)
      test_ds = raw_test_ds.map(vectorize_text)
      ```
    * `raw_<data_type>_ds`(ex. `raw_train_ds`) is the raw dataset that contains pairs of text samples and their corresponding labels.
    * `raw_train_ds.map(vectorize_text)` applies the `vectorize_text` function to each element (text sample and label pair) in the training dataset.
    * The result is that the text samples in the training dataset are processed and transformed into integer sequences using the preconfigured `vectorize_layer`. The labels remain unchanged.
    * The processed data is stored in the `train_ds` dataset, which is now ready for training.
9. Optimize the data for training
    * ```python
      AUTOTUNE = tf.data.AUTOTUNE

      train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
      val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
      test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
      ```
    * Setting `AUTOTUNE` allows TensorFlow to automatically adjust the degree of parallelism during data loading and preprocessing operations. This helps TensorFlow to use available CPU and memory resources efficiently.
    * `.cache()` applies caching to the dataset. Caching stores a portion of the dataset in memory after the first epoch, which can speed up data loading during subsequent epochs.
    * `.prefetch(buffer_size=AUTOTUNE)` applies prefetching to the dataset. Prefetching allows the data pipeline to prefetch a specified number of batches of data ahead of the current batch during model training. The buffer_size parameter determines the number of batches to prefetch.
      * In this script, `AUTOTUNE` is used as the `buffer_size`, which means TensorFlow will automatically determine the optimal prefetching buffer size based on available system resources.
10. Create a sequential model with 5 layers
   * ```python
      embedding_dim = 16

      model = tf.keras.Sequential([
         layers.Embedding(max_features, embedding_dim),
         layers.Dropout(0.2),
         layers.GlobalAveragePooling1D(),
         layers.Dropout(0.2),
         layers.Dense(1)])
     ```
   * `tf.keras.Sequential` creates a Keras Sequential model, which is a linear stack of layers. The layers are defined within the square brackets `[]` as a list.
     * The model will process the input data sequentially through the specified layers.
   * `layers.Embedding(max_features, embedding_dim)` The first layer in the model is an embedding layer (`layers.Embedding`). It is responsible for converting integer-encoded words (word IDs) into dense vector representations (embeddings).
     * `max_features` is the maximum number of words in the vocabulary, and `embedding_dim` is the dimensionality of the embeddings. In this case, it is set to 16, which means each word in the text will be represented as a 16-dimensional vector in the embedding layer.
     * The embedding layer will be used to learn word representations specific to the task at hand.
   * `layers.Dropout(0.2)` this dropout layer is added with a dropout rate of `0.2`. 
     * Dropout is a regularization technique that helps prevent overfitting by randomly "dropping out" a fraction of the neurons during training, reducing the model's reliance on any single neuron.
   * `layers.GlobalAveragePooling1D()` layer is added to compute the average of the embeddings across all word positions in each input sequence. 
     * This reduces the sequence of embeddings to a fixed-length vector, which captures the overall sentiment of the input text.
     * It helps in reducing the dimensionality of the data and summarizing the information from the entire sequence.
   * `layers.Dropout(0.2)` Another dropout layer is added after the global average pooling layer with the same dropout rate of 0.2.
   * `layers.Dense(1)` The final layer in the model is a dense layer with a single unit.
     * This layer is responsible for making the binary classification prediction, where a value close to 1 indicates a positive sentiment, and a value close to 0 indicates a negative sentiment.
11. 

## Links
* https://www.tensorflow.org/tutorials/keras/text_classification
* https://www.tensorflow.org/api_docs/python/tf/data/Dataset