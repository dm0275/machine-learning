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
7. 

## Links
* https://www.tensorflow.org/tutorials/keras/text_classification
* https://www.tensorflow.org/api_docs/python/tf/data/Dataset