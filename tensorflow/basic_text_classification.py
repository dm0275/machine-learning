import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from keras import layers
from keras import losses

# Dataset URL
url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

# tf.keras.utils.get_file - Downloads a file from a URL if it not already in the cache.
#     By default the file at the url `origin` is downloaded to the
#     cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
#     and given the filename `fname`. The final location of a file
#     `example.txt` would therefore be `~/.keras/datasets/example.txt`.
#
#     Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
#     Passing a hash will verify the file after download. The command line
#     programs `shasum` and `sha256sum` can compute the hash.
dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True, cache_dir='.', cache_subdir='datasets')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')

print(dataset_dir)
print(os.listdir(dataset_dir))
print(os.listdir(train_dir))

sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
with open(sample_file) as f:
    print(f.read())

remove_dir = os.path.join(train_dir, 'unsup')
# Recursively delete a directory tree
shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

# Generates a `tf.data.Dataset` from text files in a directory.
raw_train_ds = tf.keras.utils.text_dataset_from_directory(
    # Directory where the data is located.
    directory=dataset_dir + "/train",
    # Size of the batches of data
    batch_size=batch_size,
    # fraction of data to reserve for validation
    validation_split=0.2,
    # subset: Subset of the data to return. One of `"training"`, `"validation"` or `"both"`. Only used if
    # `validation_split` is set.
    subset='training',
    # random seed for shuffling and transformations
    seed=seed)

# Dataset methods: https://www.tensorflow.org/api_docs/python/tf/data/Dataset#take
for text_batch, label_batch in raw_train_ds.take(1):
    for i in range(3):
        print("Review: " + text_batch.numpy()[i].decode("utf-8"))
        print("Label: " + str(label_batch.numpy()[i]))

print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
    # Directory where the data is located.
    directory=dataset_dir + "/train",
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    # Directory where the data is located.
    directory=dataset_dir + "/test",
    batch_size=batch_size)


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(input=stripped_html,
                                    # string or scalar string `Tensor`, regular expression to use
                                    pattern='[%s]' % re.escape(string.punctuation),
                                    #  string or scalar string `Tensor`, value to use in match replacement
                                    rewrite='')


max_features = 10000
sequence_length = 250

# A preprocessing layer which maps text features to integer sequences.
# This layer has basic options for managing text in a Keras model. It
#     transforms a batch of strings (one example = one string) into either a list
#     of token indices (one example = 1D tensor of integer token indices) or a
#     dense representation (one example = 1D tensor of float values representing
#     data about the example's tokens).
vectorize_layer = layers.TextVectorization(
    # standardize: standardization to apply to the input text. Values can be:
    #  - `None`: No standardization.
    #  - `"lower_and_strip_punctuation"`: Text will be lowercased and all punctuation removed.
    #  - `"lower"`: Text will be lowercased.
    #  - `"strip_punctuation"`: All punctuation will be removed.
    #  - Callable: Inputs will passed to the callable function, which should be standardized and returned.
    standardize=custom_standardization,
    # Maximum size of the vocabulary for this layer
    max_tokens=max_features,
    # specification for the output of the layer. Values can be `"int"`, `"multi_hot"`, `"count"` or `"tf_idf"`,
    # configuring the layer as follows:
    #  - `"int"`: Outputs integer indices, one integer index per split string token. When `output_mode == "int"`, 0
    #  is reserved for masked locations; this reduces the vocab size to `max_tokens - 2` instead of `max_tokens - 1`.
    #  - `"multi_hot"`: Outputs a single int array per batch, of either vocab_size or max_tokens size, containing 1s
    #  in all elements where the token mapped to that index exists at least once in the batch item.
    #  - `"count"`: Like `"multi_hot"`, but the int array contains a count of the number of times the token at that
    #  index appeared in the batch item.
    #  - `"tf_idf"`: Like `"multi_hot"`, but the TF-IDF algorithm is applied to find the value in each token slot.
    output_mode='int',
    # Only valid in INT mode. If set, the output will have its time dimension padded or truncated to exactly
    # `output_sequence_length` values, resulting in a tensor of shape `(batch_size, output_sequence_length)` regardless
    # of how many tokens resulted from the splitting step. Defaults to `None`.
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)

# adapt is used to fit the state of the preprocessing layer to the dataset. This will cause the model to build an
# index of strings to integers.
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

# cache() - Caches the elements in this dataset. The first time the dataset is iterated over, its elements will be
# cached either in the specified file or in memory. Subsequent iterations will use the cached data.
#
# prefetch() - Creates a `Dataset` that prefetches elements from this dataset. Most dataset input pipelines should
# end with a call to `prefetch`. This allows later elements to be prepared while the current element is being
# processed. This often improves latency and throughput, at the cost of using additional memory to store prefetched
# elements.
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

# Creates a `Sequential` model instance.
model = tf.keras.Sequential([
    # Embedding() - Turns positive integers (indexes) into dense vectors of fixed size. This layer can only be used
    # on positive integer inputs of a fixed range. The `tf.keras.layers.TextVectorization`,
    # `tf.keras.layers.StringLookup`, and `tf.keras.layers.IntegerLookup` preprocessing layers can help prepare
    # inputs for an `Embedding` layer.
    layers.Embedding(max_features, embedding_dim),
    # Dropout() - The Dropout layer randomly sets input units to 0 with a frequency of `rate` at each step during
    # training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the
    # sum over all inputs is unchanged.
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(1)])

model.summary()

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
