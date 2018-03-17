# Get Started

TensorFlow provides multiple APIs. 

The lowest level API --TensorFlow Core-- provides you with complete programming control. We recommend TensorFlow Core for machine learning researchers and others who require fine levels of control over their models. 

The higher level APIs are built on top of TensorFlow Core. These higher level APIs are typically easier to learn and use than TensorFlow Core. In addition, the higher level APIs make repetitive tasks easier and more consistent between different users. A high-level API like tf.estimator helps you manage data sets, estimators, training and inference.

**Tensors**: A tensor consists of a set of primitive values shaped into an array of any number of dimensions. A tensor's rank is its number of dimensions. 
```python
3 # a rank 0 tensor; a scalar with shape []
[1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

**The Computational Graph**: A computational graph is a series of TensorFlow operations arranged into a graph of nodes. Each node takes zero or more tensors as inputs and produces a tensor as an output.

A graph can be parameterized to accept external inputs, known as **placeholders**. A placeholder is a promise to provide a value later.

We can evaluate this graph with multiple inputs by using the `feed_dict` argument to the `run` method to feed concrete values to the placeholders.
> Important: This tensor will produce an error if evaluated. Its value must be fed using the `feed_dict` optional argument to `Session.run()`, `Tensor.eval()`, or `Operation.run()`.

To make the model trainable, we need to be able to modify the graph to get new outputs with the same input. **Variables** allow us to add trainable parameters to a graph.

Constants are initialized when you call `tf.constant`, and their value can never change. By contrast, variables are not initialized when you call `tf.Variable`. To initialize all the variables:
```python
init = tf.global_variables_initializer()
sess.run(init)
```



# Programmers Guide

## Importing Data

The `Dataset` API enables you to build complex input pipelines from simple, reusable pieces, makes it easy to deal with large amounts of data, different data formats, and complicated transformations.

* A `tf.data.Dataset` represents a sequence of elements, in which each element contains one or more `Tensor` objects. (e.g. in an image pipeline, an element might be a single training example, with a pair of tensors representing the image data and a label)
* 

# Tutorials

# Python API Guides
> <https://www.tensorflow.org/api_guides/python/>

## Reading Data

There are three other methods of getting data into a TensorFlow program: Feeding, Reading from files, and Preloaded data.

### Feeding

TensorFlow's feed mechanism lets you inject data into any Tensor in a computation graph.

A placeholder exists solely to serve as the target of feeds. It is not initialized and contains no data.
Supply feed data through the `feed_dict` argument to a `run()` or `eval()` call that initiates computation.

```python
with tf.Session():
  input = tf.placeholder(tf.float32)
  classifier = ...
  print(classifier.eval(feed_dict={input: my_python_preprocessing_fn()}))
```

> Note: "Feeding" is the least efficient way to feed data into a tensorflow program and should only be used for small experiments and debugging.

### Reading from files

An input pipeline reads the data from files at the beginning of a TensorFlow graph.

Pass the list of filenames to the `tf.train.string_input_producer` function. (a FIFO queue)

e.g. Read csv files with `tf.TextLineReader` and the `tf.decode_csv`.
```python
filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[1], [1], [1], [1], [1]]
col1, col2, col3, col4, col5 = tf.decode_csv(
    value, record_defaults=record_defaults)
features = tf.stack([col1, col2, col3, col4])

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1200):
    # Retrieve a single instance:
    example, label = sess.run([features, col5])

  coord.request_stop()
  coord.join(threads)
```

> Many of the `tf.train` functions listed above add `tf.train.QueueRunner` objects to your graph. These require that you call `tf.train.start_queue_runners` before running any training or inference steps, or it will hang forever. This will start threads that run the input pipeline, filling the example queue so that the dequeue to get the examples will succeed. This is best combined with a `tf.train.Coordinator` to cleanly shut down these threads when there are errors.

### Preloaded data

This is only used for small data sets that can be loaded entirely in memory. There are two approaches:
* Store the data in a constant.
* Store the data in a variable, that you initialize (or assign to) and then never change. (trainable=False)


## Neural Network
