> ref: <https://web.stanford.edu/class/cs20si/>

# Overview of Tensorflow
TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.

**Simplified TensorFlow?**
1. TF Learn (tf.contrib.learn): simplified interface that helps users
transition from the the world of one-liner such as scikit-learn
2. TF Slim (tf.contrib.slim): lightweight library for defining, training and
evaluating complex models in TensorFlow.
3. High level API: Keras, TFLearn, Pretty Tensor

But we don’t need baby TensorFlow ...
Off-the-shelf models are not the main purpose of TensorFlow.
TensorFlow provides an extensive suite of functions and classes that allow users to
define models from scratch.
And this is what we are going to learn.

**Data Flow Graphs**
> ref: <https://danijar.com/what-is-a-tensorflow-session/>

TensorFlow separates definition of computations from their execution
* Phase 1: assemble a graph
* Phase 2: use a session to execute operations in the graph.

Tensor: An n-dimensional array (Tensors are data)

* Nodes: operators, variables, and constants
* Edges: tensors

```python
import tensorflow as tf
a = tf.add(3, 5)
print a # >> Tensor("Add:0", shape=(), dtype=int32)

sess = tf.Session()
print sess.run(a) # >> 8
sess.close()

# or ~
with tf.Session() as sess:
    print sess.run(a)
```

```python
x = 2
y = 3
add_op = tf.add(x, y)
mul_op = tf.mul(x, y)
useless = tf.mul(x, add_op)
pow_op = tf.pow(add_op, mul_op)
with tf.Session() as sess:
z = sess.run(pow_op)

# Because we only want the value of pow_op and pow_op doesn’t
depend on useless, session won’t compute value of useless
# → save computation
```

**Distributed Computation**

Possible to break graphs into several chunks and run them parallelly across multiple CPUs, GPUs, or devices
```python
# Creates a graph.
with tf.device('/gpu:2'):
 a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
 b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
 c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print sess.run(c)
```

**Graphs and Sessions**
```
g = tf.Graph()
with g.as_default():
  a = 3
  b = 5
  x = tf.add(a, b)

sess = tf.Session(graph=g) # session is run on the graph g
# run session
sess.close()
```



# Operations

## Fun with TensorBoard
```python
import tensorflow as tf
a = tf.constant(2, name="a")
b = tf.constant(3, name="b")
x = tf.add(a, b, name="add")
writer = tf.summary.FileWriter("./graphs", sess.graph)
with tf.Session() as sess:
  print(sess.run(x))
  writer.close() # close the writer when you’re done using it
```
```
$ python [yourprogram].py
$ tensorboard --logdir="./graphs" --port 6006
Then open your browser and go to: http://localhost:6006/
```

## Constant types
```python
# create constants of scalar or tensor values.
tf​.​constant​(​value​,​ dtype​=​None​,​ shape​=​None​,​ name​=​'Const'​,​ verify_shape​=​False)

# create tensors whose elements are of a specific value
tf​.​zeros​(​shape​,​ dtype​=​tf​.​float32​,​ name​=​None)
tf​.​zeros_like​(​input_tensor​,​ dtype​=​None​,​ name​=​None​,​ optimize​=​True)
tf​.​ones​(​shape​,​ dtype​=​tf​.​float32​,​ name​=​None)
tf​.​ones_like​(​input_tensor​,​ dtype​=​None​,​ name​=​None​,​ optimize​=​True)
tf​.​fill​(​dims​,​ value​,​ name​=​None​)

# create constants that are sequences
tf​.​linspace​(​start​,​ stop​,​ num​,​ name​=​None)
tf​.​range​(​start​,​ limit​=​None​,​ delta​=​1​,​ dtype​=​None​,​ name​=​'range')
# Note that unlike NumPy or Python sequences, TensorFlow sequences are not iterable.

# generate random constants from certain distributions
tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
name=None)
tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
name=None)
tf.random_shuffle(value, seed=None, name=None)
tf.random_crop(value, size, seed=None, name=None)
tf.multinomial(logits, num_samples, seed=None, name=None)
tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
```

## Math Operations
TensorFlow math ops are pretty standard, quite similar to NumPy. Visit <https://www.tensorflow.org/api_guides/python/math_ops> for more because listing math ops is boring.

## Data Types
* Python Native Types:
TensorFlow takes in Python native types such as Python boolean values, numeric values
(integers, floats), and strings. Values will be converted to n-d tensors.
* TensorFlow Native Types
Like NumPy, TensorFlow also its own data types as you’ve seen tf.int32, tf.float32.
* NumPy Data Types

## Variables
Constants are stored in the graph definition. When constants are
memory expensive, it will be slow each time you have to load the graph.

**Declare variables**

To declare a variable, you create an instance of the class tf.Variable. Note that it’s tf.constant but
tf.Variable and not tf.variable because tf.constant is an op, while tf.Variable is a class.

tf.Variable holds several ops:
```python
x = tf.Variable(...)
x.initializer # init
x.value() # read op
x.assign(...) # write op
x.assign_add(...)
# and more
```

**You have to initialize variables before using them**
```python
# The easiest way is initializing all variables at once using tf.global_variables_initializer()
init = tf.global_variables_initializer()
with tf.Session() as sess:
  tf.run(init)

# To initialize only a subset of variables using tf.variables_initializer()
init_ab = tf.variables_initializer([a, b], name="init_ab")
with tf.Session() as sess:
  tf.run(init_ab)

# You can also initialize each variable separately using tf.Variable.initializer
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
  tf.run(W.initializer)
```

**Evaluate values of variables**
```python
# To get the value of a variable, we need to evaluate it using eval()
W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
  sess.run(W.initializer)
  print W.eval()
```

**Assign values to variables**
```python
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
  sess.run(W.initializer)
  print W.eval() # >> 10
  sess.run(assign_op)
  print W.eval() # >> 100
```

**You can, of course, declare a variable that depends on other variables**
```python
W = tf.Variable(tf.truncated_normal([700, 10]))
# use initialized_value() to make sure that W is initialized
U = tf.Variable(W.intialized_value() * 2)
```

## InteractiveSession
```python
sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval()' without passing 'sess'
print(c.eval())
sess.close()
```
> Use tf.Session rather than tf.InteractiveSession. This better separates the process of creating the graph (model specification) and the process of evaluating the graph (model fitting). It generally makes for cleaner code.

## Control Dependencies
```python
# your graph g have 5 ops: a, b, c, d, e
with g.control_dependencies([a, b, c]):
  # `d` and `e` will only run after `a`, `b`, and `c` have executed.
  d = ...
  e = …
```

## Placeholders and feed_dict
In short, you use tf.Variable for trainable variables such as weights (W) and biases (B) for your model. tf.placeholder is used to feed actual training examples.
```python
# define
tf.placeholder(dtype, shape=None, name=None)
```
```python
# create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)
# use the placeholder as you would a constant or a variable
c = a + b # Short for tf.add(a, b)
with tf.Session() as sess:
  # feed [1, 2, 3] to placeholder a via the dict {a: [1, 2, 3]}
  # fetch value of c
  print(sess.run(c, {a: [1, 2, 3]}))
```

## Avoid lazy loading
Separate the assembling of graph and executing ops



# Linear and Logistic Regression

## Linear Regression in TensorFlow
```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
DATA_FILE = "data/fire_theft.xls"

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of
theft)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")

# Step 4: construct model to predict Y (number of theft) from the number of fire
Y_predicted = X * w + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name="loss")

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
  # Step 7: initialize the necessary variables, in this case, w and b
  sess.run(tf.global_variables_initializer())

  # Step 8: train the model
  for i in range(100): # run 100 epochs
    for x, y in data:
    # Session runs train_op to minimize loss
    sess.run(optimizer, feed_dict={X: x, Y:y})
    
  # Step 9: output the values of w and b  
  w_value, b_value = sess.run([w, b])
```

**Optimizers**

By default, the optimizer trains all the trainable variables whose objective function depend on. If
there are variables that you do not want to train, you can set the keyword trainable to False
when you declare a variable. One example of a variable you don’t want to train is the variable
global_step, a common variable you will see in many TensorFlow model to keep track of how
many times you’ve run your model.


You can also modify the gradients calculated by your optimizer.
```python
# create an optimizer.
optimizer = GradientDescentOptimizer(learning_rate=0.1)
# compute the gradients for a list of variables.
grads_and_vars = opt.compute_gradients(loss, <list of variables>)
# grads_and_vars is a list of tuples (gradient, variable). Do whatever you
# need to the 'gradient' part, for example, subtract each of them by 1.
subtracted_grads_and_vars = [(gv[0] - 1.0, gv[1]) for gv in grads_and_vars]
# ask the optimizer to apply the subtracted gradients.
optimizer.apply_gradients(subtracted_grads_and_vars)
```

**List of optimizers**
```
tf.train.GradientDescentOptimizer
tf.train.AdadeltaOptimizer
tf.train.AdagradOptimizer
tf.train.AdagradDAOptimizer
tf.train.MomentumOptimizer
tf.train.AdamOptimizer
tf.train.FtrlOptimizer
tf.train.ProximalGradientDescentOptimizer
tf.train.ProximalAdagradOptimizer
tf.train.RMSPropOptimizer
```

## Logistic Regression in TensorFlow
Let’s illustrate logistic regression in TensorFlow solving the good old classifier on the MNIST database.
```python
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
MNIST = input_data.read_data_sets("/data/mnist", one_hot=True)

# Step 2: Define parameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 25

# Step 3: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9.
# each label is one hot vector.
X = tf.placeholder(tf.float32, [batch_size, 784])
Y = tf.placeholder(tf.float32, [batch_size, 10])

# Step 4: create weights and bias
# w is initialized to random variables with mean of 0, stddev of 0.01
# b is initialized to 0
# shape of w depends on the dimension of X and Y so that Y = tf.matmul(X, w)
# shape of b depends on Y
w = tf.Variable(tf.random_normal(shape=[784, 10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")

# Step 5: predict Y from X and w, b
# the model that returns probability distribution of possible label of the image
# through the softmax layer
# a batch_size x 10 tensor that represents the possibility of the digits
logits = tf.matmul(X, w) + b

# Step 6: define loss function
# use softmax cross entropy with logits as the loss function
# compute mean cross entropy, softmax is applied internally
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy) # computes the mean over examples in the batch

# Step 7: define training op
# using gradient descent with learning rate of 0.01 to minimize cost
optimizer = tf.train.   GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)
  n_batches = int(MNIST.train.num_examples/batch_size)
  for i in range(n_epochs): # train the model n_epochs times
    for _ in range(n_batches):
      X_batch, Y_batch = MNIST.train.next_batch(batch_size)
      sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch})
      # average loss should be around 0.35 after 25 epochs
  
  # test the model
  n_batches = int(MNIST.test.num_examples/batch_size)
  total_correct_preds = 0
  for i in range(n_batches):
    X_batch, Y_batch = MNIST.test.next_batch(batch_size)
    _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], feed_dict={X: X_batch, Y:Y_batch})
    preds = tf.nn.softmax(logits_batch)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # similar to numpy.count_nonzero(boolarray) :(
    total_correct_preds += sess.run(accuracy)

  print("Accuracy {0}".format(total_correct_preds/MNIST.test.num_examples))
```

# Structure your TensorFlow model

How to structure your TensorFlow model?

**Phase 1: assemble your graph**
1. Define placeholders for input and output
2. Define the weights
3. Define the inference model
4. Define loss function
5. Define optimizer

**Phase 2: execute the computation (training your model)**
1. Initialize all model variables for the first time.
2. Feed in the training data. Might involve randomizing the order of data samples.
3. Execute the inference model on the training data, so it calculates for each training input
example the output with the current model parameters.
4. Compute the cost
5. Adjust the model parameters to minimize/maximize the cost depending on the model.

through an example: word2vec.

>Skip​-​gram vs CBOW​ (Continuous Bag-of-Words)
>
>Algorithmically, these models are similar, except that **CBOW predicts center words from
context words**, while the **skip-gram does the inverse and predicts source context-words from
the center words**. For example, if we have the sentence: ""The quick brown fox jumps"", then
CBOW tries to predict ""brown"" from ""the"", ""quick"", ""fox"", and ""jumps"", while
skip-gram tries to predict ""the"", ""quick"", ""fox"", and ""jumps"" from ""brown"".
>
>Statistically it has the effect that CBOW smoothes over a lot of the distributional
information (by treating an entire context as one observation). For the most part, this
turns out to be a useful thing for smaller datasets. However, skip-gram treats each
context-target pair as a new observation, and this tends to do better when we have larger
datasets.

> [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

> [理解 Word2Vec 之 Skip-Gram 模型](https://zhuanlan.zhihu.com/p/27234078)

whole word2vec program
```python
# Step 1: define the placeholders for input and output
with tf.name_scope("data"):
  center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
  target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

# Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
with tf.device('/cpu:0'):
  with tf.name_scope("embed"):
    # Step 2: define weights. In word2vec, it's actually the weights that we care about
    embed_matrix = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0), name='embed_matrix')

  # Step 3 + 4: define the inference + the loss function
  with tf.name_scope("loss"):
    # Step 3: define the inference
    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
    # Step 4: construct variables for NCE loss
    nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],stddev=1.0 / math.sqrt(EMBED_SIZE)), name='nce_weight')
    nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')
    # define loss function to be NCE loss function
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, biases=nce_bias, labels=target_words, inputs=embed, num_sampled=NUM_SAMPLED, num_classes=VOCAB_SIZE), name='loss')

  # Step 5: define optimizer
  optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    average_loss = 0.0
    for index in xrange(NUM_TRAIN_STEPS):
      batch = batch_gen.next()
      loss_batch, _ = sess.run([loss, optimizer], feed_dict={center_words: batch[0], target_words: batch[1]})
      average_loss += loss_batch
      if (index + 1) % 2000 == 0:
        print('Average loss at step {}: {:5.1f}'.format(index + 1, average_loss / (index + 1)))
```

# Managing experiments and process data [not finished]

## tf.train.Saver()
A good practice is to periodically save the model’s parameters after a certain number of steps
so that we can restore/retrain our model from that step if need be. 
```python
saver = tf.train.Saver() # defaults to saving all variables
tf.train.Saver.save(sess, save_path, global_step=None, latest_filename=None, meta_graph_suffix='meta', write_meta_graph=True, write_state=True)
tf.train.Saver.save.restore(sess, save_path)
```

## tf.summary
TensorBoard provides us with a great set of tools to visualize our
summary statistics during our training.
```python
def _create_summaries(self):
  with tf.name_scope("summaries"):
    tf.summary.scalar("loss", self.loss
    tf.summary.scalar("accuracy", self.accuracy)
    tf.summary.histogram("histogram loss", self.loss)
    # because you have several summaries, we should merge them all
    # into one op to make it easier to manage
    self.summary_op = tf.summary.merge_all()

# Because it’s an op, you have to execute it with sess.run()
loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)
# you need to write the summary to file
writer.add_summary(summary, global_step=step)
```

## Control randomization
> <http://devdocs.io/tensorflow~python/tf/set_random_seed>

Set random seed at operation level. All random tensors allow you to pass in seed value in
their initialization.
```python
my_var = tf.Variable(tf.truncated_normal((-1.0,1.0), stddev=0.1, seed=0))
c = tf.random_uniform([], -10, 10, seed=2)
with tf.Session() as sess:
  print sess.run(c) # >> 3.57493
  print sess.run(c) # >> -5.97319
```

Set random seed at graph level with `tf.Graph.seed`.
```python
tf.set_random_seed(seed)
```

## Reading Data in TensorFlow
There are two main ways to load data into a TensorFlow graph: one is through feed_dict that we
are familiar with, and another is through readers that allow us to read tensors directly from file.

Feed_dict will first send data from the storage system to the client, and then
from client to the worker process. This will cause the data to slow down, especially if the client is
on a different machine from the worker process. TensorFlow has readers that allow us to load
data directly into the worker process.

TensorFlow has several built in readers:
```
tf.TextLineReader
Outputs the lines of a file delimited by newlines
E.g. text files, CSV files

tf.FixedLengthRecordReader
Outputs the entire file when all files have same fixed lengths
E.g. each MNIST file has 28 x 28 pixels, CIFAR-10 32 x 32 x 3

tf.WholeFileReader
Outputs the entire file content

tf.TFRecordReader
Reads samples from TensorFlow's own binary format (TFRecord)

tf.ReaderBase
Allows you to create your own readers
```


# Convolutional Neural Networks [draft]
> mathematical term convolution: a function derived from two given functions by integration that expresses how the shape of one is modified by the other.
> Convolution is how the original input (in the first convolutional layer, it’s part of the original image) is modified by the kernel (or filter)

> [如何通俗易懂地解释卷积？ - 知乎](https://www.zhihu.com/question/22298352)

TensorFlow has great support for convolutional layers. The most popular one is tf.nn.conv2d.
```python
tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
Input​: Batch size x Height x Width x Channels
Filter​: Height x Width x Input Channels x Output Channels
(e.g. [5, 5, 3, 64])
Strides​: 4 element 1-D tensor, strides in each direction
(often [1, 1, 1, 1] or [1, 2, 2, 1])
Padding​: 'SAME' or 'VALID'
Data_format​: default to NHWC
```

**Convnet on MNIST**

![](https://media.licdn.com/mpr/mpr/AAEAAQAAAAAAAAm8AAAAJDJmZDVmMjIwLTcxZDAtNDgyYy05ZjMxLTE3ZmM0MjA1ZDUyMg.png)

**Variable scope**

Think of a variable scope something similar to a namespace. A variable name ‘weights’ in variable scope
‘conv1’ will become ‘conv1-weights’. 

In variable scope, we don’t create variable using tf.Variable, but instead use tf.get_variable()
```python
tf.get_variable(<name>, <shape>, <initializer>)
```
If a variable with that name already exists in that variable scope, we use that variable. If a
variable with that name doesn’t already exists in that variable scope, TensorFlow creates a new
variable.

Nodes in the same variable scope will be grouped together, and therefore you don’t have to use
name scope any more. To declare a variable scope, you do it the same way you do name
scope:
```python
with tf.variable_scope('conv1') as scope:

# For example:
with tf.variable_scope('conv1') as scope:
 w = tf.get_variable('weights', [5, 5, 1, 32])
 b = tf.get_variable('biases', [32], initializer=tf.random_normal_initializer())
 conv = tf.nn.conv2d(images, w, strides=[1, 1, 1, 1], padding='SAME')
 conv1 = tf.nn.relu(conv + b, name=scope.name)
with tf.variable_scope('conv2') as scope:
 w = tf.get_variable('weights', [5, 5, 32, 64])
 b = tf.get_variable('biases', [64], initializer=tf.random_normal_initializer())
 conv = tf.nn.conv2d(conv1, w, strides=[1, 1, 1, 1], padding='SAME')
 conv2 = tf.nn.relu(conv + b, name=scope.name)
```

# Input Pipeline

## Queues and Coordinators
In TensorFlow documentation, queues are described as “important TensorFlow objects for computing tensors
asynchronously in a graph.”

in using queues to prepare inputs for training a model, we have:
* Multiple threads prepare training examples and push them in the queue.
* A training thread executes a training op that dequeues mini-batches from the queue

The TensorFlow Session object is designed multithreaded, so multiple threads can easily use the
same session and run ops in parallel. 

TensorFlow provides two classes to help with the threading: `tf.Coordinator​` and `tf.train.QueueRunner​`. These two classes are designed to be used together.
The Coordinator class helps multiple threads stop together and report exceptions to a program that waits for them to stop.

There are two main queue classes, `tf.FIFOQueue`, `tf.RandomShuffleQueue`, `tf.PaddingFIFOQueue` and `tf.PriorityQueue`.
The QueueRunner class is used to create a number of threads cooperating to enqueue tensors in the same queue. These two queues support the `enqueue`, `enqueue_many`, and `dequeue`.

```python
N_SAMPLES = 1000
NUM_THREADS = 4
# Generating some simple data
# create 1000 random samples, each is a 1D array from the normal distribution (10, 1)
data = 10 * np.random.randn(N_SAMPLES, 4) + 1
# create 1000 random labels of 0 and 1
target = np.random.randint(0, 2, size=N_SAMPLES)
queue = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.int32], shapes=[[4], []])
enqueue_op = queue.enqueue_many([data, target])
dequeue_op = queue.dequeue()
# create NUM_THREADS to do enqueue
qr = tf.train.QueueRunner(queue, [enqueue_op] * NUM_THREADS)
with tf.Session() as sess:
  # Create a coordinator, launch the queue runner threads.
  coord = tf.train.Coordinator()
  enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
  for step in xrange(100): # do to 100 iterations
    if coord.should_stop():
      break
    data_batch, label_batch = sess.run(dequeue_op)
  coord.request_stop()
  coord.join(enqueue_threads)
```

You also don’t need to use tf.Coordinator with TensorFlow queues, but can use it to manage
threads of any thread you create. 
```python
import threading

# thread body: loop until the coordinator indicates a stop was requested.
# if some condition becomes true, ask the coordinator to stop.

def my_loop(coord):
  while not coord.should_stop():
    ...do something...
  if ...some condition...:
    coord.request_stop()

# main code: create a coordinator.
coord = tf.Coordinator()

# create 10 threads that run 'my_loop()'
# you can also create threads using QueueRunner as the example above
threads = [threading.Thread(target=my_loop, args=(coord,)) for _ in xrange(10)]

# start the threads and wait for all of them to stop.
for t in threads:
  t.start()
coord.join(threads)
```

## Data Readers
> see above

# Introduction to RNN, LSTM, GRU

# Convolutional-GRU

# Seq2seq with Attention

# Reinforcement Learning in Tensorflow

# Chatbot demo  