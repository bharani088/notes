# Get Started

TensorFlow provides a programming stack consisting of multiple API layers:

![](https://www.tensorflow.org/images/tensorflow_programming_environment.png)

We strongly recommend writing TensorFlow programs with the following APIs:

- **Estimators**, which represent **a complete model**. The Estimator API provides methods to train the model, to judge the model's accuracy, and to generate predictions.
- **Datasets**, which build **a data input pipeline**. The Dataset API has methods to load and manipulate data, and feed it into your model. The Dataset API meshes well with the Estimators API.

## Overview of programming with Estimators

An Estimator is any class derived from `tf.estimator.Estimator`. TensorFlow provides a collection of pre-made Estimators.

To write a TensorFlow program based on pre-made Estimators, you must perform the following tasks:
1. Create one or more **input functions**.
2. Define the model's **feature columns**.
3. **Instantiate an Estimator**, specifying the feature columns and various hyperparameters.
4. **Call one or more methods on the Estimator object**, passing the appropriate input function as the source of the data.

### Create input functions

**An input function is a function that returns a `tf.data.Dataset` object which outputs the following two-element tuple**:

* `features` - A Python **dictionary** in which:
  * Each key is the name of a feature.
  * Each value is an array containing all of that feature's values.
* `label` - An **array** containing the values of the label for every example.

```py
# Just to demonstrate the format of the input function,
# here's a simple implementation:
def input_evaluation_set():
    features = {'SepalLength': np.array([6.4, 5.0]),
                'SepalWidth':  np.array([2.8, 2.3]),
                'PetalLength': np.array([5.6, 3.3]),
                'PetalWidth':  np.array([2.2, 1.0])}
    labels = np.array([2, 1])
    return features, labels
```

Your input function may generate the `features` dictionary and `label` list any way you like. However, **we recommend using TensorFlow's `Dataset` API, which can parse all sorts of data**.

```py
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)
```

![](https://www.tensorflow.org/images/dataset_classes.png)

### Define the feature columns

The **`tf.feature_column` module** provides many options describing **how the model should use raw input data from the features dictionary**.

```py
# Feature columns describe how to use the input.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

### Instantiate an estimator

```py
# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=3)
```

TensorFlow provides several **pre-made** classifier Estimators, including:

* `tf.estimator.DNNClassifier` for deep models that perform multi-class classification.
* `tf.estimator.DNNLinearCombinedClassifier` for wide & deep models.
* `tf.estimator.LinearClassifier` for classifiers based on linear models.

### Train, Evaluate, and Predict

```py
# Train the Model.
classifier.train(
    input_fn=lambda:iris_data.train_input_fn(train_x, train_y, args.batch_size),
    steps=args.train_steps)

# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:iris_data.eval_input_fn(test_x, test_y, args.batch_size))
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Generate predictions from the model
expected = ['Setosa', 'Versicolor', 'Virginica']
predict_x = {
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
}
predictions = classifier.predict(
    input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                            batch_size=args.batch_size))

for pred_dict, expec in zip(predictions, expected):
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print(template.format(iris_data.SPECIES[class_id],
                          100 * probability, expec))
```

Train the model by calling the Estimator's `train` method.

When the model has been trained, we can get some statistics on its performance  by calling the Estimator's `evaluate` method.

The `predict` method returns a Python iterable, yielding a dictionary of prediction results for each example.

## Details

### Checkpoints

Estimators automatically write the following to disk:

* **checkpoints**, which are versions of the model created during training.
* **event files**, which contain information that TensorBoard uses to create visualizations.

To specify the top-level directory in which the Estimator stores its information, assign a value to the optional `model_dir` argument of any Estimator's constructor.

You may alter the default checkpointing frequency schedule by passing a `RunConfig` object to the Estimator's `config` argument.

Once checkpoints exist, TensorFlow rebuilds the model each time you call `train()`, `evaluate()`, or `predict()`.

![](https://www.tensorflow.org/images/subsequent_calls.png)

### Feature Columns

#### Feature Columns

Think of **feature columns** as the **intermediaries between raw data and Estimators**. Feature columns are very rich, enabling you to transform a diverse range of raw data into formats that Estimators can use, allowing easy experimentation.

To create feature columns, call **functions from the `tf.feature_column` module**.

![](https://www.tensorflow.org/images/feature_columns/some_constructors.jpg)

**Bucketized column** represents discretized dense input. Buckets include the left boundary, and exclude the right boundary. Namely, boundaries=[0., 1., 2.] generates buckets (-inf, 0.), [0., 1.), [1., 2.), and [2., +inf).

> Note that specifying a three-element boundaries vector creates a four-element bucketized vector. 分桶之后，会直接转换成one-hot形式的。

![](https://www.tensorflow.org/images/feature_columns/bucketized_column.jpg)

```py
# First, convert the raw input to a numeric column.
numeric_feature_column = tf.feature_column.numeric_column("Year")

# Then, bucketize the numeric column on the years 1960, 1980, and 2000.
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000])
```

**Categorical identity columns** can be seen as a special case of bucketized columns. In a categorical identity column, each bucket represents a single, unique integer.

![](https://www.tensorflow.org/images/feature_columns/categorical_column_with_identity.jpg)

```py
# Create categorical output for an integer feature named "my_feature_b",
# The values of my_feature_b must be >= 0 and < num_buckets
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4) # Values [0, 4)
```

**Categorical vocabulary columns** provide a good way to represent strings as a one-hot vector.

![](https://www.tensorflow.org/images/feature_columns/categorical_column_with_vocabulary.jpg)

```py
# Given input "feature_name_from_input_fn" which is a string,
# create a categorical feature by mapping the input to one of
# the elements in the vocabulary list.
vocabulary_feature_column =
    tf.feature_column.categorical_column_with_vocabulary_list(
        key="a feature returned by input_fn()",
        vocabulary_list=["kitchenware", "electronics", "sports"])


# Given input "feature_name_from_input_fn" which is a string,
# create a categorical feature to our model by mapping the input to one of
# the elements in the vocabulary file
# product_class.txt should contain one line for each vocabulary element.
vocabulary_feature_column =
    tf.feature_column.categorical_column_with_vocabulary_file(
        key="a feature returned by input_fn()",
        vocabulary_file="product_class.txt",
        vocabulary_size=3)
```

**Hashed Column**

Often though, _**the number of categories can be so big**_ that it's not possible to have individual categories for each vocabulary word or integer because that would consume too much memory.

![](https://www.tensorflow.org/images/feature_columns/hashed_column.jpg)

**Crossed column**

Combining features into a single feature, better known as feature crosses, enables the model to learn separate weights for each combination of features.

#### Indicator and embedding columns

**Used to wrap any `categorical_column_*` (e.g., to feed to DNN), treats each category as an element in a one-hot vector.** Use `embedding_column` if the inputs are sparse.

![](https://www.tensorflow.org/images/feature_columns/embedding_vs_indicator.jpg)

```py
categorical_column = ... # Create any type of categorical column.

# Represent the categorical column as an indicator column.
indicator_column = tf.feature_column.indicator_column(categorical_column)
```

Actually, the assignments happen during training. That is, **the model learns the best way to map your input numeric categorical values to the embeddings vector value in order to solve your problem**. Embedding columns increase your model's capabilities, since an embeddings vector learns new relationships between categories from the training data.

#### Passing feature columns to Estimators

Not all Estimators permit all types of feature_columns argument(s):

* `LinearClassifier` and `LinearRegressor`: Accept all types of feature column.
* `DNNClassifier` and `DNNRegressor`: Only accept dense columns. Other column types must be wrapped in either an indicator_column or embedding_column.
* `DNNLinearCombinedClassifier` and `DNNLinearCombinedRegressor`:
  * The `linear_feature_columns` argument accepts any feature column type.
  * The `dnn_feature_columns` argument only accepts dense columns.

### Datasets Quick Start

The **`tf.data` module** contains a collection of classes that allows you to easily **load** data, **manipulate** it, and **pipe** it into your model.

#### Basic input

```py
import iris_data
# Fetch the data
train, test = iris_data.load_data()
features, labels = train

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Build the Iterator, and return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
```

#### Reading a CSV File

```py
# Metadata describing the text columns
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth',
                    'label']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
FIELD_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0]]
def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=FIELD_DEFAULTS)
    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))
    # Separate the label from the features
    label = features.pop('label')
    return features, label

def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    # `map` parse each line.
    dataset = tf.data.TextLineDataset(csv_path).skip(1).map(_parse_line)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the dataset.
    return dataset
```

![](https://www.tensorflow.org/images/datasets/map.png)

The `map` method takes a `map_func` argument that describes how each item in the Dataset should be transformed.

```py
train_path, test_path = iris_data.maybe_download()

# All the inputs are numeric
feature_columns = [tf.feature_column.numeric_column(name)
    for name in iris_data.CSV_COLUMN_NAMES[:-1]]

# Build the estimator
est = tf.estimator.LinearClassifier(feature_columns, n_classes=3)
# Train the estimator
batch_size = 100
est.train(
    steps=1000,
    input_fn=lambda : iris_data.csv_input_fn(train_path, batch_size))
```

### Creating Custom Estimators

Pre-made Estimators are **subclasses** of the `tf.estimator.Estimator` base class, while custom Estimators are an **instance** of `tf.estimator.Estimator`

![](https://www.tensorflow.org/images/custom_estimators/estimator_types.png)

A model function (**model_fn**) implements the ML algorithm. The only difference between working with pre-made Estimators.

#### Write an Input function

```py
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
```

#### Create feature columns

```py
# Feature columns describe how to use the input.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

#### Write a model function

```py
# The model function we'll use has the following call signature:
def my_model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labels from input_fn
   mode,     # Optional. Specifies if this training, evaluation or prediction. See ModeKeys.
   params):  # Optional dict of hyperparameters. Will receive what is passed to Estimator in params parameter.

classifier = tf.estimator.Estimator(
    model_fn=my_model,
    params={
        'feature_columns': my_feature_columns,
        # Two hidden layers of 10 nodes each.
        'hidden_units': [10, 10],
        # The model must choose between 3 classes.
        'n_classes': 3,
    })
```

To implement a typical model function, you must do the following:
* Define the model.
* Specify additional calculations for each of the three different modes:
  * Predict
  * Evaluate
  * Train

#### Define the model

```py
    # Use `input_layer` to apply the feature columns.
    net = tf.feature_column.input_layer(features, params['feature_columns'])

    # Build the hidden layers, sized according to the 'hidden_units' param.
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
```

#### Implement prediction, evaluation, training

Focus on that third argument, mode. As the following table shows, when someone calls train, evaluate, or predict, the Estimator framework invokes your model function with the mode parameter set as follows:

| Estimator method | Estimator Mode
|---------| --------------
| train() |	ModeKeys.TRAIN
| evaluate() | ModeKeys.EVAL
| predict() |	ModeKeys.PREDICT

For each mode value, your code must return an instance of `tf.estimator.EstimatorSpec`, which contains the information the caller requires. Let's examine each mode.

**Predict**

```py
# Compute predictions.
predicted_classes = tf.argmax(logits, 1)
if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)
```

**Caculate the loss**

For both training and evaluation we need to calculate the model's loss. This is the objective that will be optimized.

```py
# Compute loss.
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
```

**Evaluate**

The EstimatorSpec returned for evaluation typically contains the following information:
* **loss**, which is the model's loss
* **eval_metric_ops**, which is an optional dictionary of metrics.

```py
# Compute evaluation metrics.
accuracy = tf.metrics.accuracy(labels=labels,
                               predictions=predicted_classes,
                               name='acc_op')
metrics = {'accuracy': accuracy}
tf.summary.scalar('accuracy', accuracy[1])
if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=metrics)
```

**Train**

The EstimatorSpec returned for training must have the following fields set:
* **loss**, which contains the value of the loss function.
* **train_op**, which executes a training step.

```py
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
```

### TensorBoard

```py
# Replace PATH with the actual path passed as model_dir
tensorboard --logdir=PATH
# Then, open TensorBoard by browsing to: http://localhost:6006
```
