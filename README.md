# FlexModel
Flexible deep learning model with open source

## Why I created it?

In this work, I implemented Deep Neural Network architecture by myself. So, this is partly educational project as you can see how this all works.

## Basic usage
You can build model by layer.
The most simple model:

```python
from flexmodel import FlexModel, Dense
model = FlexModel()
model.add(Dense(n_neurons=1, activation='sigmoid'))

progress = model.fit(X, y, learning_rate=0.1)

predictions = model.predict(test_X)
```

Multilayer model with more parameters:
```python
model = FlexModel()

model.add(Dense(n_neurons=50, activation='lerelu', dropout=0.6))
model.add(Dense(n_neurons=70, activation='lerelu', dropout=0.6))
model.add(Dense(n_neurons=20, activation='lerelu'))
model.add(Dense(n_neurons=7, activation='lerelu'))
model.add(Dense(n_neurons=1, activation='sigmoid'))

progress, progressEval = model.fit(
                     X,
                     y,
                     optimization='adam', # available optimization: ADAM, Momentum, RMSProp
                     learning_rate=0.0045,
                     lambd=0.3, # Lambda for L2 weights norm
                     printLoss=True, # Prints loss into console
                     batches_size=64, # Mini-batches size
                     decay_rate=0.0015, # Learning rate decay
                     decay_type='hyperbolic', # Learning rate decay type: exponential, hyperbolic, squared
                     n_iter=1000, # Number of iterations
                     eval_set=(dev_X, dev_y)) # Set for evaluation while training
```
