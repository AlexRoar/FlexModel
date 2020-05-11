# FlexModel
Flexible deep learning model with open source

## Why I created it?

In this work, I implemented Deep Neural Network architecture by myself. So, this is partly educational project as you can see how this all works.

## Basic usage
You can build model by layer.
The most simple model:

`
from flexmodel import FlexModel, Dense

model = FlexModel()

model.add(Dense(n_neurons=1, activation='sigmoid'))

`