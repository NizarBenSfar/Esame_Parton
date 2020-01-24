{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from keras.layers import Input, Conv2D, Dense, MaxPooling2D , Flatten\n",
    "from keras.models import Model, Sequential, load_model\n",
    "from collections import deque\n",
    "import keras"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [],
   "source": [
    "class DQN:\n",
    "    def __init__(self, input_shape, output_shape, discount=0.99, update_target_every=10, memory_size=2000):\n",
    "        self.input_shape=input_shape\n",
    "        self.output_shape=output_shape\n",
    "        self.discount=discount\n",
    "        self.update_target_every=update_target_every\n",
    "        self.policy_net=self.create_model()\n",
    "        self.memory=deque(maxlen=memory_size)\n",
    "        self.target_counter=0 \n",
    "    \n",
    "    def create_model(self):\n",
    "        model=Sequential()\n",
    "        model.add(Conv2D(input_shape=self.input_shape, filters=128, kernel_size=(7,7), strides=(1,1), padding=\"valid\", \n",
    "                        activation=\"relu\", use_bias=True,))\n",
    "        model.add(MaxPooling2D(pool_size=(3,3), padding=\"valid\"))\n",
    "        model.add(Conv2D(filters=128, kernel_size=(7,7), strides=(2,2), padding=\"valid\", \n",
    "                        activation=\"relu\", use_bias=True,))\n",
    "        model.add(MaxPooling2D(pool_size=(2,2), padding=\"valid\"))\n",
    "        model.add(Conv2D(filters=128, kernel_size=(7,7), strides=(2,2), padding=\"valid\", \n",
    "                        activation=\"relu\", use_bias=True,))\n",
    "        model.add(Flatten())\n",
    "        model.add(Dense(512, activation=\"relu\"))\n",
    "        model.add(Dense(self.output_shape, activation=\"softmax\"))\n",
    "        adm=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)\n",
    "        model.compile(loss=\"mse\", optimizer=adm, metrics=[\"accuracy\"])\n",
    "        return model\n",
    "    \n",
    "    def get_action (self,state):\n",
    "        action_prob= self.policy_net.predict(state)\n",
    "        return \n",
    "        \n",
    "       \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "dqn=DQN([210,160,3] , 6)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "rete = dqn.policy_net"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "conv2d_35 (Conv2D)           (None, 204, 154, 128)     18944     \n",
      "_________________________________________________________________\n",
      "max_pooling2d_23 (MaxPooling (None, 68, 51, 128)       0         \n",
      "_________________________________________________________________\n",
      "conv2d_36 (Conv2D)           (None, 31, 23, 128)       802944    \n",
      "_________________________________________________________________\n",
      "max_pooling2d_24 (MaxPooling (None, 15, 11, 128)       0         \n",
      "_________________________________________________________________\n",
      "conv2d_37 (Conv2D)           (None, 5, 3, 128)         802944    \n",
      "_________________________________________________________________\n",
      "flatten_11 (Flatten)         (None, 1920)              0         \n",
      "_________________________________________________________________\n",
      "dense_21 (Dense)             (None, 512)               983552    \n",
      "_________________________________________________________________\n",
      "dense_22 (Dense)             (None, 6)                 3078      \n",
      "=================================================================\n",
      "Total params: 2,611,462\n",
      "Trainable params: 2,611,462\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "rete.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
