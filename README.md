<h1 align="center"> Teeny Go </h1>

<h4 align="center"> 9x9 go agent inspired by Alpha Go </h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-v3.6+-blue.svg">
  <img src="https://img.shields.io/badge/Pytorch-v1.3-orange.svg">
  <img src="https://img.shields.io/badge/Build-Passing-green.svg">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg">
</p>

<p align="center">
  <a href="#About">About</a> •
  <a href="#Network-Architecture">Network Architecture</a> •
  <a href="#Training">Training</a> •
  <a href="#Results">Results</a> •
  <a href="#Sources">Sources</a>
</p>

## About
This project aims at creating a 9x9 go agent using the methods implemented by the Google Deepmind team in both AlphaGo and AlphaGo Zero. This begins by training a neural network using supervised learning for board evaluation and move prediction and then moving into a self improvement stage using reinforcement learning. The next step is to train a policy and value network tabula rasa using pure reinforcement learning and self-play.

<p align="center">
  <img src="https://github.com/Gregory-Eales/Teeny-Go/blob/master/utils/assets/images/go_sample.png" width="200"/>
</p>

## Network Architecture

This model utilizes a convolutional neural network with residual blocks and either a policy or value head to make predictions about a given board state. The policy head utilizes a 1x1 convolution, batch norm, and a fully connected layer and the value head...

## Training

### Supervised Learning

#### Value Training

<p align="center">
  <img src="https://github.com/Gregory-Eales/Teeny-Go/blob/master/utils/assets/Val_Net_Model_Comparison.png" height="550"/>
</p>

These value network models were trained on a 2000 game subset of the 40,000 9x9 go games collected from the OGS website. Each game has at least one dan level player ensuring some degree of optimal play. The games were processed from the standard game format to a (n, 11, 9, 9) tensor, where n represents the number of states in the game, 11 where 2 sets of 5 dimensions are allocated for white and blacks stone positions for the past 5 moves and the final dimension for the player turn at that state. Each model was trained for 50 iterations with seeded random shuffling of data to predict the winner of the game. This was done using the a tanh activation function where 1 represents the current player winning and -1 represents the opposing player winning. The decision boundry for a successful prediction is one where |p| > 1/3, anything less than 1/3 it taken to be an uncertainty interval. All models tested converge to an accuracy of about 55%-60% which is relativly good considering the ambiguity of early game states. Although the accuracy remained relativly consistent for the majority of training, using model versions where the validation and training loss were at their lowest proved make the most reasonable predictions in user tesing.

<p align="center">
  <img src="https://github.com/Gregory-Eales/Teeny-Go/blob/master/utils/assets/Val_Net_Training.png" height="400"/>
</p>

#### Policy Training

<p align="center">
  <img src="https://github.com/Gregory-Eales/Teeny-Go/blob/master/utils/assets/Pol_Net_Model_Comparison.png" height="550" />
</p>


<p align="center">
  <img src="https://github.com/Gregory-Eales/Teeny-Go/blob/master/utils/assets/Pol_Net_Training.png" height="400"/>
</p>

### Reinforcement Learning
- in progress

## Results

### Supervised Learning
- in progress

### Reinforcement Learning
- in progress

## Sources

### Go Datasets

- Aya and Natsukaze's selfplay games (http://www.yss-aya.com/ayaself/ayaself.html#nats2018)
- CGOS Archives for Board Size 9x9 (http://www.yss-aya.com/cgos/9x9/archive.html)
- Mini-Go 9x9 sgf (https://console.cloud.google.com/storage/browser/minigo-pub/v3-9x9/sgf/)
- Professional + Mini-go 9x9 (https://homepages.cwi.nl/~aeb/go/games/index.html)

### Go Engines

- Michi Go Engine (https://github.com/pasky/michi)
- Pachi Go Engine (https://pachi.or.cz/)
- Disco Python Go Engine (http://shed-skin.blogspot.com/2009/08/disco-elegant-python-go-player-update.html)
- Simple Go (http://londerings.cvs.sourceforge.net/londerings/go/simple_go/)

### Papers

- Mastering the game of Go with deep neural networks and tree search (https://www.nature.com/articles/nature16961)

- Mastering the game of Go without human knowledge   
(https://www.nature.com/articles/nature24270)

- ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero   
(https://arxiv.org/pdf/1902.04522.pdf)

### Libraries

- OpenSpiel (https://github.com/deepmind/open_spiel)

- PyTorch (https://pytorch.org/)

### Additional Sources

- A Simple Alpha(Go) Zero Tutorial (https://web.stanford.edu/~surag/posts/alphazero.html)
- EGF Elo Rating System (https://senseis.xmp.net/?EGFRatingSystem)


## Meta

Gregory Eales – [@GregoryHamE](https://twitter.com/GregoryHamE) – gregory.hamilton.e@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/Gregory-Eales](https://github.com/Gregory-Eales)
