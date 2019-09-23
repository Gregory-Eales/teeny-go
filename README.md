# Teeny-Go

9x9 go agent inspired by Alpha Go

<p align="center">
  <img src="https://github.com/Gregory-Eales/Teeny-Go/blob/master/utils/assets/images/go_sample.png" width="500"/>
</p>

## About
This project aims at creating a 9x9 go agent using the methods implemented by the Google Deepmind team in both AlphaGo and AlphaGo Zero. This begins by training a neural network using supervised learning for board evaluation and then moving into a self improvement stage using reinforcement learning. The next step is to train a policy and value network tabula rasa using pure reinforcement learning and self-play.


## To Do
 - [x] create network 
 - [ ] create elo anchor agents
 - [ ] create dataset
 - [ ] train network architectures using supervised learning (sl)
 - [ ] test networks for optimal architecture using elo
 - [ ] train best network using sl for highest elo rating
 - [ ] add monte carlo tree search (mcts)
 - [ ] test sl network + mcts combination
 - [ ] train sl network with reinforcement learning (rl) and mcts
 - [ ] test new rl network using elo
 - [ ] train new network architectures using rl only
 - [ ] test new rl networks for optimal architecture using elo
 - [ ] train best rl only network
 - [ ] test rl only netowork using elo

## Network Architecture

<p align="center">
  <img src="https://github.com/Gregory-Eales/Teeny-Go/blob/master/utils/assets/AlphaGo-CheatSheet.png" width="500"/>
</p>

<p align="center">
  <img src="https://github.com/Gregory-Eales/Teeny-Go/blob/master/utils/assets/Network_Architecture_Diagram.png" width="500"/>
</p>



## Training

### Supervised Learning
- in progress

### Self Play
- in progress

## Results

### Supervised Learning
- in progress

### Self Play
- in progress

## Sources

### Go Datasets

- Aya and Natsukaze's selfplay games (http://www.yss-aya.com/ayaself/ayaself.html#nats2018)
- CGOS Archives for Board Size 9x9 (http://www.yss-aya.com/cgos/9x9/archive.html)
- Mini-Go 9x9 sfg (https://console.cloud.google.com/storage/browser/minigo-pub/v3-9x9/sgf/)

### Go Engines

- Michi Go Engine (https://github.com/pasky/michi)
- Pachi Go Engine (https://pachi.or.cz/)
- Disco Python Go Engine (http://shed-skin.blogspot.com/2009/08/disco-elegant-python-go-player-update.html)
- Simple Go (http://londerings.cvs.sourceforge.net/londerings/go/simple_go/)

### Papers

- Mastering the game of Go with deep neural networks and tree search (https://www.nature.com/articles/nature16961)

- Mastering the game of Go without human knowledge 
(https://www.nature.com/articles/nature24270)

### Libraries

- OpenSpiel (https://github.com/deepmind/open_spiel)

- PyTorch (https://pytorch.org/)

## Meta

Gregory Eales – [@GregoryHamE](https://twitter.com/GregoryHamE) – gregory.hamilton.e@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/Gregory-Eales](https://github.com/Gregory-Eales)

