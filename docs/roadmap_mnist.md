# Roadmap for moving MNIST task

The [moving MNIST task](http://www.cs.toronto.edu/~nitish/unsupervised_video/) is a sequence prediction task with 2 MNIST digits moving and bouncing around the walls of 64 x 64 frame. It is one of the standard benchmarks for unsupervised learning. Given the high level of controllability for this task and that we can easily [generate a customized dataset for training and testing](https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe), we hope to use it as a first step to understand the models we wish to use to investigate our conjecture (details [here](./The%20role%20of%20serotonin.pdf)).

## Experimental manipulations

- [/] Generate custom datasets
  - [x] Get a working script
  - [ ] Refactor for easily generating new tasks
- [ ] Generate datasets:
  - [/] v0.1
    - [x] Position glitch
    - [/] Digit identity glitch
    - [ ] Bounce
  - [ ] v0.2
    - [ ] Speed glitch

### Version 0 implementation

#### v0.1

- [x] Position glitch
  - [x] Start from left-bottom/-top and move to the opposite corner
  - [x] Switch video at frame 10
- [ ] Digit identity glitch
  - [ ] Use position glitch data but swap out second half of videos instead of first half
- [ ] Bounce
  - [ ] Start from left-middle and move diagonally ($\pm 45^{\circ}$) with same speed to bounce digits at top-/bottom-middle

##### Lessons

Note here

#### v0.2

- [ ] Speed glitch
  - [ ] Increase or decrease the speed midway through the video

## Analysis

To be organized at some point in the future. For now, we are only listing the ideas.

### Quantities of interest

We need to figure out which quantities are of interest for the plots. Currently, we're plotting the latent prior variance, but we need to get the equation for this and make sure that it is a reasonable quantity to plot. We should also check other quantities of interest that come close to the prediction error or some measure of the uncertainty/surprise in the latent variables.

### Organization of the latent space

We also need to investigate how the latent space is organized. Hopefully, there is a clean separation of the position and the digit identity encoding in the latent dimensions. If not, one idea could be to try increasing the $\beta$ for the $\beta$-VAE, which is known to disentangle representations in the latent space (cite). But regardless, the point would be to show that multiple _features_ of the state prediction error may evoke a signal, but if we only measure the average activity across serotonergic neurons in the dorsal raphe nucleus (DRN), as for instance is done with fiber photometry, then we expect the strongest signal (prediction error) when all the features of the state are disrupted. Given that we as the experimenters know which features have been manipulated, we can regress the activity of the neurons that are responsive for that signal to show that different neurons indicate the prediction error of specific latent features.
