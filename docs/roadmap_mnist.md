# Roadmap for moving MNIST task

The [moving MNIST task](http://www.cs.toronto.edu/~nitish/unsupervised_video/) is a sequence prediction task with 2 MNIST digits moving and bouncing around the walls of 64 x 64 frame. It is one of the standard benchmarks for unsupervised learning. Given the high level of controllability for this task and that we can easily [generate a customized dataset for training and testing](https://gist.github.com/praateekmahajan/b42ef0d295f528c986e2b3a0b31ec1fe), we hope to use it as a first step to understand the models we wish to use to investigate our conjecture (details [here](./The%20role%20of%20serotonin.pdf)).

## Experimental manipulations

We would like to cover the following experimental manipulations:

- Position glitch
- Digit identity glitch
- Bounce
- Speed glitch

### Implementation

Here is the reasoning behind the [code design for Moving MNIST data generation](./code_design_moving_mnist.md).
#### v0.1 Try whatever works

- [x] Position glitch
	- [x] Start from left-bottom/-top and move to the opposite corner
	- [x] Switch video at frame 10
- [x] Digit identity glitch
	- [x] Use position glitch data but swap out second half of videos instead of first half
- [-] Bounce
	- [-] Start from left-middle and move diagonally ($\pm 45^{\circ}$) with same speed to bounce digits at top-/bottom-middle
- [-] Speed glitch
	- [-] Increase or decrease the speed midway through the video

> [!NOTE] Abandoned in favor of refactoring
> The code was getting too tedious to use for each specific use case. Instead, we'll focus on refactoring the data generation code entirely to easily generate arbitrary new trajectories.

#### v0.2: Refactor data generation code

The goal is the design modular code generating trajectories. For v0.2, the focus is on the architecture, so just get classes functioning for the standard trajectories. Here's the [[code design for Moving MNIST data generation]].

- [x] helper functions:
	- [x] add type hints
	- [x] update names to please linter
- [x] config class for data
- [x] trajectory generator classes
- [x] MNIST Sampler classes
- [x] Moving MNIST Factory

#### v0.3: Add classes for required manipulations

- [ ] Trajectory generator
	- [/] Position glitch
	- [ ] Speed glitch
	- [ ] Bounce
- [ ] MNIST sampler
	- [ ] Digit identity glitch
- [ ] Moving MNIST factories
	- [ ] Position glitch
	- [ ] Speed glitch
	- [ ] Digit identity glitch
	- [ ] Bounce

---

## Analysis

To be organized at some point in the future. For now, we are only listing the ideas.

### Quantities of interest

We need to figure out which quantities are of interest for the plots. Currently, we're plotting the latent prior variance, but we need to get the equation for this and make sure that it is a reasonable quantity to plot. We should also check other quantities of interest that come close to the prediction error or some measure of the uncertainty/surprise in the latent variables.

### Organization of the latent space

We also need to investigate how the latent space is organized. Hopefully, there is a clean separation of the position and the digit identity encoding in the latent dimensions. If not, one idea could be to try increasing the $\beta$ for the $\beta$-VAE, which is known to disentangle representations in the latent space (cite). But regardless, the point would be to show that multiple *features* of the state prediction error may evoke a signal, but if we only measure the average activity across serotonergic neurons in the dorsal raphe nucleus (DRN), as for instance is done with fiber photometry, then we expect the strongest signal (prediction error) when all the features of the state are disrupted. Given that we as the experimenters know which features have been manipulated, we can regress the activity of the neurons that are responsive for that signal to show that different neurons indicate the prediction error of specific latent features.