### Code design for Moving MNIST data generation

[The publicly available code for generating moving MNIST data](https://gist.github.com/tencia/afb129122a64bde3bd0c) is a great starting point, but it has very high coupling, which makes it tedious for us to generate all the cases we want. We would like to refactor it to separate the different aspects of data generation that we would like to control. To do this, we will follow the [factory design pattern](https://www.youtube.com/watch?v=s_4ZrtQs8Do) with the following sets of classes:

1. `DataConfig`
	- dataclass with key data configuration attributes:
	- `num_frames`, `nums_per_frame`, `num_videos`, `shape_canvas`, `size_digit`
2. `TrajectoryGenerator`
	- generates trajectories of digit positions
3. `MNISTSampler`
	- samples indices from the MNIST dataset to go with the trajectories
4. `MovingMNISTFactory`
	- makes the full video dataset given 1-3