from dataclasses import dataclass

from . import digit_sampler, trajectory_generator
from .config import VideoConfig, load_dataset
from .moving_mnist_factory import MovingMNISTFactory

CONFIG = VideoConfig(num_frames=20, num_images=100, nums_per_image=2)
MNIST_DATA = load_dataset()


@dataclass
class StandardDataGenerator:
    traj_generator = trajectory_generator.Standard(config=CONFIG)
    mnist_sampler = digit_sampler.Standard(config=CONFIG)
    save_name = "standard"


@dataclass
class PositionGlitchDataGenerator:
    traj_generator = trajectory_generator.PositionGlitch(config=CONFIG)
    mnist_sampler = digit_sampler.Standard(config=CONFIG)
    save_name = "random_position_glitch"


@dataclass
class OffsetPositionGlitchDataGenerator:
    traj_generator = trajectory_generator.OffsetPositionGlitch(config=CONFIG)
    mnist_sampler = digit_sampler.Standard(config=CONFIG)
    save_name = "offset_position_glitch"


@dataclass
class ReverseDirectionGlitchDataGenerator:
    traj_generator = trajectory_generator.ReverseDirectionGlitch(config=CONFIG)
    mnist_sampler = digit_sampler.Standard(config=CONFIG)
    save_name = "reverse_direction_glitch"


@dataclass
class RandomDirectionGlitchDataGenerator:
    traj_generator = trajectory_generator.RandomDirectionGlitch(config=CONFIG)
    mnist_sampler = digit_sampler.Standard(config=CONFIG)
    save_name = "random_direction_glitch"


@dataclass
class SpeedGlitchDataGenerator:
    traj_generator = trajectory_generator.SpeedGlitch(config=CONFIG)
    mnist_sampler = digit_sampler.Standard(config=CONFIG)
    save_name = "speed_glitch"


@dataclass
class TimedBounceDataGenerator:
    traj_generator = trajectory_generator.TimedBounce(config=CONFIG)
    mnist_sampler = digit_sampler.Standard(config=CONFIG)
    save_name = "timed_bounce"


@dataclass
class DigitGlitchDataGenerator:
    traj_generator = trajectory_generator.Standard(config=CONFIG)
    mnist_sampler = digit_sampler.DigitGlitch(config=CONFIG)
    save_name = "digit_glitch"


@dataclass
class PositionAndDigitGlitchDataGenerator:
    traj_generator = trajectory_generator.PositionGlitch(config=CONFIG)
    mnist_sampler = digit_sampler.DigitGlitch(config=CONFIG)
    save_name = "position_and_digit_glitch"


def generate_data(data_gen, path="../../../data/MNIST/") -> None:
    """Generate data with a given configuration and save it to a given path"""
    factory = MovingMNISTFactory(
        data_gen.traj_generator, data_gen.mnist_sampler, MNIST_DATA, CONFIG
    )
    print(f"Generating data for {data_gen.save_name}...")
    data = factory.make()
    factory.save(path + data_gen.save_name, data)
    print(f"Data saved to {path+data_gen.save_name}!\n")


def main(path="../../../data/MNIST/") -> None:
    generate_data(StandardDataGenerator(), path)
    generate_data(PositionGlitchDataGenerator(), path)
    generate_data(OffsetPositionGlitchDataGenerator(), path)
    generate_data(ReverseDirectionGlitchDataGenerator(), path)
    generate_data(RandomDirectionGlitchDataGenerator(), path)
    generate_data(SpeedGlitchDataGenerator(), path)
    generate_data(TimedBounceDataGenerator(), path)
    generate_data(DigitGlitchDataGenerator(), path)
    generate_data(PositionAndDigitGlitchDataGenerator(), path)


if __name__ == "__main__":
    main()
