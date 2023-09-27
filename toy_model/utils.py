import numpy as np
import matplotlib.pyplot as plt

class SinusoidalConcatenation:
    def __init__(self, sample_rate=1000):
        """Initializes the SinusoidalConcatenation class.
        
        Parameters:
            sample_rate (int): Number of samples per second.
        """
        self.sample_rate = sample_rate

    def generate_signal(self, frequencies, start_periods, durations):
        """Generates a concatenated sinusoidal signal.
        
        Parameters:
            frequencies (list): List of frequencies in Hz.
            start_periods (list): List of starting periods in seconds.
            duration (float): Total duration of the signal in seconds.
            
        Returns:
            np.array: Concatenated sinusoidal signal.
        """
        if len(frequencies) != len(start_periods):
            raise ValueError("Length of frequencies must match length of start_periods.")
        total_duration = np.sum(durations)
        t = np.linspace(0, total_duration, int(self.sample_rate * total_duration), endpoint=False)
        signal = np.zeros_like(t)

        current_time = 0
        for freq, start,duration in zip(frequencies, start_periods,durations):
            if current_time >= total_duration:
                break
            
            next_time = current_time + duration
            t_segment = t[(t >= current_time) & (t < next_time)]
            
            # Generate the sinusoidal signal for this segment
            signal_segment = np.sin(2*np.pi*freq*(t_segment - current_time) + 2*np.pi*start)
            
            # Update the overall signal
            signal[(t >= current_time) & (t < next_time)] = signal_segment

            current_time = next_time

        return signal

# Example usage
if __name__ == "__main__":
    gen = SinusoidalConcatenation(sample_rate=5000)
    frequencies = [1, 5, 10]  # frequencies in Hz
    start_periods = [0.5, 0.5, 0.5]  # starting periods in seconds
    duration = 1.5  # total duration in seconds

    signal = gen.generate_signal(frequencies, start_periods, duration)

    # Plotting the generated signal
    plt.figure()
    plt.title("Concatenated Sinusoidal Signal")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.plot(np.linspace(0, duration, int(gen.sample_rate * duration), endpoint=False), signal)
    plt.show()