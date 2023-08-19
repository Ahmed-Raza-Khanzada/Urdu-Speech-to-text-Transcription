class Preporcess_Data()
    def __init__(self) -> None:
        # An integer scalar Tensor. The window length in samples.
frame_length = 256#600#256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160#307#160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384#650#384
