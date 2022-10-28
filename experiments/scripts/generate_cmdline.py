import numpy as np

RNG = np.random.default_rng()


class ParameterWriter:

    def __init__(self, name):
        self.name = name
        self.val = None

    def get_val(self):
        return self.val

    def __call__(self):
        self.val = self.gen()
        return f"--{self.name}={self.val}"


class CopycatWriter(ParameterWriter):

    def __init__(self, name, master):
        super().__init__(name)

        self.master = master

    def gen(self):
        return self.master.get_val()


class UniformIntegerWriter(ParameterWriter):

    def __init__(self, name, vals):
        super().__init__(name)
        self.vals = vals

    def gen(self):
        index = RNG.integers(len(self.vals))
        return self.vals[index]


class ExponentialUniformWriter(ParameterWriter):

    def __init__(self, name, low, high):
        super().__init__(name)
        self.low = np.log10(low)
        self.high = np.log10(high)

    def gen(self):
        log_val = RNG.uniform(self.low, self.high)
        return np.power(10, log_val)


def main():
    encoder_size = UniformIntegerWriter("encoder_size", (1, 2, 3, 4, 5, 8, 10))
    decoder_size = CopycatWriter("decoder_size", encoder_size)

    writers = (
        ExponentialUniformWriter("lr", 1e-3, 5e-2),
        UniformIntegerWriter("control_rnn_size", (8, 16, 24, 32, 64, 128)),
        encoder_size,
        decoder_size,
    )

    print(" ".join(w() for w in writers))


if __name__ == "__main__":
    main()
