import time

import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner

import sys


def main():
    # Just as we did when building, we can compose multiple loaders together
    # to achieve the behavior we want. Specifically, we want to load a serialized
    # engine from a file, then deserialize it into a TensorRT engine.
    engine_path = sys.argv[1]
    input_npy = sys.argv[2]
    load_engine = EngineFromBytes(BytesFromPath(engine_path))

    # Inference remains virtually exactly the same as before:
    with TrtRunner(load_engine) as runner:
        #        input_data = np.random.randn(1, 3, 640, 640).astype(np.float32)
        #        np.save('yolo_intput.npy', input_data)
        input_data = np.load(input_npy)
        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        for i in range(10):
            start = time.time()
            outputs = runner.infer(feed_dict={"images": input_data})
            end = time.time()
            cost = (end - start) * 1000
            print(f'cost {cost} ms')

        print("Inference succeeded!")
        for k, v in outputs.items():
            # np.save(f'{k}.npy', v)
            expect_output = np.load(f'{k}.npy')
#            print(expect_output[0][0])
#            print('------')
#            print('------')
#            print(v[0][0])
            print('ASSERT_EQUAL', np.array_equal(expect_output, v))


if __name__ == "__main__":
    main()
