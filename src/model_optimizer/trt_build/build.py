import argparse
import logging
import os
import time

import tensorrt as trt

from termcolor import colored
# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

print_color = "green"


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "bf16",
    workspace_mb: int = 8192,
    min_shapes: dict = None,
    opt_shapes: dict = None,
    max_shapes: dict = None,
):
    """
    Build TensorRT engine from ONNX model.

    Args:
        onnx_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', 'bf16', 'fp8')
        workspace_mb: Workspace size in MB
        min_shapes: Minimum input shapes (dict: name -> shape tuple)
        opt_shapes: Optimal input shapes (dict: name -> shape tuple)
        max_shapes: Maximum input shapes (dict: name -> shape tuple)
    """
    logger.info("=" * 80)
    logger.info("TensorRT Engine Builder")
    logger.info("=" * 80)
    logger.info(f"ONNX model: {onnx_path}")
    logger.info(f"Engine output: {engine_path}")
    logger.info(f"Precision: {precision.upper()}")
    logger.info(f"Workspace: {workspace_mb} MB")
    logger.info("=" * 80)

    print(colored("=" * 80, print_color))
    print(colored("TensorRT Engine Builder", print_color))
    print(colored("=" * 80), print_color)
    print(colored(f"ONNX model: {onnx_path}", print_color))
    print(colored(f"Engine output: {engine_path}", print_color))
    print(colored(f"Precision: {precision.upper()}", print_color))
    print(colored(f"Workspace: {workspace_mb} MB", print_color))
    print(colored("=" * 80, print_color))

    # Create TensorRT logger
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    # Create builder and network
    logger.info("\n[Step 1/5] Creating TensorRT builder...")
    print(colored("\n[Step 1/5] Creating TensorRT builder...", print_color))
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    logger.info("\n[Step 2/5] Parsing ONNX model...")
    print(colored("\n[Step 2/5] Parsing ONNX model...", print_color))
    if not parser.parse_from_file(onnx_path):
        logger.error("Failed to parse ONNX file")
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        raise RuntimeError("ONNX parsing failed")

    # Parser successful. Network is loaded
    logger.info(f"Network inputs: {network.num_inputs}")
    print(colored(f"Network inputs: {network.num_inputs}", print_color))
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        logger.info(f"  Input {i}: {inp.name} {inp.shape}")
        print(colored(f"  Input {i}: {inp.name} {inp.shape}", print_color))

    logger.info(f"Network outputs: {network.num_outputs}")
    print(colored(f"Network outputs: {network.num_outputs}", print_color))
    for i in range(network.num_outputs):
        out = network.get_output(i)
        logger.info(f"  Output {i}: {out.name} {out.shape}")
        print(colored(f"  Output {i}: {out.name} {out.shape}", print_color))

    # Create builder config
    logger.info("\n[Step 3/5] Configuring builder...")
    print(colored("\n[Step 3/5] Configuring builder...", print_color))
    config = builder.create_builder_config()

    # Enable detailed profiling for engine inspection
    # This allows get_layer_information() to return layer types, precisions, tactics, etc.
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    logger.info("Enabled DETAILED profiling verbosity for engine inspection")
    print(colored("Enabled DETAILED profiling verbosity for engine inspection", print_color))

    # Set workspace
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1024**2))

    # Set precision
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("Enabled FP16 mode")
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
        logger.info("Enabled BF16 mode")
    elif precision == "fp8":
        config.set_flag(trt.BuilderFlag.FP8)
        logger.info("Enabled FP8 mode")
    elif precision == "fp32":
        logger.info("Using FP32 (default precision)")
    else:
        raise ValueError(f"Unknown precision: {precision}")

    # Set optimization profiles for dynamic shapes
    if min_shapes and opt_shapes and max_shapes:
        logger.info("\n[Step 4/5] Setting optimization profiles...")
        print(
            colored("\n[Step 4/5] Setting optimization profiles...", print_color))
        profile = builder.create_optimization_profile()

        for i in range(network.num_inputs):
            inp = network.get_input(i)
            input_name = inp.name

            if input_name in min_shapes:
                min_shape = min_shapes[input_name]
                opt_shape = opt_shapes[input_name]
                max_shape = max_shapes[input_name]

                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                logger.info(f"  {input_name}:")
                logger.info(f"    min: {min_shape}")
                logger.info(f"    opt: {opt_shape}")
                logger.info(f"    max: {max_shape}")

        config.add_optimization_profile(profile)
    else:
        raise RuntimeError("Provide min/max and opt shapes for dynamic axes")

    # Build engine
    logger.info("\n[Step 5/5] Building TensorRT engine...")
    print(colored("\n[Step 5/5] Building TensorRT engine...", print_color))

    start_time = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - start_time

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    logger.info(
        f"Engine built in {build_time:.1f} seconds ({build_time / 60:.1f} minutes)")

    # Save engine
    logger.info(f"\nSaving engine to {engine_path}...")
    print(colored(f"\nSaving engine to {engine_path}...", print_color))
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    engine_size_mb = os.path.getsize(engine_path) / (1024**2)
    logger.info(f"Engine saved! Size: {engine_size_mb:.2f} MB")
    print(colored(f"Engine saved! Size: {engine_size_mb:.2f} MB", print_color))

    logger.info("\n" + "=" * 80)
    print(colored("\n" + "=" * 80, print_color))
    logger.info("ENGINE BUILD COMPLETE!")
    print(colored("ENGINE BUILD COMPLETE!", print_color))
    logger.info("=" * 80)
    print(colored("=" * 80, print_color))
    logger.info(f"Engine file: {engine_path}")
    print(colored(f"Engine file: {engine_path}", print_color))
    logger.info(f"Size: {engine_size_mb:.2f} MB")
    print(colored(f"Size: {engine_size_mb:.2f} MB", print_color))
    logger.info(f"Build time: {build_time:.1f}s")
    print(colored(f"Build time: {build_time:.1f}s", print_color))
    logger.info(f"Precision: {precision.upper()}")
    print(colored(f"Precision: {precision.upper()}", print_color))
    logger.info("=" * 80)
