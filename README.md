# TokenSim

TokenSim is a tool for simulating the behavior of large language models (LLMs) in a distributed environment. It provides a flexible framework for modeling and analyzing the performance of LLMs under various conditions.

[TokenSim: Enabling Hardware and Software Exploration for Large Language Model Inference Systems](https://arxiv.org/abs/2503.08415)

## Key Features

- Dynamic Workload Simulation: TokenSim supports dynamic LLM request inputs sampled from real-world datasets, allowing for realistic simulations of concurrent requests and varying request lengths.
- Customizable Scheduling and Memory Management: Users can define their own scheduling policies and memory management strategies at the operator level, enabling fine-grained control over system optimizations.
- Extensive Hardware Support: TokenSim supports a wide range of hardware configurations, including CPUs, GPUs, and FPGAs, and allows for the simulation of different compute simulators like GenZ and LLMCompass.
- Accurate Performance Modeling: With support for detailed memory simulation and operator-level hooks, TokenSim achieves high accuracy in modeling the performance of LLM inference systems.
- Scalable and Modular Design: The framework is built using the SimPy discrete-event simulation library, ensuring efficient and scalable simulations that can run on personal computers without requiring specialized hardware.

## Installation

```shell
$ git clone https://github.com/pku-lemonade/TokenSim.git
$ git submodule update --init --recursive
$ conda create -n tokensim python=3.11
$ conda activate tokensim
$ pip install -r requirements.txt
```

## Quick Start

You can run the benchmark script with the following command:

```shell
$ ./scripts/benchmark.sh
```

## Configuration Parameters

The benchmark script supports the following parameters:

- `--qps`: The number of requests per second
- `--batching`: The batching strategy to use (greedy, dynamic, etc.)
- `--distribution`: The distribution of request lengths
- `--block_size`: The size of the batch to use
- `--swap_policy`: The swap policy to use
- `--model`: The LLM model to simulate
- `--hardware`: The hardware configuration path
- `--duration`: Simulation duration in seconds, if not specified, the simulation will run until all requests are processed

## Note

TransformerRoofline is not open-sourced in this repository. Instead, we provide pre-compiled shared libraries (`.so` files) for Linux systems. These libraries are essential for accurate performance modeling of transformer-based models. The shared libraries are located in the `lib/` directory and will be automatically loaded when running the simulator.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TokenSim in your research, please cite our paper:

```bibtex
@misc{wu2025tokensimenablinghardwaresoftware,
  title={TokenSim: Enabling Hardware and Software Exploration for Large Language Model Inference Systems},
  author={Feiyang Wu and Zhuohang Bian and Guoyang Duan and Tianle Xu and Junchi Wu and Teng Ma and Yongqiang Yao and Ruihao Gong and Youwei Zhuo},
  year={2025},
  eprint={2503.08415},
  archivePrefix={arXiv},
  primaryClass={cs.DC},
  url={https://arxiv.org/abs/2503.08415},
}
```

## Acknowledgments

- Thanks to all contributors who have helped shape TokenSim
- Special thanks to the SimPy community for their excellent discrete-event simulation library
