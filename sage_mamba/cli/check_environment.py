"""Print installed versions, device availability, and expected/actual model sizes."""

from importlib import metadata
from pathlib import Path
import sys


def main():
    print('Python', sys.version.split()[0])
    for line in (Path(__file__).resolve().parents[2] / 'requirements.txt').read_text().splitlines():
        name, expected = line.split('==')
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = 'NOT INSTALLED'
        match = actual.split('+')[0] == expected
        print(f'{name}=={actual}  expected={expected}  {"OK" if match else "DIFF"}')
    try:
        import torch
        from sage_mamba.model import ModelConfig, SAGENet, expected_parameter_count, parameter_count
        print('CUDA available:', torch.cuda.is_available())
        print('PyTorch CUDA:', torch.version.cuda)
        if torch.cuda.is_available():
            print('Device:', torch.cuda.get_device_name(0))
        for dataset in ['seed', 'dreamer']:
            for backbone in ['lstm', 'gru', 'mamba']:
                cfg = ModelConfig.for_dataset(dataset, backbone)
                expected = expected_parameter_count(cfg)
                try:
                    model = SAGENet(cfg)
                    actual = parameter_count(model)
                    print(f'{dataset} {backbone}: actual={actual:,}; expected={expected:,}')
                    if actual != expected:
                        raise RuntimeError('Model parameter count mismatch.')
                    del model
                except ImportError:
                    print(f'{dataset} {backbone}: expected={expected:,}; Mamba not installed')
    except ImportError as exc:
        print('Model inspection unavailable:', exc)


if __name__ == '__main__':
    main()
