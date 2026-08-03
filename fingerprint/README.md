# How to Extend New Fingerprinting Methods

This directory is for implementing and extending various model fingerprinting methods.

## 1. Implement a New Fingerprinting Method
- Create a new class following `fingerprint/fingerprint_interface.py`, and implement the three required methods (such as `fit`, `extract`, `compare`, etc.—refer to the interface for details).
- Ensure your implementation is compatible with the interface.

### Example
```python
from fingerprint.fingerprint_interface import FingerprintInterface

class MyNewFingerprint(FingerprintInterface):
    def fit(self, data):
        # Training/initialization logic
        pass
    def extract(self, model):
        # Fingerprint extraction logic
        pass
    def compare(self, fp1, fp2):
        # Fingerprint comparison logic
        pass
```

## 2. Register the New Method
- Register your new fingerprinting method in `fingerprint/fingerprint_factory.py`.
- Follow the existing registration pattern to add your class to the factory or registry.

## 3. Reference Files
- `fingerprint/fingerprint_interface.py`: Fingerprinting method interface definition
- `fingerprint/fingerprint_factory.py`: Fingerprinting method registration and factory

## PlugAE

PlugAE uses the transferred-embedding protocol from Yang et al. (2025). It
optimizes fingerprints only for `pretrained` candidate models, temporarily
transfers each candidate embedding into derivatives declared by the benchmark
configuration, and queries all suspects through their normal LeafBench
generation path. Incompatible transfers are logged as `NaN` and excluded from
aggregate metrics. Model checkpoints are never modified on disk.

The query set in `data/plugae_questions.csv` is copied from the official
MIT-licensed [ProFLingo repository](https://github.com/hengvt/ProFLingo),
which PlugAE uses in its experiments.

If you have any questions, please refer to the above files or contact the maintainer.
