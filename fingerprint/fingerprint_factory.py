from fingerprint.fingerprint_interface import LLMFingerprintInterface
from fingerprint.plugae.plugae import PlugAEFingerprint
from fingerprint.reef.reef import REEFFingerprint
from fingerprint.trap.trap import TRAPFingerprint
from fingerprint.zeroprint.zeroprint import ZeroPrintFingerprint


def create_fingerprint_method(config=None, accelerator=None) -> LLMFingerprintInterface:
    method_name = config.get("fingerprint_method", None)
    if method_name == "reef":
        return REEFFingerprint(config=config, accelerator=accelerator)
    elif method_name == "plugae":
        return PlugAEFingerprint(config=config, accelerator=accelerator)
    elif method_name == "trap":
        return TRAPFingerprint(config=config, accelerator=accelerator)
    elif method_name == "zeroprint":
        return ZeroPrintFingerprint(config=config, accelerator=accelerator)
    else:
        raise ValueError(f"Unknown fingerprinting method: {method_name}")
