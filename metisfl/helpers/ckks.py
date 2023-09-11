import os
import argparse

from typing import Optional
from metisfl.encryption.fhe import CKKS


def get_file_path(file_path: str, default_file_name: str) -> str:
    """Returns the file path. If the file path is the default file name, the file path is set to the current working directory."""

    if file_path == default_file_name:
        file_path = os.path.join(os.getcwd(), file_path)
    return file_path


def generate_keys(
    batch_size: int = 4096,
    scaling_factor_bits: int = 52,
    crypto_context_path: Optional[str] = "crypto_context.txt",
    public_key_path: Optional[str] = "public_key.txt",
    private_key_path: Optional[str] = "private_key.txt"
) -> None:
    """Generates the crypto context and keys and saves them to the specified paths.
        If no path is specified, the default paths are used (crypto_context.txt, public_key.txt, private_key.txt).
        and the files are saved in the current working directory.

    Parameters
    ----------
    batch_size : int, (default=4096)
        The batch size of the encryption scheme.
    scaling_factor_bits : int, (default=52)
        The number of bits to use for the scaling factor.
    crypto_context_path : Optional[str], (default="crypto_context.txt")
        The path to the crypto context file. By default "crypto_context.txt"
    public_key_path : Optional[str], (default="public_key.txt")
        The path to the public key file. By default "public_key.txt"
    private_key_path : Optional[str], (default="private_key.txt")
        The path to the private key file. By default "private_key.txt"
    """

    crypto_context_path = get_file_path(
        crypto_context_path, "crypto_context.txt")
    public_key_path = get_file_path(public_key_path, "public_key.txt")
    private_key_path = get_file_path(private_key_path, "private_key.txt")

    CKKS.gen_crypto_params_files(batch_size, scaling_factor_bits,
                                 crypto_context_path,
                                 public_key_path,
                                 private_key_path)


if __name__ == "__main__":

    args = argparse.ArgumentParser()

    args.add_argument("--batch_size", type=int, default=8192)
    args.add_argument("--scaling_factor_bits", type=int, default=40)
    args.add_argument("--crypto_context_path", type=str,
                      default="crypto_context.txt")
    args.add_argument("--public_key_path", type=str, default="public_key.txt")
    args.add_argument("--private_key_path", type=str,
                      default="private_key.txt")
    args = args.parse_args()

    generate_keys(**vars(args))
