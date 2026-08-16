"""CLI utility to generate Fernet keys and encrypt NVIDIA NIM API credentials."""

from __future__ import annotations

import argparse
import sys
from .encryption import encrypt_api_key, decrypt_api_key, generate_fernet_key, mask_api_key


def main():
    parser = argparse.ArgumentParser(
        description="NVIDIA NIM API Key Fernet Encryption Tool"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # generate-key command
    subparsers.add_parser("genkey", help="Generate a new Fernet secret key")

    # encrypt command
    enc_parser = subparsers.add_parser("encrypt", help="Encrypt an API key using a Fernet key")
    enc_parser.add_argument("--key", "-k", type=str, help="Fernet secret key (if omitted, a new one is generated)")
    enc_parser.add_argument("--api-key", "-a", type=str, required=True, help="Plaintext NVIDIA NIM API key to encrypt")

    # decrypt command
    dec_parser = subparsers.add_parser("decrypt", help="Decrypt a Fernet token to verify")
    dec_parser.add_argument("--key", "-k", type=str, required=True, help="Fernet secret key")
    dec_parser.add_argument("--token", "-t", type=str, required=True, help="Encrypted token")

    args = parser.parse_args()

    if args.command == "genkey":
        key = generate_fernet_key()
        print("\n=======================================================")
        print(" Generated Fernet Secret Key")
        print("=======================================================")
        print(f"FERNET_SECRET_KEY={key}")
        print("\nKeep this secret key safe! Do NOT commit this to Git.\n")

    elif args.command == "encrypt":
        fernet_key = args.key or generate_fernet_key()
        token = encrypt_api_key(args.api_key, fernet_key)
        print("\n=======================================================")
        print(" NVIDIA NIM API Key Encrypted Successfully")
        print("=======================================================")
        print(f"Masked Original API Key: {mask_api_key(args.api_key)}")
        print(f"FERNET_SECRET_KEY={fernet_key}")
        print(f"FERNET_ENCRYPTED_NVIDIA_API_KEY={token}")
        print("=======================================================")
        print("\nTo use in your local environment (.env file):")
        print(f"FERNET_SECRET_KEY={fernet_key}")
        print(f"FERNET_ENCRYPTED_NVIDIA_API_KEY={token}\n")
        print("To use in GitHub Actions Repository Secrets:")
        print("1. Set FERNET_SECRET_KEY as a secret")
        print("2. Set FERNET_ENCRYPTED_NVIDIA_API_KEY as a secret\n")

    elif args.command == "decrypt":
        try:
            plain = decrypt_api_key(args.token, args.key)
            print("\n=======================================================")
            print(" Decryption Verification Succeeded")
            print("=======================================================")
            print(f"Decrypted API Key: {mask_api_key(plain)}")
            print("Token is valid and matches the Fernet secret key.\n")
        except Exception as e:
            print(f"\n[ERROR] Decryption failed: {e}\n", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
