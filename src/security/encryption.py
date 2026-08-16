"""Fernet Key Encryption & Decryption Manager for NVIDIA NIM Credentials.

Provides secure at-rest and in-transit credential protection using
cryptography.fernet AES-128-CBC with HMAC authentication.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple
from dotenv import load_dotenv
from cryptography.fernet import Fernet, InvalidToken

# Load environment variables if present
load_dotenv()


class EncryptionError(Exception):
    """Raised when encryption or decryption operations fail."""
    pass


def generate_fernet_key() -> str:
    """Generate a new secure base64-encoded 32-byte Fernet key."""
    return Fernet.generate_key().decode("utf-8")


def encrypt_api_key(api_key: str, fernet_key: str | bytes) -> str:
    """Encrypt a plaintext NVIDIA NIM API key using a Fernet key.
    
    Args:
        api_key: Plaintext API key (e.g. nvapi-...)
        fernet_key: 32-byte url-safe base64-encoded secret key
        
    Returns:
        Encrypted ciphertext string (base64-encoded Fernet token)
    """
    if not api_key or not isinstance(api_key, str):
        raise EncryptionError("API key to encrypt must be a non-empty string.")
    
    try:
        key_bytes = fernet_key.encode("utf-8") if isinstance(fernet_key, str) else fernet_key
        f = Fernet(key_bytes)
        token = f.encrypt(api_key.strip().encode("utf-8"))
        return token.decode("utf-8")
    except Exception as e:
        raise EncryptionError(f"Encryption failed: {str(e)}") from e


def decrypt_api_key(encrypted_token: str | bytes, fernet_key: str | bytes) -> str:
    """Decrypt a Fernet ciphertext token back into the plaintext NVIDIA NIM API key.
    
    Args:
        encrypted_token: Encrypted base64 Fernet token
        fernet_key: 32-byte url-safe base64-encoded secret key
        
    Returns:
        Plaintext API key string
    """
    if not encrypted_token:
        raise EncryptionError("Encrypted token must not be empty.")
    if not fernet_key:
        raise EncryptionError("Fernet secret key must not be empty.")

    try:
        key_bytes = fernet_key.encode("utf-8") if isinstance(fernet_key, str) else fernet_key
        token_bytes = encrypted_token.encode("utf-8") if isinstance(encrypted_token, str) else encrypted_token
        f = Fernet(key_bytes)
        decrypted_bytes = f.decrypt(token_bytes)
        return decrypted_bytes.decode("utf-8")
    except InvalidToken as e:
        raise EncryptionError("Decryption failed: Invalid token or mismatched Fernet secret key.") from e
    except Exception as e:
        raise EncryptionError(f"Decryption failed: {str(e)}") from e


def resolve_api_credentials(
    encrypted_token: Optional[str] = None,
    fernet_key: Optional[str] = None,
    raw_api_key: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """Resolve and decrypt API key from parameters or environment variables.
    
    Priority order:
    1. Provided encrypted_token + fernet_key (parameters)
    2. Provided raw_api_key (parameter)
    3. Environment variables: FERNET_ENCRYPTED_NVIDIA_API_KEY + FERNET_SECRET_KEY
    4. Environment variables: NVIDIA_NIM_API_KEY or NVIDIA_API_KEY
    
    Returns:
        Tuple of (resolved_api_key_or_none, credential_source_description)
    """
    # 1. Direct encrypted inputs
    if encrypted_token and fernet_key:
        try:
            key = decrypt_api_key(encrypted_token, fernet_key)
            return key, "decrypted_from_provided_token"
        except Exception as e:
            raise EncryptionError(f"Failed to decrypt provided token: {e}")

    # 2. Direct raw API key
    if raw_api_key and raw_api_key.strip():
        return raw_api_key.strip(), "provided_raw_key"

    # 3. Encrypted environment variables
    env_enc = os.getenv("FERNET_ENCRYPTED_NVIDIA_API_KEY")
    env_key = os.getenv("FERNET_SECRET_KEY")
    if env_enc and env_key:
        try:
            key = decrypt_api_key(env_enc, env_key)
            return key, "decrypted_from_environment"
        except Exception as e:
            raise EncryptionError(f"Failed to decrypt FERNET_ENCRYPTED_NVIDIA_API_KEY from env: {e}")

    # 4. Plaintext environment variables
    plain_env = os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY")
    if plain_env and plain_env.strip():
        return plain_env.strip(), "environment_raw_key"

    return None, "none"


def get_nvidia_api_key(
    encrypted_token: Optional[str] = None,
    fernet_key: Optional[str] = None,
    raw_api_key: Optional[str] = None,
) -> Optional[str]:
    """Convenience helper to retrieve decrypted NVIDIA NIM API key."""
    key, _ = resolve_api_credentials(encrypted_token, fernet_key, raw_api_key)
    return key


def mask_api_key(api_key: Optional[str]) -> str:
    """Mask an API key for safe UI and logging display."""
    if not api_key:
        return "[Not Configured]"
    cleaned = api_key.strip()
    if len(cleaned) <= 8:
        return "********"
    prefix = cleaned[:5]
    suffix = cleaned[-4:]
    return f"{prefix}...{suffix}"
