"""Unit tests for the Fernet encryption and decryption module."""

import pytest
from src.security.encryption import (
    EncryptionError,
    decrypt_api_key,
    encrypt_api_key,
    generate_fernet_key,
    get_nvidia_api_key,
    mask_api_key,
    resolve_api_credentials,
)


def test_generate_fernet_key():
    key = generate_fernet_key()
    assert isinstance(key, str)
    assert len(key) == 44  # Standard base64 32-byte Fernet key length


def test_encrypt_and_decrypt_cycle():
    key = generate_fernet_key()
    plain_api_key = "nvapi-test-secret-api-key-1234567890abcdef"
    
    encrypted = encrypt_api_key(plain_api_key, key)
    assert isinstance(encrypted, str)
    assert encrypted != plain_api_key
    assert len(encrypted) > 20
    
    decrypted = decrypt_api_key(encrypted, key)
    assert decrypted == plain_api_key


def test_invalid_key_decryption_fails():
    key1 = generate_fernet_key()
    key2 = generate_fernet_key()
    plain = "nvapi-my-api-key"
    
    encrypted = encrypt_api_key(plain, key1)
    with pytest.raises(EncryptionError):
        decrypt_api_key(encrypted, key2)


def test_empty_or_malformed_inputs():
    key = generate_fernet_key()
    with pytest.raises(EncryptionError):
        encrypt_api_key("", key)
    with pytest.raises(EncryptionError):
        encrypt_api_key("nvapi-key", "invalid-key")
    with pytest.raises(EncryptionError):
        decrypt_api_key("", key)


def test_mask_api_key():
    assert mask_api_key(None) == "[Not Configured]"
    assert mask_api_key("") == "[Not Configured]"
    assert mask_api_key("short") == "********"
    assert mask_api_key("nvapi-1234567890abcdef") == "nvapi...cdef"


def test_resolve_api_credentials_direct():
    key = generate_fernet_key()
    plain = "nvapi-direct-key-999"
    encrypted = encrypt_api_key(plain, key)
    
    resolved, source = resolve_api_credentials(encrypted_token=encrypted, fernet_key=key)
    assert resolved == plain
    assert source == "decrypted_from_provided_token"


def test_resolve_api_credentials_raw():
    plain = "nvapi-raw-test-key"
    resolved, source = resolve_api_credentials(raw_api_key=plain)
    assert resolved == plain
    assert source == "provided_raw_key"


def test_resolve_api_credentials_env(monkeypatch):
    key = generate_fernet_key()
    plain = "nvapi-env-secret-key"
    enc = encrypt_api_key(plain, key)
    
    monkeypatch.setenv("FERNET_SECRET_KEY", key)
    monkeypatch.setenv("FERNET_ENCRYPTED_NVIDIA_API_KEY", enc)
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    
    resolved, source = resolve_api_credentials()
    assert resolved == plain
    assert source == "decrypted_from_environment"
