use crate::error::Result;
use anyhow::ensure;

/// Encodes plaintext by elementwise xoring with the passed in key.
pub fn encode(key: &[u8], plaintext: &[u8]) -> Result<Vec<u8>> {
    xor(key, plaintext)
}

/// Decodes ciphertext by elementwise xoring with the passed in key.
pub fn decode(key: &[u8], ciphertext: &[u8]) -> Result<Vec<u8>> {
    xor(key, ciphertext)
}

fn xor(key: &[u8], input: &[u8]) -> Result<Vec<u8>> {
    let key_len = key.len();
    ensure!(key_len == 32, "Key must be 32 bytes.");

    Ok(input
        .iter()
        .enumerate()
        .map(|(i, byte)| byte ^ key[i % key_len])
        .collect())
}

