use std::collections::VecDeque;

use mlx_rs::{
    ops::{
        concatenate_axis,
        indexing::{Ellipsis, IndexOp, TryIndexMutOp},
        zeros_dtype,
    },
    Array,
};

use crate::error::Result;

/// Chunk size: K/V are pre-allocated in `STEP`-token blocks so that decode
/// steps within a block are O(1) in-place writes instead of O(N) concats.
const STEP: i32 = 256;

/// Per-layer KV cache, ported from upstream Python `mlx_lm.models.cache.KVCache`.
///
/// Layout: `[B, H, S, D]`. `offset` is the active sequence length; the
/// underlying buffer is rounded up to the next `STEP`-token boundary. When
/// the active region exceeds the buffer, we extend by `ceil(n_new / STEP)`
/// chunks (the existing buffer is `concatenate`d with a fresh zeros block).
#[derive(Clone, Debug, Default)]
pub struct KvCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl KvCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    pub fn trim(&mut self, count: usize) -> usize {
        let trimmed = count.min(self.offset as usize);
        self.offset -= trimmed as i32;
        trimmed
    }

    /// Active K/V slice views (shape `[B, H, offset, D]`) — i.e. the meaningful
    /// region of the buffer, with any unused padding past `offset` excluded.
    /// Returns `None` before the first update.
    pub fn active(&self) -> Option<(Array, Array)> {
        let k = self.keys.as_ref()?;
        let v = self.values.as_ref()?;
        Some((
            k.index((Ellipsis, 0..self.offset, ..)),
            v.index((Ellipsis, 0..self.offset, ..)),
        ))
    }

    pub fn update_and_fetch(&mut self, k: Array, v: Array) -> Result<(Array, Array)> {
        let kshape = k.shape();
        let s_axis = kshape.len() - 2;
        let n_new = kshape[s_axis];
        let prev = self.offset;
        let new_offset = prev + n_new;

        let cur_cap = self.keys.as_ref().map(|a| a.shape()[s_axis]).unwrap_or(0);
        if new_offset > cur_cap {
            self.grow(&k, &v, prev)?;
        }

        let keys = self.keys.as_mut().expect("keys allocated by grow");
        let values = self.values.as_mut().expect("values allocated by grow");
        keys.try_index_mut((Ellipsis, prev..new_offset, ..), &k)?;
        values.try_index_mut((Ellipsis, prev..new_offset, ..), &v)?;

        // Commit only after both writes succeed — failure leaves `offset` unchanged.
        self.offset = new_offset;

        Ok((
            keys.index((Ellipsis, 0..new_offset, ..)),
            values.index((Ellipsis, 0..new_offset, ..)),
        ))
    }

    /// Build the new buffers fully before mutating `self`, so a partial
    /// failure cannot leave keys/values out of sync.
    fn grow(&mut self, k: &Array, v: &Array, prev: i32) -> Result<()> {
        let kshape = k.shape();
        let vshape = v.shape();
        let s_axis = kshape.len() - 2;
        let n_new = kshape[s_axis];
        let extra = ((n_new + STEP - 1) / STEP) * STEP;

        let mut pad_kshape = kshape.to_vec();
        pad_kshape[s_axis] = extra;
        let mut pad_vshape = vshape.to_vec();
        pad_vshape[s_axis] = extra;
        let pad_k = zeros_dtype(&pad_kshape, k.dtype())?;
        let pad_v = zeros_dtype(&pad_vshape, v.dtype())?;

        // If `prev` doesn't sit on a STEP boundary, the existing buffer has
        // unused tail past the active region — drop it before extending.
        let trim_tail = prev % STEP != 0;
        let new_keys = extend(self.keys.as_ref(), pad_k, prev, trim_tail)?;
        let new_values = extend(self.values.as_ref(), pad_v, prev, trim_tail)?;

        self.keys = Some(new_keys);
        self.values = Some(new_values);
        Ok(())
    }
}

#[derive(Debug)]
struct PromptCacheEntry {
    tokens: Vec<u32>,
    cache: Vec<KvCache>,
}

#[derive(Debug)]
pub struct PromptCache {
    entries: VecDeque<PromptCacheEntry>,
    max_size: usize,
}

impl PromptCache {
    pub fn new(max_size: usize) -> Self {
        assert!(max_size > 0, "prompt cache max_size must be positive");
        Self {
            entries: VecDeque::new(),
            max_size,
        }
    }

    pub fn fetch_nearest(&self, tokens: &[u32]) -> (Option<Vec<KvCache>>, Vec<u32>) {
        if let Some(entry) = self.entries.iter().find(|entry| entry.tokens == tokens) {
            return (Some(entry.cache.clone()), Vec::new());
        }
        if tokens.is_empty() {
            return (None, Vec::new());
        }

        let shorter = self
            .entries
            .iter()
            .filter(|entry| entry.tokens.len() > 1 && tokens.starts_with(&entry.tokens))
            .max_by_key(|entry| entry.tokens.len());
        let shorter_len = shorter.map_or(0, |entry| entry.tokens.len());

        let mut common_prefix = 0;
        let mut longer: Option<&PromptCacheEntry> = None;
        for entry in &self.entries {
            let prefix = common_prefix_len(tokens, &entry.tokens);
            if prefix == 0 {
                continue;
            }
            let prefer = prefix > common_prefix
                || (prefix == common_prefix
                    && longer.is_none_or(|current| entry.tokens.len() < current.tokens.len()));
            if prefer {
                common_prefix = prefix;
                longer = Some(entry);
            }
        }

        if let Some(entry) = longer.filter(|_| common_prefix > shorter_len) {
            let prefix = (tokens.len() - 1).min(common_prefix);
            let mut cache = entry.cache.clone();
            let trim = entry.tokens.len() - prefix;
            for layer in &mut cache {
                layer.trim(trim);
            }
            return (Some(cache), tokens[prefix..].to_vec());
        }

        if let Some(entry) = shorter {
            return (
                Some(entry.cache.clone()),
                tokens[entry.tokens.len()..].to_vec(),
            );
        }

        (None, tokens.to_vec())
    }

    pub fn insert(&mut self, tokens: Vec<u32>, cache: Vec<KvCache>) {
        self.entries.retain(|entry| entry.tokens != tokens);
        self.entries.retain(|entry| {
            entry.tokens.len() >= tokens.len() || !tokens.starts_with(&entry.tokens)
        });
        self.entries.push_back(PromptCacheEntry { tokens, cache });
        while self.entries.len() > self.max_size {
            self.entries.pop_front();
        }
    }
}

fn common_prefix_len(left: &[u32], right: &[u32]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn extend(existing: Option<&Array>, pad: Array, prev: i32, trim_tail: bool) -> Result<Array> {
    match existing {
        None => Ok(pad),
        Some(buf) => {
            let head = if trim_tail {
                buf.index((Ellipsis, 0..prev, ..))
            } else {
                buf.clone()
            };
            Ok(concatenate_axis(&[head, pad], -2)?)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_with_offset(offset: i32) -> Vec<KvCache> {
        vec![KvCache {
            keys: None,
            values: None,
            offset,
        }]
    }

    #[test]
    fn prompt_cache_fetches_exact_and_shorter_entries() {
        let mut cache = PromptCache::new(10);
        cache.insert(vec![1, 2], cache_with_offset(2));

        let (exact, exact_rest) = cache.fetch_nearest(&[1, 2]);
        assert!(exact_rest.is_empty());
        assert_eq!(exact.unwrap()[0].offset(), 2);

        let (shorter, shorter_rest) = cache.fetch_nearest(&[1, 2, 3]);
        assert_eq!(shorter_rest, [3]);
        assert_eq!(shorter.unwrap()[0].offset(), 2);
    }

    #[test]
    fn prompt_cache_trims_longer_entry_but_preserves_hidden_preseed_token() {
        let mut cache = PromptCache::new(10);
        cache.insert(vec![1, 2, 3, 4], cache_with_offset(5));

        let (nearest, rest) = cache.fetch_nearest(&[1, 2, 9]);
        assert_eq!(rest, [9]);
        assert_eq!(nearest.unwrap()[0].offset(), 3);
    }

    #[test]
    fn prompt_cache_removes_prefixes_and_evicts_oldest_entry() {
        let mut cache = PromptCache::new(2);
        cache.insert(vec![1, 2], cache_with_offset(2));
        cache.insert(vec![1, 2, 3], cache_with_offset(3));
        assert_eq!(cache.entries.len(), 1);

        cache.insert(vec![4], cache_with_offset(1));
        cache.insert(vec![5], cache_with_offset(1));
        assert_eq!(cache.entries.len(), 2);
        assert!(cache.fetch_nearest(&[1, 2, 3]).0.is_none());
    }
}
