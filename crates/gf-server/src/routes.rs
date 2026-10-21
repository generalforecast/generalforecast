//! Which shard a request's context goes to, by one rule for gf-server's
//! front and the in-process engine: a context a shard holds goes back to
//! that shard, so a repeat finds its table and cache, and a new one goes to
//! the least busy shard, by `gf_core::route::least_busy`.
//!
//! A shard forgets a context as it drops the context's ephemeral table, and
//! the map keeps at most as many contexts as the shards hold tables,
//! dropping the least recently routed first.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use gf_core::route;

/// The shards' loads, and the shard each context's table is on.
pub struct Routes {
    loads: Vec<AtomicUsize>,
    owners: Mutex<Owners>,
}

/// Each routed context's shard, least recently routed first.
struct Owners {
    capacity: usize,
    tick: u64,
    shards: HashMap<[u8; 32], (usize, u64)>,
    order: BTreeMap<u64, [u8; 32]>,
}

/// A command a shard is answering, counted in its load until dropped.
pub struct Busy<'a> {
    pub shard: usize,
    load: &'a AtomicUsize,
}

impl Drop for Busy<'_> {
    fn drop(&mut self) {
        self.load.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Routes {
    /// Routes over `shards` shards, remembering at most `capacity` contexts.
    ///
    /// # Panics
    ///
    /// If there are no shards.
    #[must_use]
    pub fn new(shards: usize, capacity: usize) -> Self {
        assert!(shards > 0, "a context needs a shard to route to");
        Self {
            loads: (0..shards).map(|_| AtomicUsize::new(0)).collect(),
            owners: Mutex::new(Owners {
                capacity,
                tick: 0,
                shards: HashMap::new(),
                order: BTreeMap::new(),
            }),
        }
    }

    #[must_use]
    pub fn shards(&self) -> usize {
        self.loads.len()
    }

    /// The shard for a request's context: the one holding it, or else the
    /// least busy, which it is then remembered on; busy until dropped.
    pub fn context(&self, fingerprint: &[u8; 32]) -> Busy<'_> {
        let mut owners = self.owners();
        let shard = owners.touch(fingerprint).unwrap_or_else(|| {
            let loads: Vec<usize> = self
                .loads
                .iter()
                .map(|load| load.load(Ordering::Relaxed))
                .collect();
            let shard = route::least_busy(fingerprint, &loads);
            owners.insert(*fingerprint, shard);
            shard
        });
        self.shard(shard)
    }

    /// Shard `shard`, busy until dropped.
    #[must_use]
    pub fn shard(&self, shard: usize) -> Busy<'_> {
        let load = &self.loads[shard];
        load.fetch_add(1, Ordering::Relaxed);
        Busy { shard, load }
    }

    /// Commands shard `shard` is answering.
    #[must_use]
    pub fn load(&self, shard: usize) -> usize {
        self.loads[shard].load(Ordering::Relaxed)
    }

    /// The shard a context was routed to, if it is remembered.
    #[must_use]
    pub fn owner(&self, fingerprint: &[u8; 32]) -> Option<usize> {
        self.owners()
            .shards
            .get(fingerprint)
            .map(|&(shard, _)| shard)
    }

    /// Contexts remembered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.owners().shards.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shard `shard` holds the context's table, so its repeats go there.
    pub fn holds(&self, fingerprint: &[u8; 32], shard: usize) {
        let mut owners = self.owners();
        owners.forget(fingerprint);
        owners.insert(*fingerprint, shard);
    }

    /// Shard `shard` dropped the context's table; a context routed to
    /// another shard since stays there.
    pub fn dropped(&self, fingerprint: &[u8; 32], shard: usize) {
        let mut owners = self.owners();
        if owners
            .shards
            .get(fingerprint)
            .is_some_and(|&(held, _)| held == shard)
        {
            owners.forget(fingerprint);
        }
    }

    fn owners(&self) -> std::sync::MutexGuard<'_, Owners> {
        self.owners.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Owners {
    /// The context's shard, now the most recently routed.
    fn touch(&mut self, fingerprint: &[u8; 32]) -> Option<usize> {
        let (shard, _) = self.forget(fingerprint)?;
        self.insert(*fingerprint, shard);
        Some(shard)
    }

    /// Remembers a context's shard, dropping the least recently routed
    /// contexts past the capacity.
    fn insert(&mut self, fingerprint: [u8; 32], shard: usize) {
        if self.capacity == 0 {
            return;
        }
        while self.shards.len() >= self.capacity {
            let Some((_, oldest)) = self.order.pop_first() else {
                break;
            };
            self.shards.remove(&oldest);
        }
        self.tick += 1;
        self.shards.insert(fingerprint, (shard, self.tick));
        self.order.insert(self.tick, fingerprint);
    }

    fn forget(&mut self, fingerprint: &[u8; 32]) -> Option<(usize, u64)> {
        let (shard, tick) = self.shards.remove(fingerprint)?;
        self.order.remove(&tick);
        Some((shard, tick))
    }
}

#[cfg(test)]
mod tests {
    use super::Routes;

    fn fingerprint(first: u64) -> [u8; 32] {
        let mut fingerprint = [0u8; 32];
        fingerprint[..8].copy_from_slice(&first.to_le_bytes());
        fingerprint
    }

    #[test]
    fn new_contexts_spread_over_the_least_busy_shards() {
        let routes = Routes::new(2, 16);
        // Both fingerprints fall to shard 0 by `shard_of`.
        let first = routes.context(&fingerprint(0));
        let second = routes.context(&fingerprint(2));
        assert_eq!((first.shard, second.shard), (0, 1));
        assert_eq!((routes.load(0), routes.load(1)), (1, 1));
        drop(first);
        assert_eq!(routes.load(0), 0);
    }

    #[test]
    fn a_repeat_goes_to_the_shard_holding_its_context_however_busy() {
        let routes = Routes::new(2, 16);
        let held = routes.context(&fingerprint(0));
        let _busy = [routes.shard(0), routes.shard(0)];
        assert_eq!(routes.context(&fingerprint(0)).shard, held.shard);
    }

    #[test]
    fn a_context_is_forgotten_with_its_table_on_its_own_shard_only() {
        let routes = Routes::new(2, 16);
        routes.holds(&fingerprint(0), 1);
        routes.dropped(&fingerprint(0), 0);
        assert_eq!(routes.owner(&fingerprint(0)), Some(1));
        routes.dropped(&fingerprint(0), 1);
        assert_eq!(routes.owner(&fingerprint(0)), None);
        assert!(routes.is_empty());
    }

    #[test]
    fn the_least_recently_routed_contexts_go_past_the_capacity() {
        let routes = Routes::new(2, 2);
        drop(routes.context(&fingerprint(1)));
        drop(routes.context(&fingerprint(2)));
        drop(routes.context(&fingerprint(1)));
        drop(routes.context(&fingerprint(3)));
        assert_eq!(routes.len(), 2);
        assert_eq!(routes.owner(&fingerprint(2)), None);
        assert!(routes.owner(&fingerprint(1)).is_some());
        assert!(routes.owner(&fingerprint(3)).is_some());
        assert!(Routes::new(1, 0).context(&fingerprint(1)).shard == 0);
    }
}
