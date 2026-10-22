// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The cache directory: where each encoded state lives, and what to evict.
//!
//! The directory tracks slots, not memory. The Python executor owns the device
//! and host buffers and applies the copies each operation returns. Each tier is
//! a pool of fixed-size slots, and an entry takes as many slots as its bytes
//! need, so entries of any size fit without fragmentation.
//!
//! Eviction follows GreedyDual-Size-Frequency by default: an entry's priority
//! is its tier's clock plus its hits times its recompute cost per byte, the
//! lowest priority goes first, and the tier's clock rises to it, so long-unused
//! entries age out. Each tier has its own clock, because the host holds row
//! representations that cost far more per byte than contexts, and dropping one
//! must not age the device's entries. An entry that moves to another tier
//! takes a priority from that tier's clock and the hits it has.
//! Least-recently-used eviction is there to compare against. Each tier keeps
//! its unpinned entries ordered by priority, so finding the next victim takes
//! logarithmic time.
//!
//! Each entry also records the rows it was computed from, so the node can find
//! every entry a changed row invalidates. A pin is a count: several steps may
//! read one entry at once, and it stays in place until the last releases it.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
use std::num::NonZeroU64;

use crate::context::{Context, Fingerprint};
use crate::table::TableId;

/// Row ids per block in the index that finds the entries a change touches.
const BLOCK_BITS: u32 = 12;

/// A loaded model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId(pub u32);

/// A version of a table's fitted state. Refitting starts a new epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Epoch(pub u64);

/// What an entry holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// Row representations of a unit's rows.
    Rows,
    /// The context cache a unit's rows encode to.
    Context,
}

impl Kind {
    /// The kind's name, as workers read it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Rows => "rows",
            Self::Context => "context",
        }
    }
}

/// What an entry is keyed on: model, table and its state, and rows with versions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CacheKey {
    pub model: ModelId,
    pub table: TableId,
    pub epoch: Epoch,
    pub kind: Kind,
    pub rows: Fingerprint,
}

/// Which unpinned entry a full tier gives up first.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Eviction {
    /// The one with the least recompute cost per byte times its hits, aged by
    /// a clock that rises to each evicted priority (GreedyDual-Size-Frequency).
    #[default]
    Gdsf,
    /// The one used least recently, whatever it costs to recompute.
    Lru,
}

impl Eviction {
    /// Every policy, for parsing names.
    pub const ALL: [Self; 2] = [Self::Gdsf, Self::Lru];

    /// The policy's name, as configurations spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Gdsf => "gdsf",
            Self::Lru => "lru",
        }
    }

    /// An entry's priority after a use: the lowest is evicted first.
    fn priority(self, clock: f64, uses: u64, hits: u64, cost: f64, bytes: u64) -> f64 {
        match self {
            Self::Gdsf => clock + hits as f64 * cost / bytes.max(1) as f64,
            Self::Lru => uses as f64,
        }
    }

    /// An entry's priority in the tier it moved to, whose clock is `clock`.
    /// Recency is kept as it was.
    fn moved(self, entry: &Entry, clock: f64) -> f64 {
        match self {
            Self::Gdsf => self.priority(clock, 0, entry.hits, entry.cost, entry.bytes),
            Self::Lru => entry.priority,
        }
    }
}

/// A memory tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    Device,
    Host,
    /// A file-backed pool that contexts spill to from the host, if configured.
    Disk,
}

impl Tier {
    /// The tier's name, as workers read it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Host => "host",
            Self::Disk => "disk",
        }
    }
}

impl Tier {
    fn index(self) -> usize {
        match self {
            Self::Device => 0,
            Self::Host => 1,
            Self::Disk => 2,
        }
    }
}

/// A slot's index in its tier's pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotId(pub u32);

/// A tier's pool: `slots` slots of `slot_bytes` each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierConfig {
    pub slot_bytes: NonZeroU64,
    pub slots: u32,
}

/// Where an entry lives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub tier: Tier,
    pub slots: Vec<SlotId>,
}

/// A copy the executor makes before using the slots an operation returns:
/// from an entry's old slots to its new placement, or a drop when `to` is None.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Movement {
    pub key: CacheKey,
    pub from: Tier,
    pub to: Option<Placement>,
}

/// The device slots an entry was given, and the copies to make first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    pub slots: Vec<SlotId>,
    pub movements: Vec<Movement>,
}

/// Why an operation was refused. The directory is left unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheError {
    /// The entry is larger than the whole device tier.
    TooLarge {
        bytes: u64,
        capacity: u64,
    },
    /// Pinned entries hold the slots the entry needs.
    NoRoom,
    Exists(CacheKey),
    Unknown(CacheKey),
    OnDevice(CacheKey),
    /// Tiers whose slots differ in size: moving an entry copies its slots.
    SlotSizes,
    /// A recompute cost that is negative or not finite, which as a priority
    /// would never be evicted.
    Cost,
}

impl fmt::Display for CacheError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge { bytes, capacity } => {
                write!(formatter, "{bytes} bytes exceed the device's {capacity}")
            }
            Self::NoRoom => write!(formatter, "pinned entries hold the slots needed"),
            Self::Exists(key) => write!(formatter, "{key:?} is already cached"),
            Self::Unknown(key) => write!(formatter, "{key:?} is not cached"),
            Self::OnDevice(key) => write!(formatter, "{key:?} is already on the device"),
            Self::SlotSizes => write!(formatter, "every tier needs slots of one size"),
            Self::Cost => write!(formatter, "a recompute cost is finite and not negative"),
        }
    }
}

impl std::error::Error for CacheError {}

#[derive(Debug)]
struct Pool {
    config: TierConfig,
    free: BTreeSet<SlotId>,
}

impl Pool {
    fn new(config: TierConfig) -> Self {
        Self {
            config,
            free: (0..config.slots).map(SlotId).collect(),
        }
    }

    fn slots_for(&self, bytes: u64) -> u64 {
        bytes.div_ceil(self.config.slot_bytes.get()).max(1)
    }

    /// Takes the lowest free slots, so freed slots are reused first.
    fn take(&mut self, count: u64) -> Vec<SlotId> {
        (0..count)
            .map(|_| self.free.pop_first().expect("callers make room first"))
            .collect()
    }

    fn release(&mut self, slots: &[SlotId]) {
        self.free.extend(slots.iter().copied());
    }
}

#[derive(Debug)]
struct Entry {
    placement: Placement,
    bytes: u64,
    cost: f64,
    hits: u64,
    priority: f64,
    /// How many running steps read the entry.
    pinned: u32,
    /// The rows the entry was computed from.
    rows: Context,
    /// Rows written since, whose representations the entry no longer holds.
    /// A page's entry stays as a source for the page's next version, which
    /// copies the rest.
    stale: Context,
}

impl Entry {
    fn is_pinned(&self) -> bool {
        self.pinned > 0
    }

    /// Where the entry sits in its tier's eviction order: the tier, its rank
    /// and its slots, or None while it is pinned.
    fn evictable(&self) -> Option<(usize, Rank, u64)> {
        (!self.is_pinned()).then(|| {
            (
                self.placement.tier.index(),
                Rank(self.priority),
                self.placement.slots.len() as u64,
            )
        })
    }
}

/// A priority, totally ordered so it can order a set.
#[derive(Clone, Copy, Debug)]
struct Rank(f64);

impl PartialEq for Rank {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Rank {}

impl PartialOrd for Rank {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rank {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// A tier's unpinned entries, lowest priority first, and the slots they hold.
#[derive(Debug, Default)]
struct Evictable {
    order: BTreeSet<(Rank, CacheKey)>,
    slots: u64,
}

/// The directory of cached entries across the device, host and disk tiers.
///
/// Entries evicted from the device move to the host, and contexts evicted from
/// the host spill to disk; row representations are cheap to recompute, so they
/// are dropped instead. Any entry below the device promotes straight to it.
#[derive(Debug)]
pub struct Cache {
    device: Pool,
    host: Pool,
    disk: Pool,
    entries: HashMap<CacheKey, Entry>,
    /// Each table's entries, so a table's are found without a full scan.
    by_table: HashMap<TableId, BTreeSet<CacheKey>>,
    /// The entries holding rows in each block of `1 << BLOCK_BITS` row ids,
    /// so a change finds the entries near its rows rather than all of them.
    by_block: HashMap<(TableId, u32), HashSet<CacheKey>>,
    /// Per tier, by `Tier::index`.
    evictable: [Evictable; 3],
    /// Per tier, by `Tier::index`.
    clocks: [f64; 3],
    eviction: Eviction,
    /// Admissions and hits so far, which order entries by recency.
    uses: u64,
}

impl Cache {
    /// An empty directory over a device and a host pool.
    ///
    /// # Errors
    ///
    /// `SlotSizes` if the tiers' slots differ in size.
    pub fn new(device: TierConfig, host: TierConfig) -> Result<Self, CacheError> {
        if device.slot_bytes != host.slot_bytes {
            return Err(CacheError::SlotSizes);
        }
        let disk = TierConfig {
            slot_bytes: device.slot_bytes,
            slots: 0,
        };
        Ok(Self {
            device: Pool::new(device),
            host: Pool::new(host),
            disk: Pool::new(disk),
            entries: HashMap::new(),
            by_table: HashMap::new(),
            by_block: HashMap::new(),
            evictable: Default::default(),
            clocks: [0.0; 3],
            eviction: Eviction::Gdsf,
            uses: 0,
        })
    }

    /// Adds an entry, and to its tier's eviction order if it is unpinned.
    fn put(&mut self, key: CacheKey, entry: Entry) {
        if !entry.is_pinned() {
            let evictable = &mut self.evictable[entry.placement.tier.index()];
            evictable.order.insert((Rank(entry.priority), key));
            evictable.slots += entry.placement.slots.len() as u64;
        }
        self.by_table.entry(key.table).or_default().insert(key);
        self.entries.insert(key, entry);
    }

    /// Removes an entry, and from its tier's eviction order.
    fn take(&mut self, key: &CacheKey) -> Option<Entry> {
        let entry = self.entries.remove(key)?;
        if !entry.is_pinned() {
            let evictable = &mut self.evictable[entry.placement.tier.index()];
            evictable.order.remove(&(Rank(entry.priority), *key));
            evictable.slots -= entry.placement.slots.len() as u64;
        }
        if let Some(keys) = self.by_table.get_mut(&key.table) {
            keys.remove(key);
            if keys.is_empty() {
                self.by_table.remove(&key.table);
            }
        }
        Some(entry)
    }

    /// Changes an entry in place, keeping the eviction order current: it is
    /// touched only if the change moved the entry within it, as a pin, an
    /// unpin or a new priority does.
    fn change(
        &mut self,
        key: &CacheKey,
        change: impl FnOnce(&mut Entry),
    ) -> Result<(), CacheError> {
        let entry = self.entries.get_mut(key).ok_or(CacheError::Unknown(*key))?;
        let before = entry.evictable();
        change(entry);
        let after = entry.evictable();
        if before != after {
            if let Some((tier, rank, slots)) = before {
                let evictable = &mut self.evictable[tier];
                evictable.order.remove(&(rank, *key));
                evictable.slots -= slots;
            }
            if let Some((tier, rank, slots)) = after {
                let evictable = &mut self.evictable[tier];
                evictable.order.insert((rank, *key));
                evictable.slots += slots;
            }
        }
        Ok(())
    }

    /// The directory with another eviction policy; set before anything is admitted.
    #[must_use]
    pub fn with_eviction(mut self, eviction: Eviction) -> Self {
        self.eviction = eviction;
        self
    }

    /// The directory with a disk tier that contexts spill to from the host.
    ///
    /// # Errors
    ///
    /// `SlotSizes` if the disk's slots differ in size from the other tiers'.
    pub fn with_disk(mut self, disk: TierConfig) -> Result<Self, CacheError> {
        if disk.slot_bytes != self.device.config.slot_bytes {
            return Err(CacheError::SlotSizes);
        }
        self.disk = Pool::new(disk);
        Ok(self)
    }

    /// Every cached entry of a table, in key order.
    #[must_use]
    pub fn keys_of(&self, table: TableId) -> Vec<CacheKey> {
        self.by_table
            .get(&table)
            .map(|keys| keys.iter().copied().collect())
            .unwrap_or_default()
    }

    fn pool_mut(&mut self, tier: Tier) -> &mut Pool {
        match tier {
            Tier::Device => &mut self.device,
            Tier::Host => &mut self.host,
            Tier::Disk => &mut self.disk,
        }
    }

    /// Whether an entry is pinned by a step that reads it.
    #[must_use]
    pub fn pinned(&self, key: &CacheKey) -> bool {
        self.entries.get(key).is_some_and(Entry::is_pinned)
    }

    /// Whether an entry is cached, without counting a hit.
    #[must_use]
    pub fn contains(&self, key: &CacheKey) -> bool {
        self.entries.contains_key(key)
    }

    /// The rows an entry was computed from, if it is cached.
    #[must_use]
    pub fn rows(&self, key: &CacheKey) -> Option<&Context> {
        self.entries.get(key).map(|entry| &entry.rows)
    }

    /// The rows of an entry written since it was computed, if it is cached.
    #[must_use]
    pub fn stale(&self, key: &CacheKey) -> Option<&Context> {
        self.entries.get(key).map(|entry| &entry.stale)
    }

    /// Notes that `rows` were written since an entry was computed, so a
    /// later version of its page copies every other row from it.
    ///
    /// # Errors
    ///
    /// If the entry is not cached.
    pub fn mark_stale(&mut self, key: &CacheKey, rows: &Context) -> Result<(), CacheError> {
        self.change(key, |entry| entry.stale |= rows)
    }

    /// A table's cached entries that hold any of `rows`, in key order.
    #[must_use]
    pub fn holding(&self, table: TableId, rows: &Context) -> Vec<CacheKey> {
        let near: BTreeSet<CacheKey> = rows
            .blocks(BLOCK_BITS)
            .filter_map(|block| self.by_block.get(&(table, block)))
            .flatten()
            .copied()
            .collect();
        near.into_iter()
            .filter(|key| self.entries[key].rows.intersects(rows))
            .collect()
    }

    /// Indexes a new entry by the blocks of rows it holds.
    fn index(&mut self, key: CacheKey) {
        let blocks: Vec<u32> = self.entries[&key].rows.blocks(BLOCK_BITS).collect();
        for block in blocks {
            self.by_block
                .entry((key.table, block))
                .or_default()
                .insert(key);
        }
    }

    /// Forgets an entry the cache dropped, which `take` left indexed.
    fn unindex(&mut self, key: &CacheKey, entry: &Entry) {
        for block in entry.rows.blocks(BLOCK_BITS) {
            if let Some(keys) = self.by_block.get_mut(&(key.table, block)) {
                keys.remove(key);
                if keys.is_empty() {
                    self.by_block.remove(&(key.table, block));
                }
            }
        }
    }

    /// Bytes the device tier holds.
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        u64::from(self.device.config.slots) * self.device.config.slot_bytes.get()
    }

    /// Bytes in a slot, on either tier.
    #[must_use]
    pub fn slot_bytes(&self) -> u64 {
        self.device.config.slot_bytes.get()
    }

    /// Free slots in a tier.
    #[must_use]
    pub fn free_slots(&self, tier: Tier) -> u32 {
        u32::try_from(self.pool(tier).free.len()).expect("pools have at most u32::MAX slots")
    }

    /// Bytes of a tier's slots that entries hold.
    #[must_use]
    pub fn used_bytes(&self, tier: Tier) -> u64 {
        let pool = self.pool(tier);
        let used = u64::from(pool.config.slots) - pool.free.len() as u64;
        used * pool.config.slot_bytes.get()
    }

    fn pool(&self, tier: Tier) -> &Pool {
        match tier {
            Tier::Device => &self.device,
            Tier::Host => &self.host,
            Tier::Disk => &self.disk,
        }
    }

    /// Where an entry lives, counting a hit, or None if it is not cached.
    pub fn lookup(&mut self, key: &CacheKey) -> Option<Placement> {
        if !self.entries.contains_key(key) {
            return None;
        }
        self.uses += 1;
        let clock = self.clocks[self.entries[key].placement.tier.index()];
        let (eviction, uses) = (self.eviction, self.uses);
        self.change(key, |entry| {
            entry.hits += 1;
            entry.priority = eviction.priority(clock, uses, entry.hits, entry.cost, entry.bytes);
        })
        .ok()?;
        Some(self.entries[key].placement.clone())
    }

    /// Places a new entry on the device, evicting or demoting others to make room.
    ///
    /// `cost` is what recomputing the entry would cost, in any consistent unit,
    /// and `rows` are the rows it is computed from.
    ///
    /// # Errors
    ///
    /// If the key is cached, the entry is larger than the device, pinned
    /// entries hold the slots it needs, or `cost` is negative or not finite.
    pub fn admit(
        &mut self,
        key: CacheKey,
        bytes: u64,
        cost: f64,
        rows: Context,
    ) -> Result<Admission, CacheError> {
        check_cost(cost)?;
        if self.entries.contains_key(&key) {
            return Err(CacheError::Exists(key));
        }
        let needed = self.device.slots_for(bytes);
        let movements = self.make_device_room(needed, bytes)?;
        let slots = self.device.take(needed);
        let placement = Placement {
            tier: Tier::Device,
            slots: slots.clone(),
        };
        let entry = self.entry(placement, bytes, cost, rows);
        self.put(key, entry);
        self.index(key);
        Ok(Admission { slots, movements })
    }

    /// Places a new entry on the host, dropping unpinned host entries to make room.
    ///
    /// For entries that are read on the device only while a step computes from
    /// them, such as row representations, so they never take device memory.
    ///
    /// # Errors
    ///
    /// If the key is cached, the entry is larger than the host, pinned
    /// entries hold the slots it needs, or `cost` is negative or not finite.
    pub fn admit_host(
        &mut self,
        key: CacheKey,
        bytes: u64,
        cost: f64,
        rows: Context,
    ) -> Result<Admission, CacheError> {
        check_cost(cost)?;
        if self.entries.contains_key(&key) {
            return Err(CacheError::Exists(key));
        }
        let needed = self.host.slots_for(bytes);
        let capacity = u64::from(self.host.config.slots) * self.host.config.slot_bytes.get();
        if needed > u64::from(self.host.config.slots) {
            return Err(CacheError::TooLarge { bytes, capacity });
        }
        let movements = self.drop_host_room(needed).ok_or(CacheError::NoRoom)?;
        let slots = self.host.take(needed);
        let placement = Placement {
            tier: Tier::Host,
            slots: slots.clone(),
        };
        let entry = self.entry(placement, bytes, cost, rows);
        self.put(key, entry);
        self.index(key);
        Ok(Admission { slots, movements })
    }

    /// A new, unpinned entry with one hit.
    fn entry(&mut self, placement: Placement, bytes: u64, cost: f64, rows: Context) -> Entry {
        self.uses += 1;
        let clock = self.clocks[placement.tier.index()];
        Entry {
            placement,
            bytes,
            cost,
            hits: 1,
            priority: self.eviction.priority(clock, self.uses, 1, cost, bytes),
            pinned: 0,
            rows,
            stale: Context::new(),
        }
    }

    /// Frees `needed` host slots by moving the lowest-priority unpinned host
    /// entries down, or returns None without changing anything if it cannot.
    fn drop_host_room(&mut self, needed: u64) -> Option<Vec<Movement>> {
        let evictable = self.evictable[Tier::Host.index()].slots;
        if (self.host.free.len() as u64) + evictable < needed {
            return None;
        }
        let mut movements = Vec::new();
        while (self.host.free.len() as u64) < needed {
            let victim = self.lowest(Tier::Host);
            movements.extend(self.spill(victim));
        }
        Some(movements)
    }

    /// Moves a host entry to disk if it is a context and the disk can take it,
    /// dropping the disk's lowest unpinned entries for room; otherwise drops it.
    fn spill(&mut self, key: CacheKey) -> Vec<Movement> {
        let mut entry = self.take(&key).expect("victims exist");
        self.raise_clock(Tier::Host, entry.priority);
        self.host.release(&entry.placement.slots);
        let needed = self.disk.slots_for(entry.bytes);
        let evictable = self.evictable[Tier::Disk.index()].slots;
        let fits = (self.disk.free.len() as u64) + evictable >= needed;
        if key.kind != Kind::Context || !fits {
            self.unindex(&key, &entry);
            return vec![Movement {
                key,
                from: Tier::Host,
                to: None,
            }];
        }
        let mut movements = Vec::new();
        while (self.disk.free.len() as u64) < needed {
            let victim = self.lowest(Tier::Disk);
            let dropped = self.take(&victim).expect("victims exist");
            self.unindex(&victim, &dropped);
            self.raise_clock(Tier::Disk, dropped.priority);
            self.disk.release(&dropped.placement.slots);
            movements.push(Movement {
                key: victim,
                from: Tier::Disk,
                to: None,
            });
        }
        let placement = Placement {
            tier: Tier::Disk,
            slots: self.disk.take(needed),
        };
        movements.push(Movement {
            key,
            from: Tier::Host,
            to: Some(placement.clone()),
        });
        entry.priority = self.eviction.moved(&entry, self.clocks[Tier::Disk.index()]);
        self.put(key, Entry { placement, ..entry });
        movements
    }

    /// Moves a host or disk entry to the device, making room as `admit` does.
    ///
    /// # Errors
    ///
    /// If the entry is already on the device, or pinned entries hold the slots it needs.
    pub fn promote(&mut self, key: &CacheKey) -> Result<Admission, CacheError> {
        let entry = self.entries.get(key).ok_or(CacheError::Unknown(*key))?;
        let from = entry.placement.tier;
        if from == Tier::Device {
            return Err(CacheError::OnDevice(*key));
        }
        let (bytes, host_slots) = (entry.bytes, entry.placement.slots.clone());
        let needed = self.device.slots_for(bytes);
        // Held in place, so demoting others to the host cannot drop it.
        self.pin(key)?;
        let room = self.make_device_room(needed, bytes);
        self.unpin(key)?;
        let mut movements = room?;
        let slots = self.device.take(needed);
        self.pool_mut(from).release(&host_slots);
        let placement = Placement {
            tier: Tier::Device,
            slots: slots.clone(),
        };
        let moved = placement.clone();
        let (eviction, clock) = (self.eviction, self.clocks[Tier::Device.index()]);
        self.change(key, |entry| {
            entry.placement = moved;
            entry.priority = eviction.moved(entry, clock);
        })?;
        movements.push(Movement {
            key: *key,
            from,
            to: Some(placement),
        });
        Ok(Admission { slots, movements })
    }

    /// Frees `needed` device slots by demoting the lowest-priority unpinned entries.
    ///
    /// Checks that enough slots can be freed before changing anything.
    fn make_device_room(&mut self, needed: u64, bytes: u64) -> Result<Vec<Movement>, CacheError> {
        let capacity = u64::from(self.device.config.slots) * self.device.config.slot_bytes.get();
        if needed > u64::from(self.device.config.slots) {
            return Err(CacheError::TooLarge { bytes, capacity });
        }
        let evictable = self.evictable[Tier::Device.index()].slots;
        if self.device.free.len() as u64 + evictable < needed {
            return Err(CacheError::NoRoom);
        }
        let mut movements = Vec::new();
        while (self.device.free.len() as u64) < needed {
            let victim = self.lowest(Tier::Device);
            movements.extend(self.demote(victim));
        }
        Ok(movements)
    }

    /// A tier's lowest-priority unpinned entry, ties broken by key for determinism.
    fn lowest(&self, tier: Tier) -> CacheKey {
        self.evictable[tier.index()]
            .order
            .first()
            .expect("callers check that an eligible entry exists")
            .1
    }

    /// Raises a tier's clock to the priority of an entry it gave up.
    fn raise_clock(&mut self, tier: Tier, priority: f64) {
        let clock = &mut self.clocks[tier.index()];
        *clock = clock.max(priority);
    }

    /// Moves a device entry to the host, dropping unpinned host entries to make
    /// room, or drops it if it cannot fit there.
    fn demote(&mut self, key: CacheKey) -> Vec<Movement> {
        let mut entry = self.take(&key).expect("victims exist");
        self.raise_clock(Tier::Device, entry.priority);
        self.device.release(&entry.placement.slots);
        let needed = self.host.slots_for(entry.bytes);
        let room = if needed > u64::from(self.host.config.slots) {
            None
        } else {
            self.drop_host_room(needed)
        };
        let Some(mut movements) = room else {
            self.unindex(&key, &entry);
            return vec![Movement {
                key,
                from: Tier::Device,
                to: None,
            }];
        };
        let placement = Placement {
            tier: Tier::Host,
            slots: self.host.take(needed),
        };
        movements.push(Movement {
            key,
            from: Tier::Device,
            to: Some(placement.clone()),
        });
        entry.priority = self.eviction.moved(&entry, self.clocks[Tier::Host.index()]);
        self.put(key, Entry { placement, ..entry });
        movements
    }

    /// Keeps an entry in place while a step reads it. Each pin takes an unpin.
    ///
    /// # Errors
    ///
    /// If the entry is not cached.
    pub fn pin(&mut self, key: &CacheKey) -> Result<(), CacheError> {
        self.change(key, |entry| entry.pinned += 1)
    }

    /// Releases one pin; the entry can be evicted again once every pin is released.
    ///
    /// # Errors
    ///
    /// If the entry is not cached.
    pub fn unpin(&mut self, key: &CacheKey) -> Result<(), CacheError> {
        self.change(key, |entry| {
            debug_assert!(entry.is_pinned(), "unpinned more than pinned");
            entry.pinned = entry.pinned.saturating_sub(1);
        })
    }

    /// Drops a model's unpinned entries for a table from epochs before `epoch`.
    ///
    /// Pinned entries are still being read; drop them with `remove` once released.
    pub fn drop_epochs_before(
        &mut self,
        model: ModelId,
        table: TableId,
        epoch: Epoch,
    ) -> Vec<Movement> {
        let stale: Vec<CacheKey> = self
            .by_table
            .get(&table)
            .into_iter()
            .flatten()
            .filter(|key| key.model == model && key.epoch < epoch && !self.entries[key].is_pinned())
            .copied()
            .collect();
        stale.iter().filter_map(|key| self.remove(key)).collect()
    }

    /// Drops an entry, returning the drop for the executor, or None if it is not cached.
    pub fn remove(&mut self, key: &CacheKey) -> Option<Movement> {
        let entry = self.take(key)?;
        self.unindex(key, &entry);
        let tier = entry.placement.tier;
        self.pool_mut(tier).release(&entry.placement.slots);
        Some(Movement {
            key: *key,
            from: tier,
            to: None,
        })
    }
}

fn check_cost(cost: f64) -> Result<(), CacheError> {
    if cost.is_finite() && cost >= 0.0 {
        Ok(())
    } else {
        Err(CacheError::Cost)
    }
}
