// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The cache directory: where each encoded state lives, and what to evict.

use std::collections::HashSet;
use std::num::NonZeroU64;

use gf_core::cache::{
    Cache, CacheError, CacheKey, Epoch, Eviction, Kind, ModelId, Movement, Tier, TierConfig,
};
use gf_core::context::{Context, Fingerprint};
use gf_core::table::{RowId, TableId, Version};

const SLOT: u64 = 100;

fn tier(slots: u32) -> TierConfig {
    TierConfig {
        slot_bytes: NonZeroU64::new(SLOT).unwrap(),
        slots,
    }
}

fn key(unit: u32) -> CacheKey {
    key_at(unit, 0, 0)
}

fn key_at(unit: u32, model: u32, epoch: u64) -> CacheKey {
    CacheKey {
        model: ModelId(model),
        table: TableId(0),
        epoch: Epoch(epoch),
        kind: Kind::Context,
        rows: Fingerprint::of(RowId(unit), Version(0)),
    }
}

fn placed_on(cache: &mut Cache, key: &CacheKey) -> Option<Tier> {
    cache.lookup(key).map(|placement| placement.tier)
}

#[test]
fn tiers_of_different_slots_and_bad_costs_are_refused() {
    let small = TierConfig {
        slot_bytes: NonZeroU64::new(SLOT / 2).unwrap(),
        slots: 4,
    };
    assert!(matches!(
        Cache::new(tier(4), small),
        Err(CacheError::SlotSizes)
    ));
    let cache = Cache::new(tier(4), tier(4)).unwrap();
    assert!(matches!(cache.with_disk(small), Err(CacheError::SlotSizes)));

    let mut cache = Cache::new(tier(4), tier(4)).unwrap();
    for cost in [f64::NAN, -1.0] {
        assert_eq!(
            cache.admit(key(1), SLOT, cost, Context::new()),
            Err(CacheError::Cost)
        );
    }
    assert!(placed_on(&mut cache, &key(1)).is_none());
}

#[test]
fn admitted_entries_take_whole_slots_and_are_found() {
    let mut cache = Cache::new(tier(10), tier(10)).unwrap();

    let admission = cache.admit(key(1), 250, 1.0, Context::new()).unwrap();

    assert_eq!(admission.slots.len(), 3);
    assert_eq!(admission.slots.iter().collect::<HashSet<_>>().len(), 3);
    assert!(admission.movements.is_empty());
    assert_eq!(cache.free_slots(Tier::Device), 7);
    let placement = cache.lookup(&key(1)).unwrap();
    assert_eq!(
        (placement.tier, placement.slots),
        (Tier::Device, admission.slots)
    );
}

#[test]
fn entries_are_specific_to_a_model_and_epoch() {
    let mut cache = Cache::new(tier(10), tier(10)).unwrap();
    cache
        .admit(key_at(1, 0, 0), 100, 1.0, Context::new())
        .unwrap();

    assert!(cache.lookup(&key_at(1, 1, 0)).is_none());
    assert!(cache.lookup(&key_at(1, 0, 1)).is_none());
    assert_eq!(
        cache.admit(key_at(1, 0, 0), 100, 1.0, Context::new()),
        Err(CacheError::Exists(key_at(1, 0, 0)))
    );
}

#[test]
fn eviction_keeps_what_is_expensive_to_recompute_per_byte() {
    let mut cache = Cache::new(tier(4), tier(0)).unwrap();
    cache.admit(key(1), 200, 1.0, Context::new()).unwrap(); // cheap and large
    cache.admit(key(2), 200, 50.0, Context::new()).unwrap(); // expensive for its size

    let admission = cache.admit(key(3), 200, 1.0, Context::new()).unwrap();

    assert_eq!(placed_on(&mut cache, &key(2)), Some(Tier::Device));
    assert_eq!(placed_on(&mut cache, &key(1)), None);
    assert_eq!(
        admission.movements,
        [Movement {
            key: key(1),
            from: Tier::Device,
            to: None
        }]
    );
}

#[test]
fn host_evictions_do_not_age_device_entries() {
    let mut cache = Cache::new(tier(2), tier(1)).unwrap();
    cache.admit(key(1), 100, 1.0, Context::new()).unwrap();
    cache.admit(key(2), 100, 1.0, Context::new()).unwrap();
    for _ in 0..5 {
        cache.lookup(&key(1));
    }
    // Representations are far costlier per byte than contexts, so dropping one
    // from the host must not raise the priority a new device entry starts at.
    cache
        .admit_host(rows_key(10), 100, 1_000.0, Context::new())
        .unwrap();
    cache
        .admit_host(rows_key(11), 100, 1_000.0, Context::new())
        .unwrap();

    cache.admit(key(3), 100, 1.0, Context::new()).unwrap();
    cache.admit(key(4), 100, 1.0, Context::new()).unwrap();

    assert_eq!(placed_on(&mut cache, &key(1)), Some(Tier::Device));
    assert_eq!(placed_on(&mut cache, &key(3)), Some(Tier::Host));
}

#[test]
fn recency_eviction_drops_the_least_recently_used_however_often_it_was_used() {
    for (eviction, kept, dropped) in [(Eviction::Gdsf, 1, 2), (Eviction::Lru, 2, 1)] {
        let mut cache = Cache::new(tier(2), tier(0))
            .unwrap()
            .with_eviction(eviction);
        cache.admit(key(1), 100, 1.0, Context::new()).unwrap();
        cache.admit(key(2), 100, 1.0, Context::new()).unwrap();
        for _ in 0..5 {
            cache.lookup(&key(1));
        }
        cache.lookup(&key(2));

        cache.admit(key(3), 100, 1.0, Context::new()).unwrap();

        assert!(cache.lookup(&key(kept)).is_some(), "{eviction:?}");
        assert!(cache.lookup(&key(dropped)).is_none(), "{eviction:?}");
    }
}

#[test]
fn pinned_entries_are_never_evicted_and_admission_refuses_cleanly() {
    let mut cache = Cache::new(tier(3), tier(0)).unwrap();
    cache.admit(key(1), 200, 1.0, Context::new()).unwrap();
    cache.admit(key(2), 100, 1.0, Context::new()).unwrap();
    cache.pin(&key(1)).unwrap();
    cache.pin(&key(2)).unwrap();

    let refused = cache.admit(key(3), 100, 100.0, Context::new());

    assert_eq!(refused, Err(CacheError::NoRoom));
    assert!(cache.lookup(&key(1)).is_some() && cache.lookup(&key(2)).is_some());
    cache.unpin(&key(2)).unwrap();
    assert!(cache.admit(key(3), 100, 1.0, Context::new()).is_ok());
    assert!(cache.lookup(&key(1)).is_some());
}

#[test]
fn entries_larger_than_the_device_are_refused() {
    let mut cache = Cache::new(tier(3), tier(10)).unwrap();

    assert_eq!(
        cache.admit(key(1), 301, 1.0, Context::new()),
        Err(CacheError::TooLarge {
            bytes: 301,
            capacity: 300
        })
    );
    assert_eq!(cache.free_slots(Tier::Device), 3);
}

#[test]
fn device_evictions_move_to_the_host_and_can_come_back() {
    let mut cache = Cache::new(tier(2), tier(4)).unwrap();
    cache.admit(key(1), 200, 1.0, Context::new()).unwrap();

    let admission = cache.admit(key(2), 200, 5.0, Context::new()).unwrap();

    assert_eq!(admission.movements.len(), 1);
    let demotion = &admission.movements[0];
    assert_eq!((demotion.key, demotion.from), (key(1), Tier::Device));
    let host = demotion.to.clone().unwrap();
    assert_eq!((host.tier, host.slots.len()), (Tier::Host, 2));
    assert_eq!(placed_on(&mut cache, &key(1)), Some(Tier::Host));

    let promotion = cache.promote(&key(1)).unwrap();

    assert_eq!(placed_on(&mut cache, &key(1)), Some(Tier::Device));
    assert_eq!(placed_on(&mut cache, &key(2)), Some(Tier::Host));
    let moved: Vec<_> = promotion
        .movements
        .iter()
        .map(|movement| movement.key)
        .collect();
    assert_eq!(moved, [key(2), key(1)]);
    assert_eq!(cache.free_slots(Tier::Host), 2);
}

#[test]
fn a_full_host_drops_its_least_valuable_entry() {
    let mut cache = Cache::new(tier(1), tier(1)).unwrap();
    cache.admit(key(1), 100, 1.0, Context::new()).unwrap();
    cache.admit(key(2), 100, 2.0, Context::new()).unwrap(); // key 1 moves to the host

    let admission = cache.admit(key(3), 100, 3.0, Context::new()).unwrap(); // key 2 moves to the host, dropping key 1

    let keys: Vec<_> = admission
        .movements
        .iter()
        .map(|movement| (movement.key, movement.to.is_some()))
        .collect();
    assert_eq!(keys, [(key(1), false), (key(2), true)]);
    assert!(cache.lookup(&key(1)).is_none());
}

#[test]
fn old_epochs_can_be_dropped_once_unread() {
    let mut cache = Cache::new(tier(10), tier(10)).unwrap();
    cache
        .admit(key_at(1, 0, 0), 300, 1.0, Context::new())
        .unwrap();
    cache
        .admit(key_at(2, 0, 1), 100, 1.0, Context::new())
        .unwrap();
    cache
        .admit(key_at(3, 1, 0), 100, 1.0, Context::new())
        .unwrap();

    let dropped = cache.drop_epochs_before(ModelId(0), TableId(0), Epoch(1));

    assert_eq!(
        dropped,
        [Movement {
            key: key_at(1, 0, 0),
            from: Tier::Device,
            to: None
        }]
    );
    assert_eq!(cache.free_slots(Tier::Device), 8);
    assert!(cache.lookup(&key_at(3, 1, 0)).is_some());
}

#[test]
fn freed_slots_are_reused() {
    let mut cache = Cache::new(tier(2), tier(0)).unwrap();
    let first = cache.admit(key(1), 100, 1.0, Context::new()).unwrap().slots;
    cache.admit(key(2), 100, 9.0, Context::new()).unwrap();

    let third = cache.admit(key(3), 100, 1.0, Context::new()).unwrap().slots;

    assert_eq!(third, first);
}

#[test]
fn only_known_entries_can_be_pinned_and_only_host_entries_promoted() {
    let mut cache = Cache::new(tier(1), tier(2)).unwrap();
    cache.admit(key(1), 100, 1.0, Context::new()).unwrap();
    cache.admit(key(2), 100, 2.0, Context::new()).unwrap();

    assert_eq!(cache.pin(&key(9)), Err(CacheError::Unknown(key(9))));
    assert_eq!(cache.pin(&key(1)), Ok(()));
    assert_eq!(cache.promote(&key(2)), Err(CacheError::OnDevice(key(2))));
}

#[test]
fn host_admissions_drop_host_entries_but_never_pinned_ones() {
    let mut cache = Cache::new(tier(1), tier(2)).unwrap();
    cache.admit_host(key(1), 100, 1.0, Context::new()).unwrap();
    cache.admit_host(key(2), 100, 1.0, Context::new()).unwrap();
    cache.pin(&key(1)).unwrap();

    let admitted = cache.admit_host(key(3), 100, 1.0, Context::new()).unwrap();
    cache.pin(&key(3)).unwrap();
    let full = cache.admit_host(key(4), 100, 1.0, Context::new());

    assert_eq!(placed_on(&mut cache, &key(1)), Some(Tier::Host));
    assert_eq!(placed_on(&mut cache, &key(3)), Some(Tier::Host));
    assert_eq!(placed_on(&mut cache, &key(2)), None);
    assert_eq!(
        admitted.movements,
        [Movement {
            key: key(2),
            from: Tier::Host,
            to: None
        }]
    );
    assert_eq!(full, Err(CacheError::NoRoom));
    assert_eq!(cache.free_slots(Tier::Device), 1);
}

#[test]
fn demotions_never_drop_pinned_host_entries() {
    let mut cache = Cache::new(tier(1), tier(1)).unwrap();
    cache.admit_host(key(1), 100, 1.0, Context::new()).unwrap();
    cache.pin(&key(1)).unwrap();
    cache.admit(key(2), 100, 1.0, Context::new()).unwrap();

    let admission = cache.admit(key(3), 100, 1.0, Context::new()).unwrap();

    assert_eq!(placed_on(&mut cache, &key(1)), Some(Tier::Host));
    assert_eq!(placed_on(&mut cache, &key(2)), None);
    assert_eq!(
        admission.movements,
        [Movement {
            key: key(2),
            from: Tier::Device,
            to: None
        }]
    );
}

#[test]
fn a_promotion_never_drops_the_entry_it_promotes() {
    let mut cache = Cache::new(tier(1), tier(1)).unwrap();
    cache.admit(key(1), 100, 1.0, Context::new()).unwrap();
    cache.admit(key(2), 100, 1.0, Context::new()).unwrap();

    let promoted = cache.promote(&key(1)).unwrap();

    assert_eq!(placed_on(&mut cache, &key(1)), Some(Tier::Device));
    assert_eq!(placed_on(&mut cache, &key(2)), None);
    assert_eq!(promoted.movements.len(), 2);
}

fn rows_key(unit: u32) -> CacheKey {
    CacheKey {
        kind: Kind::Rows,
        ..key(unit)
    }
}

#[test]
fn host_evictions_spill_contexts_to_disk() {
    let mut cache = Cache::new(tier(1), tier(1))
        .unwrap()
        .with_disk(tier(2))
        .unwrap();
    cache.admit(key(1), 100, 1.0, Context::new()).unwrap();
    cache.admit(key(2), 100, 2.0, Context::new()).unwrap();

    let admission = cache.admit(key(3), 100, 3.0, Context::new()).unwrap();

    assert_eq!(placed_on(&mut cache, &key(1)), Some(Tier::Disk));
    assert_eq!(placed_on(&mut cache, &key(2)), Some(Tier::Host));
    assert!(admission.movements.iter().any(|movement| {
        movement.key == key(1)
            && movement.from == Tier::Host
            && movement.to.as_ref().is_some_and(|to| to.tier == Tier::Disk)
    }));
}

#[test]
fn pages_are_dropped_rather_than_spilled() {
    let mut cache = Cache::new(tier(1), tier(1))
        .unwrap()
        .with_disk(tier(2))
        .unwrap();
    cache
        .admit_host(rows_key(1), 100, 1.0, Context::new())
        .unwrap();

    cache
        .admit_host(rows_key(2), 100, 1.0, Context::new())
        .unwrap();

    assert_eq!(placed_on(&mut cache, &rows_key(1)), None);
    assert_eq!(cache.free_slots(Tier::Disk), 2);
}

#[test]
fn spilled_contexts_promote_straight_to_the_device() {
    let mut cache = Cache::new(tier(1), tier(1))
        .unwrap()
        .with_disk(tier(2))
        .unwrap();
    cache.admit(key(1), 100, 1.0, Context::new()).unwrap();
    cache.admit(key(2), 100, 2.0, Context::new()).unwrap();
    cache.admit(key(3), 100, 3.0, Context::new()).unwrap();

    let promoted = cache.promote(&key(1)).unwrap();

    assert_eq!(placed_on(&mut cache, &key(1)), Some(Tier::Device));
    assert!(promoted.movements.iter().any(|movement| {
        movement.key == key(1)
            && movement.from == Tier::Disk
            && movement
                .to
                .as_ref()
                .is_some_and(|to| to.tier == Tier::Device)
    }));
    assert_eq!(cache.free_slots(Tier::Disk), 1);
}

#[test]
fn a_full_disk_drops_its_lowest_entries() {
    let mut cache = Cache::new(tier(1), tier(1))
        .unwrap()
        .with_disk(tier(1))
        .unwrap();
    for unit in 1..=4 {
        cache
            .admit(key(unit), 100, f64::from(unit), Context::new())
            .unwrap();
    }

    assert_eq!(placed_on(&mut cache, &key(4)), Some(Tier::Device));
    assert_eq!(placed_on(&mut cache, &key(3)), Some(Tier::Host));
    assert_eq!(placed_on(&mut cache, &key(2)), Some(Tier::Disk));
    assert_eq!(placed_on(&mut cache, &key(1)), None);
}

#[test]
fn pins_count_and_the_last_unpin_releases() {
    let mut cache = Cache::new(tier(2), tier(0)).unwrap();
    cache.admit(key(1), 200, 1.0, Context::new()).unwrap();
    cache.pin(&key(1)).unwrap();
    cache.pin(&key(1)).unwrap();

    cache.unpin(&key(1)).unwrap();
    let still_held = cache.admit(key(2), 100, 1.0, Context::new());
    cache.unpin(&key(1)).unwrap();
    let released = cache.admit(key(2), 100, 1.0, Context::new());

    assert_eq!(still_held, Err(CacheError::NoRoom));
    assert!(!cache.pinned(&key(1)));
    assert!(released.is_ok());
}

#[test]
fn entries_are_found_by_the_rows_they_hold() {
    let mut cache = Cache::new(tier(4), tier(4)).unwrap();
    let rows = |ids: &[u32]| -> Context { ids.iter().map(|&id| RowId(id)).collect() };
    cache.admit(key(1), 100, 1.0, rows(&[1, 2])).unwrap();
    cache.admit(key(2), 100, 1.0, rows(&[3])).unwrap();
    cache.admit_host(key(3), 100, 1.0, rows(&[2, 3])).unwrap();

    assert_eq!(cache.holding(TableId(0), &rows(&[2])), [key(1), key(3)]);
    assert_eq!(cache.holding(TableId(0), &rows(&[9])), []);
    assert_eq!(cache.holding(TableId(1), &rows(&[2])), []);
    assert_eq!(cache.rows(&key(2)), Some(&rows(&[3])));
    cache.remove(&key(1));
    assert_eq!(cache.holding(TableId(0), &rows(&[2])), [key(3)]);
    // Rows far apart sit in different blocks of the index.
    cache.admit(key(4), 100, 1.0, rows(&[2, 70_000])).unwrap();
    assert_eq!(cache.holding(TableId(0), &rows(&[70_000])), [key(4)]);
    let mut expected = vec![key(2), key(3), key(4)];
    expected.sort();
    assert_eq!(cache.holding(TableId(0), &rows(&[3, 70_000])), expected);
    cache.remove(&key(4));
    assert_eq!(cache.holding(TableId(0), &rows(&[70_000])), []);
}

/// A deterministic stream of numbers.
struct Random(u64);

impl Random {
    fn below(&mut self, limit: u64) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.0 >> 33) % limit
    }
}

#[test]
fn slots_are_never_lost_or_shared_under_random_operations() {
    let mut random = Random(5);
    let mut cache = Cache::new(tier(6), tier(5))
        .unwrap()
        .with_disk(tier(4))
        .unwrap();
    let mut keys: Vec<CacheKey> = Vec::new();
    let mut pinned: Vec<CacheKey> = Vec::new();
    for round in 0..500u32 {
        match random.below(5) {
            0 => {
                let key = if random.below(2) == 0 {
                    rows_key(round)
                } else {
                    key(round)
                };
                let bytes = random.below(3 * SLOT) + 1;
                let admitted = if random.below(3) == 0 {
                    cache.admit_host(key, bytes, 1.0, Context::new())
                } else {
                    cache.admit(key, bytes, 1.0, Context::new())
                };
                if admitted.is_ok() {
                    keys.push(key);
                }
            }
            1 if !keys.is_empty() => {
                let key = keys[random.below(keys.len() as u64) as usize];
                let _ = cache.promote(&key);
            }
            2 if !keys.is_empty() => {
                let key = keys.swap_remove(random.below(keys.len() as u64) as usize);
                pinned.retain(|other| *other != key);
                cache.remove(&key);
            }
            3 if !keys.is_empty() => {
                let key = keys[random.below(keys.len() as u64) as usize];
                if cache.pin(&key).is_ok() {
                    pinned.push(key);
                }
            }
            4 if !pinned.is_empty() => {
                let key = pinned.swap_remove(random.below(pinned.len() as u64) as usize);
                cache.unpin(&key).unwrap();
            }
            _ => {}
        }
        keys.retain(|key| cache.contains(key));
        pinned.retain(|key| cache.contains(key));
        for (tier, slots) in [(Tier::Device, 6), (Tier::Host, 5), (Tier::Disk, 4)] {
            let mut held = HashSet::new();
            let mut count = 0;
            for key in &keys {
                let placement = cache.lookup(key).unwrap();
                if placement.tier == tier {
                    count += placement.slots.len();
                    held.extend(placement.slots);
                }
            }
            assert_eq!(held.len(), count, "two entries share a slot");
            assert_eq!(
                count + cache.free_slots(tier) as usize,
                slots,
                "slots were lost"
            );
        }
        for key in &pinned {
            assert!(cache.pinned(key));
        }
    }
}
