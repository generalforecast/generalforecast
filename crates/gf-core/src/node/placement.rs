// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Placing a step's units and pages in the cache.

use super::*;

/// Entries a new step must not read: those a step in flight moves between
/// tiers, whose slots may not hold them yet, and stale ones a step in flight
/// still reads. A unit or page on such an entry is computed again, transiently.
#[derive(Clone, Copy)]
pub(super) struct Busy<'a> {
    pub(super) moving: &'a HashSet<CacheKey>,
    pub(super) doomed: &'a HashSet<CacheKey>,
    /// Entries a step in flight computes, which the worker holds only once
    /// it completes; a later step reads them, but copies from none.
    pub(super) encoding: &'a HashSet<CacheKey>,
}

impl Busy<'_> {
    pub(super) fn contains(&self, key: &CacheKey) -> bool {
        self.moving.contains(key) || self.doomed.contains(key)
    }

    /// Whether a step may copy rows from the entry.
    pub(super) fn source(&self, key: &CacheKey) -> bool {
        !self.contains(key) && !self.encoding.contains(key)
    }
}

/// Adds units of a table to a step, placing each, and the pages holding the
/// rows of those to encode that `placed` does not hold already.
#[allow(clippy::too_many_arguments)]
pub(super) fn place_units(
    cache: &mut Cache,
    busy: Busy,
    id: TableId,
    state: &TableState,
    snapshot: &Snapshot,
    units: Vec<crate::planner::Unit>,
    keys: Vec<CacheKey>,
    placed: &BTreeMap<PageId, StepPage>,
    fuse: bool,
    step: &mut Step,
    parts: &mut PlanParts,
) -> Result<(), NodeError> {
    let since = parts.start();
    // The rows of units to encode, whose pages the step needs.
    let mut encoding = Context::new();
    for (unit, key) in units.into_iter().zip(keys) {
        let mut rows: Vec<RowId> = unit.rows.iter().collect();
        let bytes = state.spec.sizes.bytes(rows.len(), cache.slot_bytes());
        let held = unit.rows.clone();
        let placement = if fuse {
            Placed::Fuse
        } else {
            place(cache, busy, key, bytes, unit.rows, &mut step.movements)?
        };
        // A unit fit on its own rows carries the values the model reads of
        // them, and reads no pages.
        let mut fit = None;
        if placement.action() != Action::Reuse {
            // The model reads a context's rows by key, so their order does
            // not depend on the order they were written in.
            rows = by_key(&state.table, &rows);
            if fuse || state.spec.summary == SummaryScope::Context {
                fit = Some(state.inputs(&rows)?);
            } else {
                encoding |= &held;
            }
        }
        step.units.push(Unit {
            table: id,
            key,
            rows,
            placed: placement,
            fit,
        });
    }
    parts.add("units", since);
    let since = parts.start();
    let missing: Vec<PageId> = state
        .index
        .pages_holding(&encoding)
        .into_iter()
        .filter(|page| !placed.contains_key(page))
        .collect();
    let pages = place_pages(cache, busy, id, state, snapshot, &missing, false, step);
    parts.add("pages", since);
    pages
}

/// Places pages of a table in the host tier and adds them to the step, pinning kept ones.
///
/// Cached pages are pinned before any new page is admitted, so admitting one
/// cannot evict another this step reads. A fit computes its pages, so they
/// carry no rows to embed.
#[allow(clippy::too_many_arguments)]
pub(super) fn place_pages(
    cache: &mut Cache,
    busy: Busy,
    table: TableId,
    state: &TableState,
    snapshot: &Snapshot,
    pages: &[PageId],
    fit: bool,
    step: &mut Step,
) -> Result<(), NodeError> {
    let mut entries = Vec::with_capacity(pages.len());
    for &page in pages {
        let context = state.index.rows(page).unwrap_or_default();
        let rows: Vec<RowId> = context
            .iter()
            .filter(|&row| snapshot.version(row).is_some())
            .collect();
        let key = CacheKey {
            model: state.spec.model,
            table,
            epoch: state.epoch,
            kind: Kind::Rows,
            rows: snapshot.fingerprint(&context),
        };
        let cached = cache.lookup(&key);
        if cached.is_some() && !busy.contains(&key) {
            cache.pin(&key)?;
        }
        // Sources are pinned with the cached pages, before any admission.
        let copies = if cached.is_none() && !fit {
            page_sources(cache, busy, table, state.epoch, &rows)?
        } else {
            Vec::new()
        };
        entries.push((key, rows, cached, copies));
    }
    for (key, rows, cached, copies) in entries {
        let placed = if let Some(placement) = cached {
            if busy.contains(&key) {
                Placed::Transient
            } else {
                Placed::Reuse(placement.slots)
            }
        } else {
            let bytes = state.spec.sizes.bytes_per_representation * rows.len() as u64;
            let cost = EMBED_COST_PER_CELL * (rows.len() * state.spec.model_features()) as f64;
            let held = rows.iter().copied().collect();
            match cache.admit_host(key, bytes, cost, held) {
                Ok(admission) => {
                    step.movements.extend(admission.movements);
                    cache.pin(&key)?;
                    Placed::Encode(admission.slots)
                }
                Err(CacheError::NoRoom | CacheError::TooLarge { .. }) => Placed::Transient,
                Err(error) => return Err(error.into()),
            }
        };
        let embed = (placed.action() != Action::Reuse && !fit)
            .then(|| {
                let copied: HashSet<usize> = copies
                    .iter()
                    .flat_map(|(_, positions)| positions.iter().map(|&(_, to)| to))
                    .collect();
                let embedded: Vec<RowId> = rows
                    .iter()
                    .enumerate()
                    .filter(|(at, _)| !copied.contains(at))
                    .map(|(_, &row)| row)
                    .collect();
                state.inputs(&embedded)
            })
            .transpose()?;
        step.pages.push(Page {
            table,
            key,
            rows,
            placed,
            embed,
            copies,
        });
    }
    Ok(())
}

/// The cached entries of this epoch holding rows of a page to compute that
/// were not written since, pinned, with the positions to copy from each.
///
/// Each row is copied from the first source holding it, in key order.
pub(super) fn page_sources(
    cache: &mut Cache,
    busy: Busy,
    table: TableId,
    epoch: Epoch,
    rows: &[RowId],
) -> Result<Copies, NodeError> {
    let wanted: Context = rows.iter().copied().collect();
    let mut copies = Vec::new();
    let mut covered = HashSet::new();
    for source in cache.holding(table, &wanted) {
        if source.kind != Kind::Rows || source.epoch != epoch || !busy.source(&source) {
            continue;
        }
        let (Some(held), Some(stale)) = (cache.rows(&source), cache.stale(&source)) else {
            continue;
        };
        let from: HashMap<RowId, usize> =
            held.iter().enumerate().map(|(at, row)| (row, at)).collect();
        let positions: Vec<(usize, usize)> = rows
            .iter()
            .enumerate()
            .filter(|&(to, row)| !covered.contains(&to) && !stale.contains(*row))
            .filter_map(|(to, row)| from.get(row).map(|&at| (at, to)))
            .collect();
        if positions.is_empty() {
            continue;
        }
        covered.extend(positions.iter().map(|&(_, to)| to));
        cache.pin(&source)?;
        copies.push((source, positions));
    }
    Ok(copies)
}

/// A series unit placed as an extension: where, from which entry, by how many
/// steps, and the patches its entry will hold.
pub(super) struct Extension {
    pub(super) placed: Placed,
    pub(super) source: CacheKey,
    pub(super) grown: usize,
    pub(super) patches: usize,
}

/// The steps a unit's grid grows `prev`'s by and the patches its extension
/// holds, the source's and the new ones: None unless the cutoff advanced by
/// whole patches, the grid grew by as many steps and the grown context still
/// fits the model's.
pub(super) fn extension_size(
    spec: &SeriesSpec,
    unit: &forecasting::PlannedUnit,
    prev: &Recent,
) -> Option<(usize, usize)> {
    let step = spec.step.get() as i64;
    let advanced = unit.cutoff.checked_sub(prev.cutoff).unwrap_or(0);
    if advanced <= 0 || advanced % step != 0 {
        return None;
    }
    let grown = usize::try_from(advanced / step).unwrap_or(usize::MAX);
    let patch = spec.patch.get();
    if grown % patch != 0 || unit.steps != prev.steps + grown {
        return None;
    }
    let patches = prev.patches + grown / patch;
    (patches * patch <= spec.context.get()).then_some((grown, patches))
}

/// Bytes an extension of `patches` patches takes: it is stored padded to the
/// bucket of its patches, as an encoding of as many is, so the programs
/// reading it serve every count in the bucket.
pub(super) fn extension_bytes(
    spec: &SeriesSpec,
    sizes: &EntrySizes,
    features: usize,
    unit: &forecasting::PlannedUnit,
    patches: usize,
    slot_bytes: u64,
) -> u64 {
    let bucket = sizes.series_patches(spec, patches * spec.patch.get());
    sizes.series_bytes_of(spec, features, unit.members.len(), bucket, slot_bytes)
}

/// Places a series unit as an extension of its group's latest kept entry,
/// when the cutoff advanced by whole patches, the grid grew by as many steps
/// and the rows before the earlier cutoff are the entry's: the new entry is
/// admitted, sized for the patches the source holds plus the new ones, and
/// both are pinned for the step, the source before room is made so that
/// making it cannot move the source off the device. None when any of that
/// fails, the grown context would pass the model's, or the device has no
/// room, so the unit is placed as any other.
#[allow(clippy::too_many_arguments)]
pub(super) fn extend(
    cache: &mut Cache,
    busy: Busy,
    snapshot: &Snapshot,
    spec: &SeriesSpec,
    sizes: &EntrySizes,
    features: usize,
    unit: &forecasting::PlannedUnit,
    key: CacheKey,
    prev: Recent,
    slot_bytes: u64,
    movements: &mut Vec<Movement>,
) -> Result<Option<Extension>, NodeError> {
    let Some((grown, patches)) = extension_size(spec, unit, &prev) else {
        return Ok(None);
    };
    let bytes = extension_bytes(spec, sizes, features, unit, patches, slot_bytes);
    if busy.contains(&prev.key) || busy.contains(&key) {
        return Ok(None);
    }
    let Some(placement) = cache.lookup(&prev.key) else {
        return Ok(None);
    };
    if placement.tier != Tier::Device {
        return Ok(None);
    }
    let prefix = forecasting::prefix_rows(unit, prev.steps);
    let print = forecasting::unit_key(
        snapshot.fingerprint(&prefix),
        prev.cutoff,
        prev.steps,
        &unit.members,
        unit.target,
    );
    if print != prev.key.rows {
        return Ok(None);
    }
    let cost = (unit.rows().len() as f64).powi(2);
    cache.pin(&prev.key)?;
    match cache.admit(key, bytes, cost, unit.rows()) {
        Ok(admission) => {
            movements.extend(admission.movements);
            cache.pin(&key)?;
            Ok(Some(Extension {
                placed: Placed::Extend {
                    source: placement.slots,
                    slots: admission.slots,
                },
                source: prev.key,
                grown,
                patches,
            }))
        }
        Err(error) => {
            let _ = cache.unpin(&prev.key);
            match error {
                CacheError::NoRoom | CacheError::TooLarge { .. } => Ok(None),
                error => Err(error.into()),
            }
        }
    }
}

/// Finds or makes room for a unit's entry on the device, pinning kept entries.
pub(super) fn place(
    cache: &mut Cache,
    busy: Busy,
    key: CacheKey,
    bytes: u64,
    rows: Context,
    movements: &mut Vec<Movement>,
) -> Result<Placed, NodeError> {
    if busy.contains(&key) {
        return Ok(Placed::Transient);
    }
    let placed = if let Some(placement) = cache.lookup(&key) {
        if placement.tier == Tier::Device {
            Ok(Placed::Reuse(placement.slots))
        } else {
            cache.promote(&key).map(|admission| {
                movements.extend(admission.movements);
                Placed::Reuse(admission.slots)
            })
        }
    } else {
        let cost = (rows.len() as f64).powi(2);
        cache.admit(key, bytes, cost, rows).map(|admission| {
            movements.extend(admission.movements);
            Placed::Encode(admission.slots)
        })
    };
    match placed {
        Ok(placed) => {
            cache.pin(&key)?;
            Ok(placed)
        }
        Err(CacheError::NoRoom | CacheError::TooLarge { .. }) => Ok(Placed::Transient),
        Err(error) => Err(error.into()),
    }
}
