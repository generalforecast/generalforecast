// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! One round of rebalancing on measured load.

use crate::assignment::{Assignment, Slice};
use crate::key::Key;
use crate::policy::{Mode, Policy};

/// What a slice's holders measured since the last round.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SliceLoad {
    /// GPU seconds a second, summed over its holders.
    pub load: f64,
    /// Rows it holds, the cost of moving it.
    pub rows: f64,
    /// The key splitting its load in half, or None if it cannot split.
    pub median: Option<Key>,
}

/// A round's new assignment and what it took.
#[derive(Debug, Clone, PartialEq)]
pub struct Rebalance {
    pub assignment: Assignment,
    /// Busiest GPU's load over the mean, after the round.
    pub imbalance: f64,
    pub moved_rows: f64,
    pub moves: usize,
    pub splits: usize,
    pub merges: usize,
    pub copies: usize,
}

struct Work {
    slice: Slice,
    load: SliceLoad,
    end: Option<Key>,
}

fn gpu_loads(work: &[Work], gpus: usize) -> Vec<f64> {
    let mut g = vec![0.0; gpus];
    for w in work {
        let share = w.load.load / w.slice.holders.len() as f64;
        for &h in &w.slice.holders {
            g[h] += share;
        }
    }
    g
}

/// The busiest GPU's load over the mean of `loads`, one per GPU; 1 when
/// nothing is loaded.
pub fn imbalance(g: &[f64]) -> f64 {
    let mean = g.iter().sum::<f64>() / g.len() as f64;
    if mean <= 0.0 {
        1.0
    } else {
        g.iter().cloned().fold(0.0, f64::max) / mean
    }
}

fn argmax(g: &[f64]) -> usize {
    (0..g.len())
        .max_by(|&a, &b| g[a].total_cmp(&g[b]))
        .unwrap_or(0)
}

fn argmin(g: &[f64]) -> usize {
    (0..g.len())
        .min_by(|&a, &b| g[a].total_cmp(&g[b]))
        .unwrap_or(0)
}

/// One round over `current` with `loads` aligned to its slices.
///
/// With `min_slices_per_gpu` at `usize::MAX`, `max_copies` at 1 and no load
/// naming a median, a round neither merges, splits nor copies: its slices are
/// `current`'s, in order and with the same starts, each held by one GPU, and
/// only who holds them changes. A caller whose slices are whole units, such as
/// one table each, may zip them with what they stand for.
///
/// # Panics
///
/// If `current` fails `Assignment::check` for `gpus` and the policy's mode,
/// or `loads` is not one per slice.
pub fn rebalance(
    current: &Assignment,
    loads: &[SliceLoad],
    gpus: usize,
    policy: &Policy,
) -> Rebalance {
    if let Err(fault) = current.check(gpus, policy.mode == Mode::Contiguous) {
        panic!("rebalancing {fault}");
    }
    assert_eq!(current.slices.len(), loads.len(), "one load per slice");
    let mut work: Vec<Work> = current
        .slices
        .iter()
        .zip(loads)
        .enumerate()
        .map(|(i, (s, l))| Work {
            slice: s.clone(),
            load: *l,
            end: current.end(i),
        })
        .collect();
    let total_rows: f64 = loads.iter().map(|l| l.rows).sum::<f64>().max(1.0);
    let budget = policy.churn_budget * total_rows;
    let mut out = Rebalance {
        assignment: current.clone(),
        imbalance: 1.0,
        moved_rows: 0.0,
        moves: 0,
        splits: 0,
        merges: 0,
        copies: 0,
    };
    match policy.mode {
        Mode::Free => free(&mut work, gpus, policy, budget, total_rows, &mut out),
        Mode::Contiguous => contiguous(&mut work, gpus, policy, budget, &mut out),
    }
    split(&mut work, gpus, policy, &mut out);
    let g = gpu_loads(&work, gpus);
    out.imbalance = imbalance(&g);
    let slices: Vec<Slice> = work.into_iter().map(|w| w.slice).collect();
    let changed = slices != current.slices;
    out.assignment = Assignment {
        generation: current.generation + u64::from(changed),
        slices,
    };
    out
}

fn free(
    work: &mut Vec<Work>,
    gpus: usize,
    p: &Policy,
    budget: f64,
    total_rows: f64,
    out: &mut Rebalance,
) {
    // Copies: a slice that cannot split and outgrows a share of a GPU spreads over more GPUs.
    let mean_gpu = work.iter().map(|w| w.load.load).sum::<f64>() / gpus as f64;
    for i in 0..work.len() {
        if work[i].load.median.is_some() || mean_gpu <= 0.0 {
            continue;
        }
        // A policy of no copies at all holds each slice once, as 1 does.
        let most = p.max_copies.clamp(1, gpus);
        let want = ((work[i].load.load / (p.copy_above * mean_gpu)).ceil() as usize).clamp(1, most);
        while work[i].slice.holders.len() < want {
            let mut g = gpu_loads(work, gpus);
            for &h in &work[i].slice.holders {
                g[h] = f64::INFINITY;
            }
            let to = argmin(&g);
            work[i].slice.holders.push(to);
            out.moved_rows += work[i].load.rows;
            out.copies += 1;
        }
        work[i].slice.holders.truncate(want);
    }

    // Merges: adjacent cold slices join while there are more than enough slices.
    let mean_slice = work.iter().map(|w| w.load.load).sum::<f64>() / work.len() as f64;
    let mut merged_rows = 0.0;
    let mut i = 0;
    while i + 1 < work.len() && work.len() > p.min_slices_per_gpu.saturating_mul(gpus) {
        let (a, b) = (&work[i], &work[i + 1]);
        let cold = a.load.load + b.load.load < mean_slice;
        let single = a.slice.holders.len() == 1 && b.slice.holders.len() == 1;
        let same = a.slice.holders == b.slice.holders;
        let affordable = merged_rows + b.load.rows <= p.merge_budget * total_rows;
        if cold && single && (same || affordable) {
            if !same {
                merged_rows += b.load.rows;
            }
            let b = work.remove(i + 1);
            let a = &mut work[i];
            a.load = SliceLoad {
                load: a.load.load + b.load.load,
                rows: a.load.rows + b.load.rows,
                median: None,
            };
            a.end = b.end;
            out.merges += 1;
        } else {
            i += 1;
        }
    }
    out.moved_rows += merged_rows;

    // Moves: from the busiest GPU to the idlest, best reduction in imbalance
    // per row first. A round may always make one move, so slices larger
    // than the budget still move.
    let mut g = gpu_loads(work, gpus);
    while (out.moves == 0 || out.moved_rows < budget) && imbalance(&g) >= p.imbalance_floor {
        let (hot, cold) = (argmax(&g), argmin(&g));
        let mut best: Option<(usize, f64, f64)> = None;
        for (i, w) in work.iter().enumerate() {
            if !w.slice.holders.contains(&hot) || w.slice.holders.contains(&cold) {
                continue;
            }
            let share = w.load.load / w.slice.holders.len() as f64;
            if share <= 0.0
                || g[cold] + share >= g[hot]
                || (out.moves > 0 && out.moved_rows + w.load.rows > budget)
            {
                continue;
            }
            let gain = g[hot] - (g[hot] - share).max(g[cold] + share);
            let weight = gain / (w.load.rows + 1.0);
            if best.is_none_or(|(_, bw, _)| weight > bw) {
                best = Some((i, weight, share));
            }
        }
        let Some((i, _, share)) = best else { break };
        let holders = &mut work[i].slice.holders;
        let at = holders
            .iter()
            .position(|&h| h == hot)
            .expect("hot GPU holds the slice");
        holders[at] = cold;
        g[hot] -= share;
        g[cold] += share;
        out.moved_rows += work[i].load.rows;
        out.moves += 1;
    }
}

fn contiguous(work: &mut [Work], gpus: usize, p: &Policy, budget: f64, out: &mut Rebalance) {
    // Each GPU g holds slices [cut g, cut g+1); cuts move toward equal-load
    // quantiles, scaled together until the rows they move fit the budget.
    let n = work.len();
    let owners: Vec<usize> = work.iter().map(|w| w.slice.holders[0]).collect();
    let mut cuts: Vec<usize> = (0..=gpus)
        .map(|g| owners.partition_point(|&o| o < g))
        .collect();
    cuts[gpus] = n;
    let load_prefix: Vec<f64> = std::iter::once(0.0)
        .chain(work.iter().scan(0.0, |s, w| {
            *s += w.load.load;
            Some(*s)
        }))
        .collect();
    let rows_prefix: Vec<f64> = std::iter::once(0.0)
        .chain(work.iter().scan(0.0, |s, w| {
            *s += w.load.rows;
            Some(*s)
        }))
        .collect();
    let g: Vec<f64> = (0..gpus)
        .map(|k| load_prefix[cuts[k + 1]] - load_prefix[cuts[k]])
        .collect();
    if imbalance(&g) >= p.imbalance_floor && n > 0 {
        let total = load_prefix[n];
        let target: Vec<usize> = (1..gpus)
            .map(|k| {
                load_prefix
                    .partition_point(|&c| c < total * k as f64 / gpus as f64)
                    .min(n)
            })
            .collect();
        let at = |alpha: f64| -> (Vec<usize>, f64) {
            let mut new = vec![0usize; gpus + 1];
            new[gpus] = n;
            for k in 1..gpus {
                let c = cuts[k] as f64 + alpha * (target[k - 1] as f64 - cuts[k] as f64);
                new[k] = (c.round() as usize).clamp(new[k - 1], n);
            }
            let moved = (1..gpus)
                .map(|k| (rows_prefix[new[k]] - rows_prefix[cuts[k]]).abs())
                .sum();
            (new, moved)
        };
        let alpha = if at(1.0).1 <= budget {
            1.0
        } else {
            let (mut lo, mut hi) = (0.0, 1.0);
            for _ in 0..30 {
                let mid = (lo + hi) / 2.0;
                if at(mid).1 <= budget {
                    lo = mid
                } else {
                    hi = mid
                }
            }
            lo
        };
        let (new, moved) = at(alpha);
        out.moved_rows += moved;
        cuts = new;
    }
    for k in 0..gpus {
        for w in &mut work[cuts[k]..cuts[k + 1]] {
            if w.slice.holders != [k] {
                out.moves += 1;
                w.slice.holders = vec![k];
            }
        }
    }
}

fn split(work: &mut Vec<Work>, gpus: usize, p: &Policy, out: &mut Rebalance) {
    // Hot slices split at their load median, keeping their holders, for later rounds to move.
    let mean_slice = work.iter().map(|w| w.load.load).sum::<f64>() / work.len().max(1) as f64;
    let mut i = 0;
    while i < work.len() {
        let w = &work[i];
        let inside = w
            .load
            .median
            .filter(|&m| m > w.slice.start && w.end.is_none_or(|e| m < e));
        // A slice without load is never hot, however cold the rest are.
        if let Some(m) = inside
            && w.load.load > 0.0
            && w.load.load >= p.hot_split * mean_slice
            && work.len() < p.max_slices_per_gpu.saturating_mul(gpus)
        {
            let half = SliceLoad {
                load: w.load.load / 2.0,
                rows: w.load.rows / 2.0,
                median: None,
            };
            let right = Work {
                slice: Slice {
                    start: m,
                    holders: w.slice.holders.clone(),
                },
                load: half,
                end: w.end,
            };
            work[i].load = half;
            work[i].end = Some(m);
            work.insert(i + 1, right);
            out.splits += 1;
            i += 2;
        } else {
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loads(values: &[f64]) -> Vec<SliceLoad> {
        values
            .iter()
            .map(|&l| SliceLoad {
                load: l,
                rows: 100.0,
                median: None,
            })
            .collect()
    }

    #[test]
    fn free_moves_load_off_the_busiest_gpu_within_budget() {
        let a = Assignment::even(16, 4, false);
        let mut values = vec![1.0; 16];
        for (i, s) in a.slices.iter().enumerate() {
            if s.holders[0] == 0 {
                values[i] = 5.0;
            }
        }
        let p = Policy {
            min_slices_per_gpu: 1,
            churn_budget: 0.2,
            max_copies: 1,
            ..Policy::default()
        };
        let before = imbalance(&gpu_loads(
            &a.slices
                .iter()
                .zip(loads(&values))
                .map(|(s, l)| Work {
                    slice: s.clone(),
                    load: l,
                    end: None,
                })
                .collect::<Vec<_>>(),
            4,
        ));
        let r = rebalance(&a, &loads(&values), 4, &p);
        assert!(r.imbalance < before, "{} !< {before}", r.imbalance);
        assert!(r.moved_rows <= 0.2 * 1600.0);
        assert!(r.moves > 0);
        assert_eq!(r.assignment.generation, 1);
    }

    #[test]
    fn balanced_load_moves_nothing() {
        let a = Assignment::even(8, 4, false);
        let r = rebalance(
            &a,
            &loads(&[1.0; 8]),
            4,
            &Policy {
                min_slices_per_gpu: 2,
                max_copies: 1,
                ..Policy::default()
            },
        );
        assert_eq!(r.moves + r.splits + r.merges + r.copies, 0);
        assert_eq!(r.assignment, a);
    }

    #[test]
    fn contiguous_keeps_ranges_and_balances() {
        let a = Assignment::even(32, 4, true);
        let values: Vec<f64> = (0..32).map(|i| if i < 8 { 10.0 } else { 1.0 }).collect();
        let p = Policy {
            mode: Mode::Contiguous,
            churn_budget: 1.0,
            ..Policy::default()
        };
        let r = rebalance(&a, &loads(&values), 4, &p);
        let owners: Vec<usize> = r.assignment.slices.iter().map(|s| s.holders[0]).collect();
        assert!(owners.windows(2).all(|w| w[0] <= w[1]), "{owners:?}");
        assert!(r.imbalance < 1.5, "{}", r.imbalance);
    }

    #[test]
    fn contiguous_respects_the_budget() {
        let a = Assignment::even(32, 4, true);
        let values: Vec<f64> = (0..32).map(|i| if i < 8 { 10.0 } else { 1.0 }).collect();
        let p = Policy {
            mode: Mode::Contiguous,
            churn_budget: 0.05,
            ..Policy::default()
        };
        let r = rebalance(&a, &loads(&values), 4, &p);
        assert!(r.moved_rows <= 0.05 * 3200.0 + 1e-9);
    }

    #[test]
    #[should_panic(expected = "slice 1 breaks the run of GPU 0")]
    fn contiguous_mode_refuses_slices_out_of_run() {
        let mut a = Assignment::even(4, 2, true);
        a.slices[0].holders = vec![1];
        let p = Policy {
            mode: Mode::Contiguous,
            ..Policy::default()
        };
        rebalance(&a, &loads(&[1.0; 4]), 2, &p);
    }

    #[test]
    #[should_panic(expected = "slice 2 is held by GPU 2 of 2")]
    fn a_holder_beyond_the_gpus_is_refused() {
        let a = Assignment::even(4, 4, false);
        rebalance(&a, &loads(&[1.0; 4]), 2, &Policy::default());
    }

    #[test]
    fn hot_slices_split_at_their_median() {
        let a = Assignment::even(4, 2, false);
        let mut l = loads(&[1.0, 1.0, 1.0, 9.0]);
        let mid = a.slices[3].start + (u64::MAX - a.slices[3].start) / 2;
        l[3].median = Some(mid);
        let r = rebalance(
            &a,
            &l,
            2,
            &Policy {
                min_slices_per_gpu: 1,
                imbalance_floor: 100.0,
                ..Policy::default()
            },
        );
        assert_eq!(r.splits, 1);
        assert_eq!(r.assignment.slices.len(), 5);
        assert_eq!(r.assignment.slices[4].start, mid);
        assert_eq!(
            r.assignment.slices[4].holders,
            r.assignment.slices[3].holders
        );
    }

    #[test]
    fn an_idle_table_splits_nothing() {
        let a = Assignment::even(4, 2, false);
        let l: Vec<SliceLoad> = (0..4)
            .map(|i| SliceLoad {
                load: 0.0,
                rows: 10.0,
                median: Some(a.slices[i].start + 1),
            })
            .collect();
        let p = Policy {
            min_slices_per_gpu: 1,
            ..Policy::default()
        };
        let r = rebalance(&a, &l, 2, &p);
        assert_eq!(r.splits, 0);
        assert_eq!(r.assignment, a);
    }

    #[test]
    fn no_copies_holds_each_slice_once() {
        let a = Assignment::even(4, 4, false);
        let p = Policy {
            min_slices_per_gpu: 1,
            max_copies: 0,
            ..Policy::default()
        };
        let r = rebalance(&a, &loads(&[1.0, 1.0, 1.0, 9.0]), 4, &p);
        assert_eq!(r.copies, 0);
        assert!(r.assignment.slices.iter().all(|s| s.holders.len() == 1));
    }

    #[test]
    fn a_moves_only_policy_keeps_every_slice() {
        let a = Assignment {
            generation: 0,
            slices: (0..12)
                .map(|i| Slice {
                    start: i,
                    holders: vec![(i % 3) as usize],
                })
                .collect(),
        };
        let l: Vec<SliceLoad> = (0..12)
            .map(|i| SliceLoad {
                load: if i % 3 == 0 { 8.0 } else { 0.5 },
                rows: 1.0 + i as f64,
                median: None,
            })
            .collect();
        let p = Policy {
            min_slices_per_gpu: usize::MAX,
            max_copies: 1,
            churn_budget: 1.0,
            ..Policy::default()
        };
        let r = rebalance(&a, &l, 3, &p);
        assert!(r.moves > 0);
        let starts = |x: &Assignment| x.slices.iter().map(|s| s.start).collect::<Vec<_>>();
        assert_eq!(starts(&r.assignment), starts(&a));
        assert!(r.assignment.slices.iter().all(|s| s.holders.len() == 1));
        assert_eq!(r.merges + r.splits + r.copies, 0);
    }

    #[test]
    fn imbalance_is_the_busiest_over_the_mean() {
        assert_eq!(imbalance(&[1.0, 3.0]), 1.5);
        assert_eq!(imbalance(&[0.0, 0.0]), 1.0);
    }

    #[test]
    fn an_unsplittable_hot_slice_gains_copies() {
        let a = Assignment::even(4, 4, false);
        let r = rebalance(
            &a,
            &loads(&[1.0, 1.0, 1.0, 9.0]),
            4,
            &Policy {
                min_slices_per_gpu: 1,
                ..Policy::default()
            },
        );
        assert!(r.copies >= 1);
        assert!(r.assignment.slices[3].holders.len() > 1);
    }
}
