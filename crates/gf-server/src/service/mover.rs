// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Placing tables by load: the mover's rounds, which move tables between this
//! server's shards one at a time.

use super::frames::invalid;
use super::*;

/// What a server that places tables by load keeps between rounds.
pub struct Mover {
    pub(super) tables: PathBuf,
    pub(super) keys: crate::wal::Keys,
    pub(super) churn_budget: f64,
    pub(super) imbalance_floor: f64,
    /// Each shard's worker seconds a second over the last round.
    pub(super) shard_loads: Mutex<Vec<f64>>,
    /// Each table's worker seconds at the last round.
    pub(super) seen: Mutex<HashMap<String, f64>>,
    /// The shard of each table placed since the last round.
    pub(super) fresh: Mutex<HashMap<String, usize>>,
    /// When the last round ran, or the server started.
    pub(super) last_round: Mutex<Instant>,
}

impl Mover {
    /// Drops what the mover remembers of a table.
    pub(super) fn forget(&self, table: &str) {
        self.seen
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(table);
        self.fresh
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(table);
    }
}

/// What a round of moving tables did.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Round {
    pub moved: Vec<(String, usize, usize)>,
    pub imbalance_before: f64,
    pub imbalance_after: f64,
    pub shard_loads: Vec<f64>,
    pub seconds: f64,
}

impl Service {
    /// Moves a table to shard `to`: its calls wait, its shard finishes what
    /// it holds and lets go of the table, and `to` restores it from its log.
    ///
    /// # Errors
    ///
    /// If the server does not move tables, the table or shard does not
    /// exist, or either shard refuses.
    pub async fn move_table(&self, table: &TableName, to: usize) -> Result<f64, Status> {
        let mover = self
            .mover
            .clone()
            .ok_or_else(|| invalid("tables move only with placement.assign = \"load\""))?;
        if to >= self.shards.len() {
            return Err(invalid(format!(
                "the server has {} shards",
                self.shards.len()
            )));
        }
        let started = Instant::now();
        let _moving = self.locked(table, |lock| lock.write_owned()).await?;
        let from = self.placement(table).await?;
        if from.shard == to {
            return Ok(0.0);
        }
        if from.series.is_some() {
            return Err(invalid("a table of series does not move yet"));
        }
        let name = table.clone();
        self.ask(from.shard, |reply| Command::Handoff { table: name, reply })
            .await?;
        let path = mover.tables.join(format!("{table}.log"));
        let restored = async {
            let replay = crate::setup::open_log(&path, &mover.keys)
                .map_err(|error| Status::internal(format!("opening the log: {error}")))?
                .ok_or_else(|| Status::internal("the log did not open"))?;
            self.ask(to, |reply| Command::Restore {
                replay: Box::new(replay),
                reply,
            })
            .await
        };
        let (shard, levels) = match restored.await {
            Ok(levels) => (to, levels),
            Err(error) => {
                // Serve it where it was rather than nowhere.
                tracing::error!(table = table.as_str(), to, "moving a table: {error}");
                let back = async {
                    let replay = crate::setup::open_log(&path, &mover.keys)
                        .map_err(|error| Status::internal(format!("reopening the log: {error}")))?
                        .ok_or_else(|| Status::internal("the log did not reopen"))?;
                    self.ask(from.shard, |reply| Command::Restore {
                        replay: Box::new(replay),
                        reply,
                    })
                    .await
                };
                match back.await {
                    Ok(levels) => (from.shard, levels),
                    Err(error) => {
                        // No shard serves it now. Its log stays, so a restart
                        // brings it back, and a creation of its name fails on
                        // the log rather than replacing it.
                        tracing::error!(
                            table = table.as_str(),
                            from = from.shard,
                            "a moved table is out of service until the server restarts: {error}"
                        );
                        let mut placement = self.placement.write().await;
                        placement.remove(table);
                        self.metrics.tables.set(placement.len() as u64);
                        drop(placement);
                        self.metrics.forget(table);
                        mover.forget(table);
                        return Err(Status::internal(format!(
                            "table {table:?} could not move and is out of service until the \
                             server restarts: {error}"
                        )));
                    }
                }
            }
        };
        if let Some(placed) = self.placement.write().await.get_mut(table) {
            placed.shard = shard;
            placed.levels = levels.map(Arc::new);
        }
        let seconds = started.elapsed().as_secs_f64();
        self.metrics.moves.add(u64::from(shard == to));
        self.metrics.move_seconds.observe(started.elapsed());
        tracing::info!(
            table = table.as_str(),
            from = from.shard,
            to = shard,
            seconds,
            "moved a table"
        );
        if shard == to {
            Ok(seconds)
        } else {
            Err(Status::internal("the table could not move and stayed"))
        }
    }

    /// One round: each table's worker seconds since the last round are its
    /// load, `gf-assign` chooses moves within the churn budget, and they run
    /// one at a time.
    ///
    /// # Errors
    ///
    /// If the server does not move tables.
    pub async fn rebalance_round(&self) -> Result<Round, Status> {
        let mover = self
            .mover
            .clone()
            .ok_or_else(|| invalid("tables move only with placement.assign = \"load\""))?;
        let started = Instant::now();
        let elapsed = {
            let mut last = mover
                .last_round
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            std::mem::replace(&mut *last, started)
                .elapsed()
                .as_secs_f64()
        };
        let gpus = self.shards.len();
        let mut tables: Vec<(TableName, usize)> = self
            .placement
            .read()
            .await
            .iter()
            .filter(|(_, placed)| placed.series.is_none())
            .map(|(name, placed)| (name.clone(), placed.shard))
            .collect();
        tables.sort_by(|a, b| a.0.cmp(&b.0));
        // A move costs the rows the destination replays, which the shard
        // holding the table counts; a table that cannot say sits this round out.
        let mut placed = Vec::with_capacity(tables.len());
        for (name, shard) in tables {
            let table = name.clone();
            match self
                .ask(shard, |reply| Command::Describe { table, reply })
                .await
            {
                Ok(described) => placed.push((name, shard, described.rows)),
                Err(error) => tracing::warn!(table = name.as_str(), "not measured: {error}"),
            }
        }
        let mut loads = Vec::with_capacity(placed.len());
        {
            let mut seen = mover.seen.lock().unwrap_or_else(PoisonError::into_inner);
            for (name, _, rows) in &placed {
                let now = self
                    .metrics
                    .existing_table(name)
                    .map_or(0.0, |counts| counts.gpu_seconds.get());
                let before = seen.insert(name.to_string(), now).unwrap_or(0.0);
                let load = if now >= before { now - before } else { now } / elapsed.max(1e-9);
                loads.push(gf_assign::SliceLoad {
                    load,
                    rows: *rows as f64,
                    median: None,
                });
            }
        }
        let mut shard_loads = vec![0.0; gpus];
        for ((_, shard, _), load) in placed.iter().zip(&loads) {
            shard_loads[*shard] += load.load;
        }
        *mover
            .shard_loads
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = shard_loads.clone();
        mover
            .fresh
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
        let imbalance = |g: &[f64]| {
            let mean = g.iter().sum::<f64>() / g.len() as f64;
            if mean > 0.0 {
                g.iter().cloned().fold(0.0, f64::max) / mean
            } else {
                1.0
            }
        };
        let imbalance_before = imbalance(&shard_loads);
        let mut round = Round {
            moved: Vec::new(),
            imbalance_before,
            imbalance_after: imbalance_before,
            shard_loads,
            seconds: 0.0,
        };
        if placed.is_empty() || gpus < 2 {
            return Ok(round);
        }
        // Each table is one slice; tables are never split, merged or copied.
        let assignment = gf_assign::Assignment {
            generation: 0,
            slices: placed
                .iter()
                .enumerate()
                .map(|(index, (_, shard, _))| gf_assign::Slice {
                    start: index as u64,
                    holders: vec![*shard],
                })
                .collect(),
        };
        let policy = gf_assign::Policy {
            mode: gf_assign::Mode::Free,
            churn_budget: mover.churn_budget,
            imbalance_floor: mover.imbalance_floor,
            min_slices_per_gpu: usize::MAX,
            max_copies: 1,
            ..gf_assign::Policy::default()
        };
        let next = gf_assign::rebalance(&assignment, &loads, gpus, &policy);
        for ((name, from, _), slice) in placed.iter().zip(&next.assignment.slices) {
            let to = slice.holders[0];
            if to != *from {
                match self.move_table(name, to).await {
                    Ok(_) => round.moved.push((name.to_string(), *from, to)),
                    Err(error) => tracing::warn!(table = name.as_str(), "not moved: {error}"),
                }
            }
        }
        // The imbalance left by the moves that ran, not the ones planned.
        let mut after = vec![0.0; gpus];
        for ((name, from, _), load) in placed.iter().zip(&loads) {
            let shard = round
                .moved
                .iter()
                .find(|(moved, _, _)| moved == name.as_str())
                .map_or(*from, |&(_, _, to)| to);
            after[shard] += load.load;
        }
        round.imbalance_after = imbalance(&after);
        round.seconds = started.elapsed().as_secs_f64();
        self.metrics
            .placement_imbalance
            .set((round.imbalance_after * 1000.0) as u64);
        Ok(round)
    }

    /// Runs a round every `every` until the server stops.
    pub async fn rebalance(self, every: Duration) {
        let mut ticks = tokio::time::interval(every);
        ticks.tick().await;
        loop {
            ticks.tick().await;
            match self.rebalance_round().await {
                Ok(round) if !round.moved.is_empty() => {
                    tracing::info!(
                        moved = round.moved.len(),
                        imbalance = round.imbalance_after,
                        "rebalanced"
                    );
                }
                Ok(_) => {}
                Err(error) => tracing::warn!("rebalancing: {error}"),
            }
        }
    }
}
