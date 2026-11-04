// Temporary benchmark, not committed: a plan's scan on every core.
use std::num::NonZeroUsize;
use std::time::Instant;

mod support;

use gf_core::context::Context;
use gf_core::pages::{Bucket, Dense, PageIndex};
use gf_core::table::RowId;
use support::{Keeping, Kept};

struct Noise(u64);
impl Noise {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn env(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |v| v.parse().unwrap())
}

#[test]
#[ignore]
fn scan_threads() {
    let (rows, dims, queries, count) = (
        env("ROWS", 585_000),
        env("DIMS", 435),
        env("QUERIES", 1024),
        env("COUNT", 12_288),
    );
    let threads = env("THREADS", 16);
    let mut noise = Noise(3);
    let centers: Vec<Vec<f32>> = (0..64)
        .map(|_| (0..dims).map(|_| noise.next() * 3.0).collect())
        .collect();
    let mut index = PageIndex::new(
        NonZeroUsize::new(dims).unwrap(),
        NonZeroUsize::new(256).unwrap(),
    );
    for start in (0..rows).step_by(50_000) {
        let data: Vec<(RowId, Bucket, Vec<f32>)> = (start..(start + 50_000).min(rows))
            .map(|row| {
                (
                    RowId(row as u32),
                    Bucket(0),
                    centers[row % 64].iter().map(|c| c + noise.next()).collect(),
                )
            })
            .collect();
        index.put_many(&data).unwrap();
    }
    let allowed: Context = (0..rows as u32).map(RowId).collect();
    let points: Vec<Vec<f32>> = (0..queries)
        .map(|q| centers[q % 64].iter().map(|c| c + noise.next()).collect())
        .collect();
    let refs: Vec<&[f32]> = points.iter().map(Vec::as_slice).collect();
    let dense = Dense::read(&allowed, dims, &Kept, &Kept).unwrap();
    let scan = index.scan(&allowed, &dense);
    let share = queries.div_ceil(threads);
    let mut reference: Option<Vec<Context>> = None;
    for round in 0..3 {
        let started = Instant::now();
        let found: Vec<Context> = std::thread::scope(|scope| {
            let handles: Vec<_> = refs
                .chunks(share)
                .map(|chunk| {
                    let scan = &scan;
                    scope.spawn(move || scan.nearest(chunk, count).unwrap())
                })
                .collect();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect()
        });
        let seconds = started.elapsed().as_secs_f64();
        if let Some(reference) = &reference {
            assert_eq!(*reference, found);
        } else {
            reference = Some(found);
        }
        eprintln!(
            "SCAN rows {rows} dims {dims} queries {queries} count {count} threads {threads} block {}: round {round} {seconds:.3} s",
            std::env::var("GF_SCAN_QUERIES").unwrap_or_else(|_| "8".into())
        );
    }
    if let Ok(path) = std::env::var("SCAN_OUT") {
        let reference = reference.unwrap();
        let text: String = reference
            .iter()
            .map(|c| format!("{:?}\n", c.iter().map(|r| r.0).collect::<Vec<_>>()))
            .collect();
        std::fs::write(path, text).unwrap();
    }
}
