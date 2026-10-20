//! Placing tables on nodes with rendezvous hashing.

use std::collections::HashMap;

use gf_router::placement::Placement;

fn nodes(count: usize) -> Vec<String> {
    (0..count).map(|index| format!("node-{index}")).collect()
}

fn tables() -> Vec<String> {
    (0..2000).map(|index| format!("table-{index}")).collect()
}

#[test]
fn placement_is_deterministic_and_ranks_every_node() {
    let placement = Placement::new(nodes(4));

    let ranked = placement.ranked("payments");

    assert_eq!(ranked, Placement::new(nodes(4)).ranked("payments"));
    let mut sorted = ranked.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, [0, 1, 2, 3]);
    assert_eq!(placement.primary("payments"), ranked[0]);
}

#[test]
fn tables_spread_evenly_across_nodes() {
    let placement = Placement::new(nodes(4));
    let mut counts = HashMap::new();

    for table in tables() {
        *counts.entry(placement.primary(&table)).or_insert(0) += 1;
    }

    // 500 per node on average.
    assert!(
        counts.values().all(|&count| (400..600).contains(&count)),
        "{counts:?}"
    );
}

#[test]
fn adding_a_node_moves_only_the_tables_it_takes() {
    let before = Placement::new(nodes(4));
    let after = Placement::new(nodes(5));

    let moved: Vec<String> = tables()
        .into_iter()
        .filter(|table| before.primary(table) != after.primary(table))
        .collect();

    assert!(moved.iter().all(|table| after.primary(table) == 4));
    // About one table in five moves.
    assert!((300..500).contains(&moved.len()), "{} moved", moved.len());
}

#[test]
fn removing_a_node_moves_only_its_tables() {
    let before = Placement::new(nodes(4));
    let survivors: Vec<String> = nodes(4)
        .into_iter()
        .filter(|node| node != "node-2")
        .collect();
    let after = Placement::new(survivors.clone());

    for table in tables() {
        let old = &nodes(4)[before.primary(&table)];
        let new = &survivors[after.primary(&table)];
        if old != "node-2" {
            assert_eq!(old, new);
        }
    }
}
