// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Per-token request quotas.

use std::collections::HashMap;

use gf_server::auth::digest;
use std::time::Duration;

use tokio::time::Instant;

use gf_router::quota::{Limit, Quotas, TokenBucket};

#[test]
fn a_bucket_allows_a_burst_then_refills_at_its_rate() {
    let start = Instant::now();
    let mut bucket = TokenBucket::new(
        Limit {
            per_second: 2.0,
            burst: 3,
        },
        start,
    );

    let burst: Vec<bool> = (0..4).map(|_| bucket.take(start)).collect();
    let after_half_a_second = bucket.take(start + Duration::from_millis(500));
    let right_after = bucket.take(start + Duration::from_millis(500));

    assert_eq!(burst, [true, true, true, false]);
    assert!(after_half_a_second);
    assert!(!right_after);
}

#[test]
fn a_bucket_never_holds_more_than_its_burst() {
    let start = Instant::now();
    let mut bucket = TokenBucket::new(
        Limit {
            per_second: 100.0,
            burst: 2,
        },
        start,
    );
    let later = start + Duration::from_secs(60);

    let taken = (0..5).filter(|_| bucket.take(later)).count();

    assert_eq!(taken, 2);
}

#[test]
fn tokens_without_a_limit_are_not_limited_and_each_token_has_its_own_bucket() {
    let limits = HashMap::from([(
        digest("slow"),
        Limit {
            per_second: 1.0,
            burst: 1,
        },
    )]);
    let quotas = Quotas::new(limits);

    assert!(quotas.check(&digest("slow")).is_ok());
    assert!(quotas.check(&digest("slow")).is_err());
    assert!((0..100).all(|_| quotas.check(&digest("fast")).is_ok()));
}

#[test]
fn a_reload_refills_no_bucket() {
    let limit = |burst| {
        HashMap::from([(
            digest("slow"),
            Limit {
                per_second: 0.001,
                burst,
            },
        )])
    };
    let quotas = Quotas::new(limit(2));
    assert!(quotas.check(&digest("slow")).is_ok());

    // The same limit goes on from the one call left.
    let reloaded = Quotas::new(limit(2)).continuing(&quotas);
    assert!(reloaded.check(&digest("slow")).is_ok());
    assert!(reloaded.check(&digest("slow")).is_err());

    // A larger burst refills nothing, and a smaller one caps what is left.
    let raised = Quotas::new(limit(5)).continuing(&reloaded);
    assert!(raised.check(&digest("slow")).is_err());
    let fresh = Quotas::new(limit(3));
    assert!(fresh.check(&digest("slow")).is_ok());
    let lowered = Quotas::new(limit(1)).continuing(&fresh);
    assert!(lowered.check(&digest("slow")).is_ok());
    assert!(lowered.check(&digest("slow")).is_err());
    // A token no longer limited leaves its bucket behind.
    let unlimited = Quotas::new(HashMap::new()).continuing(&raised);
    assert!((0..10).all(|_| unlimited.check(&digest("slow")).is_ok()));
}
