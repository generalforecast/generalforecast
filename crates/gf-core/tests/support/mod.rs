// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Row vectors for tests that build a page index alone, which keeps none.

use std::cell::RefCell;
use std::collections::HashMap;

use gf_core::pages::{Bucket, IndexError, PageId, PageIndex, Vectors};
use gf_core::rows::RowsError;
use gf_core::table::{Keys, RowId};

thread_local! {
    /// Each row's vector, as the test put it into an index.
    static VECTORS: RefCell<HashMap<RowId, Vec<f32>>> = RefCell::new(HashMap::new());
}

/// The vectors this test's rows were put in with.
pub struct Kept;

impl Vectors for Kept {
    fn vector(&self, row: RowId) -> Result<Vec<f32>, RowsError> {
        VECTORS.with(|vectors| {
            vectors
                .borrow()
                .get(&row)
                .cloned()
                .ok_or(RowsError::Unknown(row))
        })
    }
}

/// Each row's key is its id, as these tests insert them.
impl Keys for Kept {
    fn key(&self, row: RowId) -> u128 {
        u128::from(row.0)
    }
}

/// Puts rows into an index and takes them out, keeping their vectors.
pub trait Keeping {
    fn put(&mut self, row: RowId, bucket: Bucket, vector: &[f32]) -> Result<PageId, IndexError>;
    // Not every test file takes rows out.
    #[allow(dead_code)]
    fn take(&mut self, row: RowId) -> Result<PageId, IndexError>;
    // Nor puts many in at once.
    #[allow(dead_code)]
    fn put_many(&mut self, rows: &[(RowId, Bucket, Vec<f32>)]) -> Result<Vec<PageId>, IndexError>;
}

impl Keeping for PageIndex {
    fn put(&mut self, row: RowId, bucket: Bucket, vector: &[f32]) -> Result<PageId, IndexError> {
        let new = VECTORS.with(|vectors| {
            let mut vectors = vectors.borrow_mut();
            let new = !vectors.contains_key(&row);
            if new {
                vectors.insert(row, vector.to_vec());
            }
            new
        });
        let placed = self.insert(row, bucket, vector, &Kept);
        if placed.is_err() && new {
            VECTORS.with(|vectors| vectors.borrow_mut().remove(&row));
        }
        placed
    }

    fn put_many(&mut self, rows: &[(RowId, Bucket, Vec<f32>)]) -> Result<Vec<PageId>, IndexError> {
        VECTORS.with(|vectors| {
            let mut vectors = vectors.borrow_mut();
            for (row, _, vector) in rows {
                vectors.entry(*row).or_insert_with(|| vector.clone());
            }
        });
        self.insert_many(rows, &Kept)
    }

    fn take(&mut self, row: RowId) -> Result<PageId, IndexError> {
        let taken = self.remove(row, &Kept);
        if taken.is_ok() {
            VECTORS.with(|vectors| vectors.borrow_mut().remove(&row));
        }
        taken
    }
}
