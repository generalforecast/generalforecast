// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Flight calls as Arrow data: the rows a request's batches carry, decoded
//! and charged as they arrive, and the batches predictions answer with.

use super::*;

pub(super) fn invalid(message: impl Into<String>) -> Status {
    Status::invalid_argument(message.into())
}

/// The command, table and body an exchange's descriptor path names: a
/// prediction's `predict/payments`, or a partition's call with its JSON body
/// third.
pub(super) fn exchange_path(
    descriptor: Option<&FlightDescriptor>,
) -> Result<(String, TableName, Vec<u8>), Status> {
    let mut parts = descriptor
        .map(|descriptor| descriptor.path.clone())
        .unwrap_or_default();
    let body = if parts.len() == 3 {
        parts.pop().map(String::into_bytes).unwrap_or_default()
    } else {
        Vec::new()
    };
    let named = FlightDescriptor::new_path(parts);
    let (command, table) = path(Some(&named))?;
    Ok((command, table, body))
}

/// The command and table a descriptor path names, such as `insert/payments`.
pub(super) fn path(descriptor: Option<&FlightDescriptor>) -> Result<(String, TableName), Status> {
    let parts = descriptor
        .map(|descriptor| descriptor.path.clone())
        .unwrap_or_default();
    match parts.as_slice() {
        [command, table] => {
            let table =
                TableName::new(table.as_str()).map_err(|error| invalid(error.to_string()))?;
            Ok((command.clone(), table))
        }
        _ => Err(invalid("the descriptor path must be [command, table]")),
    }
}

/// The descriptor a Flight stream's first message carries, without reading past it.
pub(super) async fn descriptor(
    stream: &mut Peekable<Streaming<FlightData>>,
) -> Option<FlightDescriptor> {
    match Pin::new(stream).peek().await {
        Some(Ok(first)) => first.flight_descriptor.clone(),
        _ => None,
    }
}

/// The record batches in a Flight stream, charged to `charge` as they
/// arrive, and the charge, which the call holds until the shard has answered
/// what the batches became.
///
/// `DoExchange` clients may open with a message that holds only the descriptor,
/// so messages without an Arrow header are skipped. A stream past
/// `REQUEST_BYTES` on the wire or once decoded, or past the server's budget,
/// is refused as it arrives.
pub(super) async fn batches(
    stream: Peekable<Streaming<FlightData>>,
    charge: Charge,
) -> Result<(Vec<RecordBatch>, Charge), Status> {
    let mut received = 0;
    let mut decoded = 0;
    // Shared by the wire and the decoder, whose streams outlive no call.
    let charge = Arc::new(std::sync::Mutex::new(charge));
    let wire_charge = Arc::clone(&charge);
    let decoded_charge = Arc::clone(&charge);
    let messages = stream
        .try_filter(|message| futures::future::ready(!message.data_header.is_empty()))
        .and_then(move |message| {
            received += message.data_body.len();
            let taken = if received > REQUEST_BYTES {
                Err(invalid(format!(
                    "a request may carry at most {} MiB; split it",
                    REQUEST_BYTES >> 20
                )))
            } else {
                wire_charge
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .grow(message.data_body.len() as u64)
            };
            futures::future::ready(taken.map(|()| message))
        })
        .map_err(|status| FlightError::Tonic(Box::new(status)));
    let batches: Vec<RecordBatch> = FlightRecordBatchStream::new_from_flight_data(messages)
        .and_then(move |batch| {
            // The wire bounds nothing once decoded: a column of Arrow's null
            // type takes no bytes however many rows its header claims.
            let bytes = decoded_bytes(&batch);
            decoded += bytes;
            let taken = if decoded > REQUEST_BYTES {
                Err(invalid(format!(
                    "a request's rows may take at most {} MiB in memory; split it",
                    REQUEST_BYTES >> 20
                )))
            } else {
                decoded_charge
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .grow(bytes as u64)
            };
            futures::future::ready(
                taken
                    .map(|()| batch)
                    .map_err(|status| FlightError::Tonic(Box::new(status))),
            )
        })
        .try_collect()
        .await
        .map_err(|error| match error {
            FlightError::Tonic(status) => *status,
            error => invalid(format!("bad Arrow data: {error}")),
        })?;
    let charge = Arc::into_inner(charge)
        .expect("the streams that shared the charge are done")
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner);
    Ok((batches, charge))
}

/// A query exchange's rows: its batches, held against `charge` until the
/// shard has answered them, and their features and words for `columns`.
///
/// # Errors
///
/// As `batches` and `features`, and `INVALID_ARGUMENT` for no rows at all.
pub(super) async fn query_rows(
    stream: Peekable<Streaming<FlightData>>,
    columns: &[Column],
    charge: Charge,
) -> Result<(Vec<RecordBatch>, Charge, Vec<f32>, Strings), Status> {
    let (batches, charge) = batches(stream, charge).await?;
    let (features, strings, rows) = features(&batches, columns)?;
    if rows == 0 {
        return Err(invalid("no query rows"));
    }
    Ok((batches, charge, features, strings))
}

/// One batch as a Flight response.
pub(super) fn encoded(batch: RecordBatch) -> Response<Boxed<FlightData>> {
    let encoded = FlightDataEncoderBuilder::new()
        .build(stream::iter([Ok(batch)]))
        .map_err(|error| Status::internal(error.to_string()));
    Response::new(Box::pin(encoded))
}

/// Bytes a batch takes once decoded for the shard, beyond the Arrow data it
/// arrived as, which the wire charge covers: four per cell, with each string
/// of a text column in an allocation of its own.
fn decoded_bytes(batch: &RecordBatch) -> usize {
    let rows = batch.num_rows();
    batch
        .columns()
        .iter()
        .map(|column| {
            let per_row = match column.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Dictionary(..) => 32,
                DataType::FixedSizeBinary(_) => 16,
                _ => 4,
            };
            rows * per_row
        })
        .sum()
}

fn float_column(column: &dyn Array, name: &str) -> Result<Vec<f32>, Status> {
    let values = column
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| invalid(format!("column {name:?} must be float32")))?;
    Ok((0..values.len())
        .map(|index| {
            if values.is_null(index) {
                f32::NAN
            } else {
                values.value(index)
            }
        })
        .collect())
}

/// Row-major features from the batches' feature columns, by name and kind:
/// numbers in place, and the strings of categorical columns apart, for the
/// shard to code. Columns the table does not have are left alone.
pub(super) fn features(
    batches: &[RecordBatch],
    columns: &[Column],
) -> Result<(Vec<f32>, Strings, usize), Status> {
    let mut rows = Vec::new();
    let mut strings: BTreeMap<usize, Vec<Option<String>>> = BTreeMap::new();
    let mut count = 0;
    for batch in batches {
        let mut decoded = Vec::with_capacity(columns.len());
        for column in columns {
            let array = batch
                .column_by_name(&column.name)
                .ok_or_else(|| invalid(format!("missing column {:?}", column.name)))?;
            decoded.push(columns::decode(array, column).map_err(invalid)?);
        }
        for row in 0..batch.num_rows() {
            for cells in &decoded {
                cells.extend_row(row, &mut rows);
            }
        }
        for (index, cells) in decoded.into_iter().enumerate() {
            if let Cells::Strings(values) = cells {
                strings.entry(index).or_default().extend(values);
            }
        }
        count += batch.num_rows();
    }
    for (index, values) in &strings {
        if values.len() != count {
            return Err(invalid(format!(
                "column {:?} must hold strings in every batch or in none",
                columns[*index].name
            )));
        }
    }
    Ok((rows, strings.into_iter().collect(), count))
}

/// A column that must be there, without nulls.
pub(super) fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a dyn Array, Status> {
    let column = batch
        .column_by_name(name)
        .map(AsRef::as_ref)
        .ok_or_else(|| invalid(format!("missing column {name:?}")))?;
    if column.null_count() > 0 {
        return Err(invalid(format!("column {name:?} has nulls")));
    }
    Ok(column)
}

/// A batch's 128-bit keys; see `partition::keys_of`.
fn batch_keys(batch: &RecordBatch) -> Result<Vec<u128>, Status> {
    if let Some(key) = batch.column_by_name("key")
        && key.null_count() > 0
    {
        return Err(invalid("column \"key\" has nulls"));
    }
    partition::keys_of(batch).map_err(invalid)
}

/// Each query's given context from the batches' "context" column, a list of
/// 16-byte keys per query, if they have one: in every batch, or in none.
pub(super) fn given_keys(batches: &[RecordBatch]) -> Result<Option<Vec<Vec<u128>>>, Status> {
    let having = batches
        .iter()
        .filter(|batch| batch.column_by_name("context").is_some())
        .count();
    if having == 0 {
        return Ok(None);
    }
    if having != batches.len() {
        return Err(invalid(
            "column \"context\" must be in every batch or in none",
        ));
    }
    let mut contexts = Vec::new();
    for batch in batches {
        let lists = batch
            .column_by_name("context")
            .and_then(|column| column.as_any().downcast_ref::<arrow_array::ListArray>())
            .ok_or_else(|| invalid("column \"context\" holds a list of 16-byte keys per query"))?;
        for index in 0..lists.len() {
            let list = lists.value(index);
            let keys = list
                .as_any()
                .downcast_ref::<arrow_array::FixedSizeBinaryArray>()
                .filter(|keys| keys.value_length() == 16 && keys.null_count() == 0)
                .ok_or_else(|| {
                    invalid("column \"context\" holds a list of 16-byte keys per query")
                })?;
            contexts.push(
                (0..keys.len())
                    .map(|key| u128::from_le_bytes(keys.value(key).try_into().expect("16 bytes")))
                    .collect(),
            );
        }
    }
    Ok(Some(contexts))
}

/// Each series' group of a forecast request, from a `group` column of lists
/// of series digests, or None without one.
pub(super) fn groups_column(batches: &[RecordBatch]) -> Result<Option<Vec<Vec<u64>>>, Status> {
    let having = batches
        .iter()
        .filter(|batch| batch.column_by_name("group").is_some())
        .count();
    if having == 0 {
        return Ok(None);
    }
    if having != batches.len() {
        return Err(invalid(
            "column \"group\" must be in every batch or in none",
        ));
    }
    let mut groups = Vec::new();
    for batch in batches {
        let lists = batch
            .column_by_name("group")
            .and_then(|column| column.as_any().downcast_ref::<arrow_array::ListArray>())
            .ok_or_else(|| invalid("column \"group\" holds a list of series per forecast"))?;
        for index in 0..lists.len() {
            let list = lists.value(index);
            let any = list.as_any();
            let members: Vec<u64> = if let Some(unsigned) = any.downcast_ref::<UInt64Array>() {
                unsigned.values().iter().copied().collect()
            } else if let Some(signed) = any.downcast_ref::<Int64Array>() {
                signed.values().iter().map(|&key| key as u64).collect()
            } else {
                return Err(invalid(
                    "column \"group\" holds a list of uint64 or int64 series per forecast",
                ));
            };
            groups.push(members);
        }
    }
    Ok(Some(groups))
}

/// The known future a forecast request brings, from a `future` column of
/// lists, one per series: rows of its `time` and the table's covariate
/// columns, as a write's rows carry them, null where a value is not known.
/// None without the column.
pub(super) fn future_column(
    batches: &[RecordBatch],
    series: &[u64],
    columns: &[Column],
) -> Result<Option<KnownFuture>, Status> {
    let having = batches
        .iter()
        .filter(|batch| batch.column_by_name("future").is_some())
        .count();
    if having == 0 {
        return Ok(None);
    }
    if having != batches.len() {
        return Err(invalid(
            "column \"future\" must be in every batch or in none",
        ));
    }
    let mut future = KnownFuture {
        series: Vec::new(),
        times: Vec::new(),
        features: Vec::new(),
        strings: Vec::new(),
    };
    let mut strings: BTreeMap<usize, Vec<Option<String>>> = BTreeMap::new();
    let mut at = 0;
    for batch in batches {
        let lists = batch
            .column_by_name("future")
            .and_then(|column| column.as_any().downcast_ref::<arrow_array::ListArray>())
            .ok_or_else(|| invalid("column \"future\" holds a list of rows per forecast"))?;
        let rows = lists
            .values()
            .as_any()
            .downcast_ref::<arrow_array::StructArray>()
            .ok_or_else(|| invalid("column \"future\" holds rows: a list of structs"))?;
        let rows = RecordBatch::from(rows.clone());
        let times = column(&rows, "time")?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| invalid("a future row's \"time\" must be int64"))?;
        let (values, words, count) = features(std::slice::from_ref(&rows), columns)?;
        let offsets = lists.value_offsets();
        for index in 0..lists.len() {
            let key = series[at + index];
            let span = usize::try_from(offsets[index + 1] - offsets[index]).unwrap_or(0);
            future.series.extend(std::iter::repeat_n(key, span));
        }
        at += lists.len();
        future.times.extend(times.values().iter().copied());
        future.features.extend(values);
        for (column, words) in words {
            strings.entry(column).or_default().extend(words);
        }
        if future.times.len() != future.series.len() || count != times.len() {
            return Err(invalid("a future's rows must each belong to a forecast"));
        }
    }
    future.strings = strings.into_iter().collect();
    Ok(Some(future))
}

/// The keys of delete batches.
pub(super) fn keys(batches: &[RecordBatch]) -> Result<Vec<u128>, Status> {
    let mut keys = Vec::new();
    for batch in batches {
        keys.extend(batch_keys(batch)?);
    }
    Ok(keys)
}

/// Keys, features and targets from insert or update batches, or keys and
/// targets alone from label batches. A target not known yet is NaN, or null
/// or -1 for a class.
pub(super) fn changes(
    batches: &[RecordBatch],
    placement: &Placement,
    id: Option<String>,
    labels_only: bool,
) -> Result<Incoming, Status> {
    let mut keys = Vec::new();
    let mut targets = match placement.task {
        Task::Values => Targets::Values(Vec::new()),
        Task::Classes(_) => Targets::Classes(Vec::new()),
    };
    // String labels, which the table's label vocabulary codes after its columns'.
    let mut labels_given: Vec<Option<String>> = Vec::new();
    for batch in batches {
        keys.extend(batch_keys(batch)?);
        // A null target is one not known yet.
        let target = batch
            .column_by_name("target")
            .map(AsRef::as_ref)
            .ok_or_else(|| invalid("missing column \"target\""))?;
        match &mut targets {
            Targets::Classes(codes) => {
                if let Some(given) = target.as_any().downcast_ref::<Int32Array>() {
                    codes.extend(given.iter().map(|code| code.unwrap_or(UNLABELED)));
                    continue;
                }
                let label = Column {
                    name: "target".to_string(),
                    kind: ColumnKind::Categorical,
                };
                let array = batch.column_by_name("target").expect("found above");
                let Ok(Cells::Strings(strings)) = columns::decode(array, &label) else {
                    return Err(invalid(
                        "column \"target\" of a classification table holds int32 class \
                         codes or string labels",
                    ));
                };
                codes.extend(std::iter::repeat_n(0, strings.len()));
                labels_given.extend(strings);
            }
            Targets::Values(values) => {
                let targets = float_column(target, "target")?;
                // A series table stores a step without a target as NaN, and
                // any table a row whose label is not known yet.
                if targets.iter().any(|value| value.is_infinite()) {
                    return Err(invalid(if placement.series.is_some() {
                        "targets of a series table are finite or NaN, for a step without one"
                    } else {
                        "regression targets are finite, or NaN for rows not labeled yet"
                    }));
                }
                values.extend(targets);
            }
        }
    }
    let (features, mut strings, rows) = if labels_only {
        (Vec::new(), Vec::new(), keys.len())
    } else {
        features(batches, &placement.columns)?
    };
    if rows != keys.len() {
        return Err(invalid("every row needs a key"));
    }
    if !labels_given.is_empty() {
        if labels_given.len() != rows {
            return Err(invalid(
                "column \"target\" must hold labels in every batch or in none",
            ));
        }
        strings.push((placement.columns.len(), labels_given));
    }
    let series = if placement.series.is_some() && !labels_only {
        series_keys(batches, &keys)?
    } else {
        Vec::new()
    };
    Ok(Incoming {
        rows: Rows {
            keys,
            features,
            targets,
            times: times(batches, "time")?.map(|times| times.into_iter().map(Some).collect()),
            id,
            words: Vec::new(),
            series,
            writes: Vec::new(),
            layout: None,
            table: None,
        },
        strings,
    })
}

/// The series the observations of a write to a table of series belong to,
/// each once: its digest, the high half of its observations' row keys, and
/// its key as the client gave it, which column `"series_key"` holds per
/// observation as the JSON of a `GivenKey`, plain or dictionary-encoded text.
/// A key that does not digest to its observations' series is refused.
fn series_keys(batches: &[RecordBatch], keys: &[u128]) -> Result<Vec<SeriesKey>, Status> {
    let mut named = Vec::new();
    let mut seen = HashSet::new();
    let mut row = 0;
    for batch in batches {
        let column = column(batch, "series_key")?;
        if !matches!(
            column.data_type(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Dictionary(..)
        ) {
            return Err(invalid(
                "column \"series_key\" holds each observation's series key as text",
            ));
        }
        let texts = arrow_cast::cast(column, &DataType::Utf8)
            .map_err(|error| invalid(format!("column \"series_key\": {error}")))?;
        let texts = texts
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("cast to utf8");
        for index in 0..texts.len() {
            let Some(key) = keys.get(row) else {
                return Err(invalid("every observation needs a key"));
            };
            row += 1;
            let digest = (key >> 64) as u64;
            if !seen.insert(digest) {
                continue;
            }
            if texts.is_null(index) {
                return Err(invalid("a series key cannot be missing"));
            }
            let given: GivenKey = serde_json::from_str(texts.value(index)).map_err(|error| {
                invalid(format!(
                    "column \"series_key\" holds each series' key as the JSON of its kind and \
                     value: {error}"
                ))
            })?;
            let series = SeriesKey { digest, key: given };
            series
                .check()
                .map_err(|error| invalid(format!("column \"series_key\": {error}")))?;
            named.push(series);
        }
    }
    if row != keys.len() {
        return Err(invalid("every observation needs a series key"));
    }
    Ok(named)
}

/// The series a forecast batch names, by their keys' 64-bit digests: a uint64
/// or int64 column, or 16-byte keys as `generalforecast.keys.series_keys` digests them.
pub(super) fn series_column(batches: &[RecordBatch]) -> Result<Vec<u64>, Status> {
    let mut keys = Vec::new();
    for batch in batches {
        let column = column(batch, "series")?;
        let any = column.as_any();
        if let Some(unsigned) = any.downcast_ref::<UInt64Array>() {
            keys.extend(unsigned.values().iter().copied());
        } else if let Some(signed) = any.downcast_ref::<Int64Array>() {
            keys.extend(signed.values().iter().map(|&key| key as u64));
        } else if let Some(wide) = any.downcast_ref::<FixedSizeBinaryArray>()
            && wide.value_length() == 16
        {
            keys.extend((0..wide.len()).map(|index| {
                u64::from_le_bytes(wide.value(index)[..8].try_into().expect("8 bytes"))
            }));
        } else {
            return Err(invalid(
                "column \"series\" must be uint64, int64 or fixed_size_binary(16)",
            ));
        }
    }
    Ok(keys)
}

/// An optional int64 column's values across batches: in every batch, or in
/// none, and never null.
pub(super) fn times(batches: &[RecordBatch], name: &str) -> Result<Option<Vec<i64>>, Status> {
    let having = batches
        .iter()
        .filter(|batch| batch.column_by_name(name).is_some())
        .count();
    if having == 0 {
        return Ok(None);
    }
    if having < batches.len() {
        return Err(invalid(format!(
            "column {name:?} must be in every batch or in none"
        )));
    }
    let mut values = Vec::new();
    for batch in batches {
        let times = column(batch, name)?
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| invalid(format!("column {name:?} must be int64")))?;
        if times.null_count() > 0 {
            return Err(invalid(format!("column {name:?} must not hold nulls")));
        }
        values.extend(times.values().iter().copied());
    }
    Ok(Some(values))
}

/// Predictions as a batch: `p0..` class probabilities, or `q001..` quantiles.
/// The quantile levels a client asked a regressor's predictions to keep, if any.
pub(super) fn quantile_levels<T>(request: &Request<T>) -> Result<Option<Vec<f64>>, Status> {
    let Some(given) = request.metadata().get(QUANTILES) else {
        return Ok(None);
    };
    given
        .to_str()
        .map_err(|_| invalid(format!("{QUANTILES} is a comma-separated list")))
        .and_then(|given| {
            quantiles::parse_levels(given).map_err(|error| invalid(format!("{QUANTILES}: {error}")))
        })
        .map(Some)
}

/// The quantile columns of predictions at `levels`, as `quantile_values`
/// computes them, named `q<level>`.
fn quantile_columns(
    values: &[f32],
    known: &[f64],
    levels: &[f64],
) -> Result<Vec<(Field, ArrayRef)>, Status> {
    let picked = quantile_values(values, known, levels)
        .ok_or_else(|| Status::internal("a regressor's predictions have no columns"))?;
    Ok(levels
        .iter()
        .zip(picked)
        .map(|(&level, picked)| {
            (
                Field::new(format!("q{level}"), DataType::Float32, false),
                Arc::new(Float32Array::from(picked)) as ArrayRef,
            )
        })
        .collect())
}

/// The table's quantile levels for its prediction columns: its own, or the
/// evenly spaced ones an earlier placement implies.
fn known_levels(placement: &Placement, columns: usize) -> Vec<f64> {
    match &placement.levels {
        Some(levels) if levels.len() == columns => levels.to_vec(),
        _ => (0..columns)
            .map(|column| (column + 1) as f64 / (columns + 1) as f64)
            .collect(),
    }
}

/// Predictions at quantile levels; columns `q<level>`.
pub(super) fn quantile_batch(
    predictions: &Predictions,
    levels: &[f64],
    placement: &Placement,
) -> Result<RecordBatch, Status> {
    let known = known_levels(placement, predictions.columns);
    let (fields, arrays): (Vec<Field>, Vec<Arc<dyn Array>>) =
        quantile_columns(&predictions.values, &known, levels)?
            .into_iter()
            .unzip();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|error| Status::internal(error.to_string()))
}

/// Forecasts as a long batch: each series' digest and the time of each step,
/// then one `q<level>` column per level, the table's own without `levels`.
pub(super) fn forecasted(
    predictions: &Predictions,
    series: &[u64],
    step: i64,
    horizon: NonZeroUsize,
    levels: Option<&[f64]>,
    placement: &Placement,
) -> Result<RecordBatch, Status> {
    let horizon = horizon.get();
    if predictions.horizon != Some(horizon)
        || predictions.rows != series.len()
        || predictions.cutoffs.len() != series.len()
    {
        return Err(Status::internal(
            "a forecast per series over the horizon asked was expected",
        ));
    }
    let known = known_levels(placement, predictions.columns);
    if placement
        .levels
        .as_ref()
        .is_some_and(|levels| levels.len() != predictions.columns)
    {
        return Err(Status::internal(
            "the forecast has a column per quantile level of the table",
        ));
    }
    let mut fields = vec![
        Field::new("series", DataType::UInt64, false),
        Field::new("time", DataType::Int64, false),
    ];
    let mut arrays: Vec<Arc<dyn Array>> = vec![
        Arc::new(UInt64Array::from(
            series
                .iter()
                .flat_map(|&key| std::iter::repeat_n(key, horizon))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            predictions
                .cutoffs
                .iter()
                .flat_map(|&cutoff| {
                    (0..horizon as i64).map(move |ahead| cutoff.saturating_add(ahead * step))
                })
                .collect::<Vec<_>>(),
        )),
    ];
    let columns = quantile_columns(&predictions.values, &known, levels.unwrap_or(&known))?;
    for (field, array) in columns {
        fields.push(field);
        arrays.push(array);
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
        .map_err(|error| Status::internal(error.to_string()))
}

pub(super) fn prediction_batch(
    predictions: &Predictions,
    task: Task,
) -> Result<RecordBatch, Status> {
    let names: Vec<String> = (0..predictions.columns)
        .map(|index| match task {
            Task::Values => format!("q{:03}", index + 1),
            Task::Classes(_) => format!("p{index}"),
        })
        .collect();
    let schema = Arc::new(Schema::new(
        names
            .iter()
            .map(|name| Field::new(name, DataType::Float32, false))
            .collect::<Vec<_>>(),
    ));
    let columns: Vec<Arc<dyn Array>> = (0..predictions.columns)
        .map(|column| {
            let values: Vec<f32> = (0..predictions.rows)
                .map(|row| predictions.values[row * predictions.columns + column])
                .collect();
            Arc::new(Float32Array::from(values)) as Arc<dyn Array>
        })
        .collect();
    RecordBatch::try_new(schema, columns).map_err(|error| Status::internal(error.to_string()))
}

/// The horizon a forecast call asks for, if it names one.
pub(super) fn horizon_of<T>(request: &Request<T>) -> Result<Option<usize>, Status> {
    let Some(given) = request.metadata().get(HORIZON) else {
        return Ok(None);
    };
    given
        .to_str()
        .ok()
        .and_then(|steps| steps.trim().parse::<usize>().ok())
        .filter(|&steps| steps > 0)
        .map(Some)
        .ok_or_else(|| invalid(format!("{HORIZON} is a whole number of steps, at least 1")))
}

/// A batch as a Flight response, with the predictions' worker time and
/// epoch in the schema's metadata. The reply streams, and if `laps` are
/// timed, the time since their last lap plus the time spent encoding each
/// message goes to `request_encode_seconds` once the stream ends or is
/// dropped.
pub(super) fn answered(
    batch: RecordBatch,
    predictions: &Predictions,
    mut laps: Laps,
    metrics: Arc<Metrics>,
) -> Result<Response<Boxed<FlightData>>, Status> {
    let schema = Schema::clone(&batch.schema()).with_metadata(HashMap::from([
        (GPU_SECONDS.to_string(), predictions.gpu_seconds.to_string()),
        (EPOCH.to_string(), predictions.epoch.to_string()),
    ]));
    let batch = batch
        .with_schema(Arc::new(schema))
        .map_err(|error| Status::internal(error.to_string()))?;
    let encoder = FlightDataEncoderBuilder::new()
        .build(stream::iter([Ok(batch)]))
        .map_err(|error| Status::internal(error.to_string()))
        .boxed();
    if !laps.timed() {
        return Ok(Response::new(encoder));
    }
    let timed = Timed::new(encoder, laps.lap(), move |spent| {
        metrics.request_encode_seconds.observe(spent);
    });
    Ok(Response::new(Box::pin(timed)))
}

/// The time between a request's parts, read only when the server times
/// requests: untimed laps read no clock and observe nothing.
#[derive(Clone, Copy)]
pub(super) struct Laps(pub(super) Option<Instant>);

impl Laps {
    pub(super) fn timed(self) -> bool {
        self.0.is_some()
    }

    /// The time since the last lap, which starts the next; zero untimed.
    pub(super) fn lap(&mut self) -> Duration {
        let Some(last) = &mut self.0 else {
            return Duration::ZERO;
        };
        let now = Instant::now();
        let spent = now - *last;
        *last = now;
        spent
    }

    /// Counts `spent` in `histogram` if timed.
    pub(super) fn observe(self, histogram: &Histogram, spent: Duration) {
        if self.timed() {
            histogram.observe(spent);
        }
    }
}

/// A stream that sums the time its inner stream takes to produce each item
/// and hands the total to `done` once, when the stream ends or is dropped.
struct Timed<S, F: FnOnce(Duration)> {
    stream: S,
    spent: Duration,
    done: Option<F>,
}

impl<S, F: FnOnce(Duration)> Timed<S, F> {
    pub(super) fn new(stream: S, spent: Duration, done: F) -> Self {
        Self {
            stream,
            spent,
            done: Some(done),
        }
    }

    pub(super) fn finish(&mut self) {
        if let Some(done) = self.done.take() {
            done(self.spent);
        }
    }
}

impl<S: Stream + Unpin, F: FnOnce(Duration) + Unpin> Stream for Timed<S, F> {
    type Item = S::Item;

    fn poll_next(
        mut self: Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<S::Item>> {
        let polling = Instant::now();
        let polled = self.stream.poll_next_unpin(context);
        if let std::task::Poll::Ready(item) = &polled {
            self.spent += polling.elapsed();
            if item.is_none() {
                self.finish();
            }
        }
        polled
    }
}

impl<S, F: FnOnce(Duration)> Drop for Timed<S, F> {
    fn drop(&mut self) {
        self.finish();
    }
}

/// A client's id for a write, from the `gf-write-id` metadata.
pub(super) fn write_id<T>(request: &Request<T>) -> Result<Option<String>, Status> {
    match request.metadata().get(WRITE_ID) {
        Some(id) => {
            let id = id
                .to_str()
                .map_err(|_| invalid(format!("{WRITE_ID} must be ASCII")))?;
            check_write_id(id).map_err(|error| invalid(format!("{WRITE_ID}: {error}")))?;
            Ok(Some(id.to_string()))
        }
        None => Ok(None),
    }
}

/// Metadata a router sets on a shipment: the source copy's terms and creation id.
pub(super) fn lineage<T>(request: &Request<T>) -> Result<Lineage, Status> {
    let terms = match request.metadata().get(TERMS) {
        Some(terms) => Some(
            terms
                .to_str()
                .ok()
                .and_then(|terms| serde_json::from_str(terms).ok())
                .ok_or_else(|| invalid(format!("{TERMS} must be a JSON list of terms")))?,
        ),
        None => None,
    };
    let created = request
        .metadata()
        .get(CREATED)
        .and_then(|created| created.to_str().ok())
        .and_then(CreationId::parse);
    Ok(Lineage { terms, created })
}

/// A new creation id, by this node's clock.
pub(super) fn creation_id() -> Result<CreationId, Status> {
    let mut bytes = [0u8; 2];
    getrandom::fill(&mut bytes).map_err(|error| Status::internal(error.to_string()))?;
    let millis = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|error| Status::internal(error.to_string()))?
        .as_millis();
    Ok(CreationId::new(millis, bytes))
}

/// What a write changed, and the creation of the table it changed.
pub(super) fn answer(written: &crate::shard::Written, placement: &Placement) -> serde_json::Value {
    json!(Answer {
        rows: Some(written.rows),
        log_length: Some(written.log_length),
        created: placement.created.clone(),
    })
}

pub(super) fn results(values: Vec<serde_json::Value>) -> Boxed<arrow_flight::Result> {
    let items = values.into_iter().map(|value| {
        Ok(arrow_flight::Result {
            body: value.to_string().into(),
        })
    });
    Box::pin(stream::iter(items.collect::<Vec<_>>()))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use futures::{StreamExt, executor, stream};

    use super::Timed;

    /// The times a `Timed` stream recorded.
    type Recorded = Arc<Mutex<Vec<Duration>>>;

    /// A stream of three items that counts how many have been pulled, and
    /// the encode times its `Timed` wrapper recorded.
    fn timed() -> (
        impl futures::Stream<Item = Result<usize, String>> + Unpin,
        Arc<AtomicUsize>,
        Recorded,
    ) {
        let pulled = Arc::new(AtomicUsize::new(0));
        let recorded = Arc::new(Mutex::new(Vec::new()));
        let counter = Arc::clone(&pulled);
        let items = stream::iter([Ok(0), Err("bad".to_string()), Ok(2)])
            .inspect(move |_| _ = counter.fetch_add(1, Ordering::SeqCst));
        let sink = Arc::clone(&recorded);
        let stream = Timed::new(items, Duration::from_secs(1), move |spent| {
            sink.lock().unwrap().push(spent);
        });
        (stream, pulled, recorded)
    }

    #[test]
    fn timed_streams_items_in_order_and_records_once_at_the_end() {
        let (mut stream, pulled, recorded) = timed();
        assert_eq!(pulled.load(Ordering::SeqCst), 0);

        let mut seen = Vec::new();
        while let Some(item) = executor::block_on(stream.next()) {
            assert_eq!(pulled.load(Ordering::SeqCst), seen.len() + 1);
            assert!(recorded.lock().unwrap().is_empty());
            seen.push(item);
        }
        assert_eq!(seen, [Ok(0), Err("bad".to_string()), Ok(2)]);
        assert_eq!(recorded.lock().unwrap().len(), 1);
        assert!(recorded.lock().unwrap()[0] >= Duration::from_secs(1));

        drop(stream);
        assert_eq!(recorded.lock().unwrap().len(), 1);
    }

    #[test]
    fn timed_records_once_when_dropped_early() {
        let (mut stream, pulled, recorded) = timed();
        assert_eq!(executor::block_on(stream.next()), Some(Ok(0)));
        assert!(recorded.lock().unwrap().is_empty());

        drop(stream);
        assert_eq!(pulled.load(Ordering::SeqCst), 1);
        assert_eq!(recorded.lock().unwrap().len(), 1);
    }
}
