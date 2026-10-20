use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const VERSION: &str = "gf/v1alpha1";

pub fn name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 63
        && value.as_bytes()[0].is_ascii_lowercase()
        && value
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPin {
    repository: String,
    revision: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PredictionSpec {
    api_version: String,
    dataset: String,
    model: ModelPin,
    target: String,
    context: String,
    max_update_lag_seconds: f64,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentConfig {
    api_version: String,
    kind: String,
    name: String,
    prediction: PredictionSpec,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetConfig {
    api_version: String,
    kind: String,
    name: String,
    features: Vec<String>,
    target: String,
    max_rows: u64,
    max_writes: u64,
}

pub fn validate(value: Value, dataset: bool) -> Result<Value, String> {
    if dataset {
        let config: DatasetConfig = serde_json::from_value(value).map_err(|e| e.to_string())?;
        let unique: std::collections::HashSet<_> = config.features.iter().collect();
        if config.api_version != VERSION
            || config.kind != "bootstrap-dataset"
            || !name(&config.name)
            || !name(&config.target)
            || config.features.is_empty()
            || config.features.len() > 32
            || unique.len() != config.features.len()
            || config
                .features
                .iter()
                .any(|f| !name(f) || f == &config.target)
            || !(1..=1024).contains(&config.max_rows)
            || !(1..=10000).contains(&config.max_writes)
        {
            return Err("invalid bounded bootstrap dataset configuration".into());
        }
        serde_json::to_value(config).map_err(|e| e.to_string())
    } else {
        let config: DeploymentConfig = serde_json::from_value(value).map_err(|e| e.to_string())?;
        let spec = &config.prediction;
        if config.api_version != VERSION
            || config.kind != "deployment"
            || !name(&config.name)
            || spec.api_version != VERSION
            || !name(&spec.dataset)
            || !name(&spec.target)
            || spec.context != "exact"
            || spec.model.repository != "general-forecast/tabicl-v2-classifier"
            || spec.model.revision.len() != 40
            || !spec
                .model
                .revision
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || !spec.max_update_lag_seconds.is_finite()
            || !(0.0..=86400.0).contains(&spec.max_update_lag_seconds)
        {
            return Err("invalid native deployment prediction specification".into());
        }
        serde_json::to_value(config).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_cannot_inject_paths() {
        for bad in ["", "A", "a/b", "a?x", "a%20", "../a", "a_b"] {
            assert!(!name(bad));
        }
        assert!(name("fraud-prod"));
    }
}
