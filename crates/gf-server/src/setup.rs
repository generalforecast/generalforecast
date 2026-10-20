//! `gf-server init` and `gf-server check`: a starter configuration for one
//! machine, and a look at everything a server needs before it starts.

use std::path::{Path, PathBuf};

use zeroize::Zeroize;

use crate::config::Config;
use crate::wal::{self, EncryptionKey, Keys, Log, LogError, Replay, TableDefinition};

/// What a starter configuration says.
pub struct Starter {
    pub listen: String,
    pub data: PathBuf,
    /// The Python that runs the workers.
    pub python: String,
    /// Model names and their specs; the built-in TabICL models if empty.
    pub models: Vec<(String, String)>,
    pub token: String,
    /// A file holding the log encryption key; `GF_LOG_KEY` if None.
    pub key_file: Option<PathBuf>,
}

/// Random bytes as hex, for tokens and keys.
///
/// # Errors
///
/// If the system gives no randomness.
pub fn random_hex(bytes: usize) -> Result<String, String> {
    let mut buffer = vec![0u8; bytes];
    getrandom::fill(&mut buffer).map_err(|error| error.to_string())?;
    Ok(buffer.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Writes a file only its owner may read, as a configuration holding tokens is.
///
/// # Errors
///
/// If the file cannot be written.
pub fn write_private(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)?;
    // A file that existed keeps its mode through `open`, so it is set here too.
    file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    file.write_all(contents)?;
    file.sync_all()
}

/// A configuration for one server on one machine, as `init` writes it.
#[must_use]
pub fn starter_config(starter: &Starter) -> String {
    let models: Vec<String> = if starter.models.is_empty() {
        vec![
            // Served under their repository ids, as local tables and the
            // estimators name them.
            "\"general-forecast/tabicl-v2-classifier\" = \"general-forecast/tabicl-v2-classifier\""
                .to_string(),
            "\"general-forecast/tabicl-v2-regressor\" = \"general-forecast/tabicl-v2-regressor\""
                .to_string(),
        ]
    } else {
        starter
            .models
            .iter()
            .map(|(name, spec)| format!("{name:?} = {spec:?}"))
            .collect()
    };
    let (key_note, key) = match &starter.key_file {
        Some(path) => (
            String::new(),
            format!("encryption_key_file = {:?}", path.display().to_string()),
        ),
        None => (
            ", with the log encryption key in\n# the environment: export GF_LOG_KEY=$(openssl rand -hex 32)".to_string(),
            "encryption_key_env = \"GF_LOG_KEY\"".to_string(),
        ),
    };
    format!(
        r#"# A generalforecast server on this machine. Check it with `gf-server check --config`
# and start it with `gf-server run --config`{key_note}

listen = {listen:?}
data = {data:?}                     # table logs, replayed on restart
tokens = [{token:?}]
{key}

# Beyond this machine, serve over TLS and listen on 0.0.0.0:8815.
# [tls]
# certificate = "/etc/generalforecast/server.pem"
# key = "/etc/generalforecast/server.key"

# metrics_listen = "127.0.0.1:9090"   # /metrics, /healthz, /readyz
# timing_log = "/var/log/generalforecast/timings.jsonl"   # a JSON line per prediction request

[[workers]]
command = [{python:?}, "-m", "generalforecast.worker_process"]
per_gpu = true                      # one worker per visible NVIDIA GPU, else one

[workers.models]
{models}
"#,
        listen = starter.listen,
        data = starter.data.display().to_string(),
        token = starter.token,
        python = starter.python,
        models = models.join("\n"),
    )
}

/// The key an environment variable holds, wiping the variable's hex once parsed.
///
/// # Errors
///
/// If the variable is unset or not 64 hex digits.
pub fn read_key(name: &str) -> Result<EncryptionKey, String> {
    let mut hex = std::env::var(name).map_err(|_| format!("set {name} to a log encryption key"))?;
    let key = EncryptionKey::from_hex(&hex);
    hex.zeroize();
    key
}

/// The key a file holds, as 64 hex digits, wiping the hex once parsed.
///
/// # Errors
///
/// If the file cannot be read or does not hold 64 hex digits.
pub fn read_key_file(path: &Path) -> Result<EncryptionKey, String> {
    let mut text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let key = EncryptionKey::from_hex(text.trim());
    text.zeroize();
    key.map_err(|error| format!("{}: {error}", path.display()))
}

/// Where a configuration's current key comes from, for messages.
fn key_source(config: &Config) -> String {
    match (&config.encryption_key_env, &config.encryption_key_file) {
        (Some(name), _) => name.clone(),
        (None, Some(path)) => path.display().to_string(),
        (None, None) => "no key".to_string(),
    }
}

/// The current and previous keys a configuration names.
///
/// # Errors
///
/// If any is unset or malformed.
pub fn keys(config: &Config) -> Result<Keys, String> {
    let current = match (&config.encryption_key_env, &config.encryption_key_file) {
        (Some(name), _) => read_key(name)?,
        (None, Some(path)) => read_key_file(path)?,
        (None, None) => return Err("the configuration names no log encryption key".to_string()),
    };
    Ok(Keys {
        current,
        previous: config
            .previous_key_envs
            .iter()
            .map(|name| read_key(name))
            .collect::<Result<_, _>>()?,
    })
}

/// What `check` found: one line per item, and whether any failed.
#[derive(Debug, Default)]
pub struct Report {
    pub lines: Vec<String>,
    pub failed: bool,
}

impl Report {
    fn ok(&mut self, what: impl Into<String>) {
        self.lines.push(format!("ok     {}", what.into()));
    }

    fn fail(&mut self, what: impl Into<String>) {
        self.failed = true;
        self.lines.push(format!("failed {}", what.into()));
    }

    fn result(&mut self, what: &str, result: Result<String, String>) {
        match result {
            Ok(detail) => self.ok(format!("{what}: {detail}")),
            Err(error) => self.fail(format!("{what}: {error}")),
        }
    }
}

fn readable(path: &Path) -> Result<String, String> {
    std::fs::metadata(path)
        .map(|_| path.display().to_string())
        .map_err(|error| format!("{}: {error}", path.display()))
}

/// Whether a worker's program can be found: a path, or a name on PATH.
fn program(command: &[String]) -> Result<String, String> {
    let Some(program) = command.first() else {
        return Err("the command is empty".to_string());
    };
    let path = Path::new(program);
    if path.components().count() > 1 {
        return readable(path);
    }
    let found = std::env::var_os("PATH")
        .map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
        .unwrap_or_default()
        .into_iter()
        .map(|directory| directory.join(program))
        .find(|candidate| candidate.is_file());
    found
        .map(|path| path.display().to_string())
        .ok_or_else(|| format!("{program} is not on PATH"))
}

/// Looks at the keys, TLS files, data directory, worker programs and table
/// logs a configuration needs, without starting anything.
#[must_use]
pub fn check(config: &Config) -> Report {
    let mut report = Report::default();
    let keys = keys(config);
    report.result(
        "log encryption key",
        keys.as_ref()
            .map(|keys| {
                format!(
                    "{} (and {} previous)",
                    key_source(config),
                    keys.previous.len()
                )
            })
            .map_err(Clone::clone),
    );
    match &config.tls {
        Some(tls) => {
            report.result("tls certificate", readable(&tls.certificate));
            report.result("tls key", readable(&tls.key));
            if let Some(client_ca) = &tls.client_ca {
                report.result("tls client ca", readable(client_ca));
            }
        }
        None => report.ok("tls: off (fine on this machine; beyond it, add [tls])"),
    }
    let tables = config.data.join("tables");
    let writable = std::fs::create_dir_all(&tables)
        .and_then(|()| {
            let probe = tables.join(".gf-check");
            std::fs::write(&probe, b"")?;
            std::fs::remove_file(&probe)
        })
        .map(|()| tables.display().to_string())
        .map_err(|error| format!("{}: {error}", tables.display()));
    report.result("data directory", writable);
    for (index, worker) in config.workers.iter().enumerate() {
        report.result(
            &format!("worker {index} program"),
            program(&worker.command).map(|path| {
                format!(
                    "{path} serving {}",
                    worker.models.keys().cloned().collect::<Vec<_>>().join(", ")
                )
            }),
        );
    }
    if let Ok(keys) = &keys
        && let Ok(logs) = table_logs(&tables)
    {
        for path in logs {
            let opened = read_log(&path, keys)
                .map(|(definition, records, rotate)| {
                    let served = config
                        .workers
                        .iter()
                        .any(|worker| worker.models.contains_key(&definition.model));
                    let model = if served {
                        definition.model.clone()
                    } else {
                        format!("{}, which no worker serves", definition.model)
                    };
                    format!(
                        "{} ({records} records, model {model}{})",
                        definition.name,
                        if rotate {
                            ", to rewrite under the current key"
                        } else {
                            ""
                        }
                    )
                })
                .map_err(|error| format!("{}: {error}", path.display()));
            report.result("table log", opened);
        }
    }
    report
}

/// Reads a log through without changing it: its definition, how many records
/// it holds, and whether it awaits rewriting under the current key.
fn read_log(path: &Path, keys: &Keys) -> Result<(TableDefinition, usize, bool), LogError> {
    let mut replay = Log::open_read_only(path, keys)?;
    let definition = replay.definition().clone();
    let rotate = replay.needs_rotation();
    let mut records = 0;
    for record in replay.by_ref() {
        record?;
        records += 1;
    }
    replay.finish()?;
    Ok((definition, records, rotate))
}

/// Opens a table's log for a server to restore at startup, or None to skip it.
///
/// A log that cannot be read for now, whose head says it lost records, or
/// whose table names a retired strategy stays in place, so nothing is lost and
/// a client cannot create a table over it; a damaged log is set aside, so a
/// replica can be copied back.
///
/// # Errors
///
/// If the log is sealed under a key the server was not given: starting
/// without it would leave the table out until a start that has it.
pub fn open_log(path: &Path, keys: &Keys) -> Result<Option<Replay>, LogError> {
    match Log::open(path, keys) {
        Ok(replay) => Ok(Some(replay)),
        Err(error @ LogError::Key { .. }) => Err(error),
        Err(error) if error.damaged() => {
            wal::set_aside_damaged(path, &error);
            Ok(None)
        }
        Err(error) => {
            tracing::error!(log = %path.display(), "not serving a table whose log does not open: {error}");
            Ok(None)
        }
    }
}

/// The table logs in a data directory's `tables`, in order.
///
/// # Errors
///
/// If the directory cannot be read.
pub fn table_logs(tables: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut logs: Vec<PathBuf> = std::fs::read_dir(tables)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|extension| extension == "log"))
        .collect();
    logs.sort();
    Ok(logs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn starter(directory: &Path) -> Starter {
        Starter {
            listen: "127.0.0.1:8815".to_string(),
            data: directory.join("data"),
            python: "python".to_string(),
            models: Vec::new(),
            token: "secret".to_string(),
            key_file: None,
        }
    }

    fn tempdir() -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "gf-setup-{}-{}",
            std::process::id(),
            random_hex(4).unwrap()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[test]
    fn the_starter_configuration_loads_and_checks() {
        let directory = tempdir();
        let path = directory.join("generalforecast.toml");
        std::fs::write(&path, starter_config(&starter(&directory))).unwrap();
        let mut config = Config::load(&path).unwrap();
        assert_eq!(config.tokens.len(), 1);
        assert_eq!(config.workers[0].models.len(), 2);

        // Without the key, the check fails and still reports the rest.
        unsafe { std::env::remove_var("GF_LOG_KEY") };
        let without = check(&config);
        assert!(without.failed);
        assert!(without.lines[0].starts_with("failed log encryption key"));
        assert!(directory.join("data").join("tables").is_dir());
        unsafe { std::env::set_var("GF_LOG_KEY", "ab".repeat(32)) };
        let with = check(&config);
        assert!(with.lines[0].starts_with("ok     log encryption key"));
        // The worker's program is python, which the test machine may lack.
        assert_eq!(
            with.failed,
            with.lines.iter().any(|line| line.contains("not on PATH"))
        );
        // Workers see neither the current key nor the ones it replaced.
        config.previous_key_envs = vec!["GF_OLD_KEY".to_string()];
        assert_eq!(config.key_envs(), ["GF_LOG_KEY", "GF_OLD_KEY"]);
        // A short token, and a file others may read, draw warnings.
        assert!(config.warnings(&path)[0].contains("shorter than"));
    }

    #[test]
    fn a_key_file_keeps_the_key_out_of_the_environment() {
        use std::os::unix::fs::PermissionsExt;
        let directory = tempdir();
        let key = directory.join("key");
        write_private(&key, "cd".repeat(32).as_bytes()).unwrap();
        let mut with_file = starter(&directory);
        with_file.key_file = Some(key.clone());
        let path = directory.join("generalforecast.toml");
        std::fs::write(&path, starter_config(&with_file)).unwrap();

        let config = Config::load(&path).unwrap();
        assert!(keys(&config).is_ok());
        assert!(config.key_envs().is_empty());
        std::fs::set_permissions(&key, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(
            config
                .warnings(&path)
                .iter()
                .any(|warning| warning.contains("chmod 600"))
        );
    }

    #[test]
    fn configurations_refuse_empty_tokens_and_plaintext_beyond_this_machine() {
        let directory = tempdir();
        let path = directory.join("generalforecast.toml");
        let base = starter_config(&starter(&directory));
        let load = |text: String| {
            std::fs::write(&path, text).unwrap();
            Config::load(&path)
        };

        let empty = load(base.replace("tokens = [\"secret\"]", "tokens = [\"\"]"));
        assert!(empty.unwrap_err().contains("empty"));
        let open = base.replace("127.0.0.1:8815", "0.0.0.0:8815");
        assert!(load(open.clone()).unwrap_err().contains("without TLS"));
        assert!(load(format!("allow_plaintext = true\n{open}")).is_ok());
        let both = base.replace(
            "encryption_key_env",
            "encryption_key_file = \"key\"\nencryption_key_env",
        );
        assert!(load(both).unwrap_err().contains("encryption_key_env or"));
    }

    #[test]
    fn timers_are_off_unless_set() {
        let directory = tempdir();
        let path = directory.join("generalforecast.toml");
        let base = starter_config(&starter(&directory));
        std::fs::write(&path, &base).unwrap();
        assert!(!Config::load(&path).unwrap().timers);
        std::fs::write(&path, format!("timers = true\n{base}")).unwrap();
        assert!(Config::load(&path).unwrap().timers);
    }

    #[test]
    fn placement_settings_are_checked() {
        let directory = tempdir();
        let path = directory.join("generalforecast.toml");
        let base = starter_config(&starter(&directory));
        let load = |placement: &str| {
            std::fs::write(&path, format!("{base}\n[placement]\n{placement}\n")).unwrap();
            Config::load(&path)
        };

        let defaults = load("assign = \"load\"").unwrap().placement;
        assert_eq!(defaults.rebalance_seconds, 0.0);
        assert_eq!(
            defaults.churn_budget,
            gf_assign::Policy::default().churn_budget
        );
        assert!(load("rebalance_seconds = 30").is_ok());
        assert!(
            load("rebalance_seconds = -1")
                .unwrap_err()
                .contains("rebalance_seconds")
        );
        assert!(
            load("churn_budget = 0")
                .unwrap_err()
                .contains("churn_budget")
        );
        assert!(
            load("churn_budget = 1.5")
                .unwrap_err()
                .contains("churn_budget")
        );
        assert!(
            load("imbalance_floor = 0.9")
                .unwrap_err()
                .contains("imbalance_floor")
        );
        assert!(load("policy = { mode = \"free\" }").is_err());
    }
}
