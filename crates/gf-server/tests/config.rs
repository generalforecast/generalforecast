// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Configuration files as predictors write them.

use gf_server::config::{Config, KeysConfig};

fn load(name: &str, http: &str) -> Config {
    let path = std::env::temp_dir().join(format!("gf-config-{}-{name}.toml", std::process::id()));
    std::fs::write(
        &path,
        format!(
            "listen = \"127.0.0.1:0\"\ndata = \"/tmp/gf\"\ntokens = [\"{}\"]\n\
             encryption_key_env = \"GF_LOG_KEY\"\n\n{http}\n\n[[workers]]\n\
             command = [\"python\"]\nper_gpu = true\n\n[workers.models]\n\
             tabicl-v2-classifier = \"tabicl-v2-classifier\"\n",
            "f".repeat(64)
        ),
    )
    .unwrap();
    let config = Config::load(&path);
    std::fs::remove_file(&path).unwrap();
    config.unwrap()
}

#[test]
fn a_deployment_key_is_scoped_and_its_front_answers_exact_hosts() {
    let config = load(
        "scoped",
        "[http]\nlisten = \"0.0.0.0:8000\"\nmodels = \"/opt/gf/models.json\"\n\
         surface = \"remote\"\nhosts = [\"unit--gf-byoc.modal.run\"]\n\n[http.keys]\n\
         kind = \"static\"\nenv = \"GF_MODAL_INFERENCE_TOKEN\"\nproject = \"demo\"\n\
         permissions = [\"predictors:predict\"]",
    );
    let http = config.http.unwrap();
    assert_eq!(http.hosts, ["unit--gf-byoc.modal.run"]);
    let KeysConfig::Static {
        env,
        project,
        permissions,
    } = http.keys
    else {
        panic!("expected static keys");
    };
    assert_eq!(env, "GF_MODAL_INFERENCE_TOKEN");
    assert_eq!(project.as_deref(), Some("demo"));
    assert_eq!(permissions, Some(vec!["predictors:predict".to_string()]));
}

#[test]
fn a_static_key_without_scope_is_the_operator_and_any_host_answers() {
    let config = load(
        "operator",
        "[http]\nlisten = \"127.0.0.1:8000\"\nmodels = \"/opt/gf/models.json\"\n\
         surface = \"self-hosted\"\n\n\
         [http.keys]\nkind = \"static\"\nenv = \"GF_API_KEY\"",
    );
    let http = config.http.unwrap();
    assert!(http.hosts.is_empty());
    assert!(matches!(
        http.keys,
        KeysConfig::Static {
            project: None,
            permissions: None,
            ..
        }
    ));
}
