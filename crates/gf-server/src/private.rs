//! A Flight `Service` over an embedding process's shards, for that process
//! alone: on a Unix socket in an owner-only directory where there are Unix
//! sockets, else on a loopback port, behind a random token no configuration
//! names.

use std::collections::HashMap;
use std::error::Error;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;

use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use tonic::transport::server::Router;

use crate::auth::{TokenConfig, Tokens};
use crate::metrics::Metrics;
use crate::service::Service;
use crate::setup;
use crate::shard::Command;

/// A running private service; dropping it stops it.
pub struct Private {
    /// Where a Flight client connects, `grpc+unix://` or `grpc+tcp://`.
    pub location: String,
    /// The bearer token the service accepts, and no other.
    pub token: String,
    stop: watch::Sender<bool>,
    task: JoinHandle<()>,
}

impl Drop for Private {
    fn drop(&mut self) {
        let _ = self.stop.send(true);
        self.task.abort();
    }
}

/// Starts the service over `shards`, with its socket under `directory`.
///
/// # Errors
///
/// If the token cannot be drawn or the endpoint cannot be bound.
pub async fn serve(
    shards: Vec<mpsc::Sender<Command>>,
    metrics: Arc<Metrics>,
    directory: &Path,
) -> Result<Private, Box<dyn Error + Send + Sync>> {
    let token = setup::random_hex(32)?;
    let tokens = Tokens::new(&[TokenConfig::Any(token.clone())]);
    let service = Service::new(shards, HashMap::new(), metrics);
    let router = tonic::transport::Server::builder()
        .add_service(service.into_server(tokens, Tokens::default()));
    let (stop, mut stopped) = watch::channel(false);
    let shutdown = async move {
        let _ = stopped.wait_for(|stop| *stop).await;
    };
    let (location, task) = listen(router, directory, shutdown).await?;
    Ok(Private {
        location,
        token,
        stop,
        task,
    })
}

#[cfg(unix)]
async fn listen(
    router: Router,
    directory: &Path,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(String, JoinHandle<()>), Box<dyn Error + Send + Sync>> {
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    let directory = directory.join("flight");
    std::fs::DirBuilder::new().mode(0o700).create(&directory)?;
    let path = directory.join("socket");
    let listener = tokio::net::UnixListener::bind(&path)?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    let incoming = tokio_stream::wrappers::UnixListenerStream::new(listener);
    let task = tokio::spawn(async move {
        if let Err(error) = router
            .serve_with_incoming_shutdown(incoming, shutdown)
            .await
        {
            tracing::error!("private service: {error}");
        }
    });
    Ok((format!("grpc+unix://{}", path.display()), task))
}

#[cfg(not(unix))]
async fn listen(
    router: Router,
    _directory: &Path,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(String, JoinHandle<()>), Box<dyn Error + Send + Sync>> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
    let task = tokio::spawn(async move {
        if let Err(error) = router
            .serve_with_incoming_shutdown(incoming, shutdown)
            .await
        {
            tracing::error!("private service: {error}");
        }
    });
    Ok((format!("grpc+tcp://{address}"), task))
}
