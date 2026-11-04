// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! A limit on how long a gRPC call may take, answered as `DEADLINE_EXCEEDED`.
//!
//! tonic's own `Server::timeout` answers a call past its limit with
//! `CANCELLED`, the status for a call its caller gave up on. Arrow's Python
//! client reads `CANCELLED` as an interrupt: `FlightStreamReader.read_all`
//! raises a `KeyboardInterrupt` in the process's main thread, whichever
//! thread made the call, which ends a program that only made one slow call.
//! A server that gives up on a call is past a deadline, so this layer says so.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use tonic::Status;
use tonic::codegen::http;
use tower::{Layer, Service};

/// Ends each call that has not answered within `limit` with `DEADLINE_EXCEEDED`.
#[derive(Clone, Copy, Debug)]
pub struct DeadlineLayer {
    limit: Duration,
    /// What sets the limit, named in the status, such as a configuration key.
    setting: &'static str,
}

impl DeadlineLayer {
    /// A layer ending calls past `limit`, which `setting` names, such as the key that sets it.
    pub fn new(limit: Duration, setting: &'static str) -> Self {
        Self { limit, setting }
    }
}

impl<S> Layer<S> for DeadlineLayer {
    type Service = Deadline<S>;

    fn layer(&self, inner: S) -> Self::Service {
        Deadline {
            inner,
            limit: self.limit,
            setting: self.setting,
        }
    }
}

/// A service whose calls past their limit answer `DEADLINE_EXCEEDED`; see `DeadlineLayer`.
#[derive(Clone, Debug)]
pub struct Deadline<S> {
    inner: S,
    limit: Duration,
    setting: &'static str,
}

impl<S, Request, Body> Service<Request> for Deadline<S>
where
    S: Service<Request, Response = http::Response<Body>>,
    S::Future: Send + 'static,
    Body: Default,
{
    type Response = http::Response<Body>;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(context)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        let (limit, setting) = (self.limit, self.setting);
        let answer = self.inner.call(request);
        Box::pin(async move {
            match tokio::time::timeout(limit, answer).await {
                Ok(answered) => answered,
                Err(_) => Ok(Status::deadline_exceeded(format!(
                    "the call took longer than the server's {setting} of {} s",
                    limit.as_secs_f64()
                ))
                .into_http()),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;
    use tower::ServiceExt;

    fn status(response: &http::Response<String>) -> Option<&str> {
        response
            .headers()
            .get("grpc-status")
            .and_then(|value| value.to_str().ok())
    }

    fn answering_after(
        wait: Duration,
    ) -> impl Service<(), Response = http::Response<String>, Error = Infallible, Future: Send> + Clone
    {
        tower::service_fn(move |()| async move {
            tokio::time::sleep(wait).await;
            Ok::<_, Infallible>(http::Response::new("answered".to_string()))
        })
    }

    #[tokio::test(start_paused = true)]
    async fn a_call_within_its_limit_answers_as_it_would() {
        let service = DeadlineLayer::new(Duration::from_secs(600), "request_timeout_seconds")
            .layer(answering_after(Duration::from_secs(599)));
        let response = service.oneshot(()).await.unwrap();
        assert_eq!(response.body(), "answered");
        assert_eq!(status(&response), None);
    }

    #[tokio::test(start_paused = true)]
    async fn a_call_past_its_limit_answers_deadline_exceeded_not_cancelled() {
        let service = DeadlineLayer::new(Duration::from_secs(600), "request_timeout_seconds")
            .layer(answering_after(Duration::from_secs(601)));
        let response = service.oneshot(()).await.unwrap();
        let code = tonic::Code::DeadlineExceeded as i32;
        assert_eq!(status(&response), Some(code.to_string().as_str()));
        let message = response.headers()["grpc-message"].to_str().unwrap();
        assert!(message.contains("request_timeout_seconds"), "{message}");
    }
}
