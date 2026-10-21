// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! The front door's upstream: every route the front does not own streams,
//! unchanged, to the control plane listening behind it on loopback.
//!
//! Method, target, headers and body pass through as they came, less the
//! hop-by-hop headers, with `X-Forwarded-*` describing the client. Redirects
//! and `Set-Cookie` come back as the upstream sent them, so browser sign-in
//! and OAuth callbacks work through the front.

use std::net::{IpAddr, SocketAddr};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use axum::body::Body;
use axum::extract::Request;
use axum::http::{HeaderMap, HeaderName, HeaderValue, header};
use axum::response::Response;
use futures::StreamExt;

use crate::http::ApiError;

/// How long a readiness probe waits for the upstream's own.
const READY_TIMEOUT: Duration = Duration::from_secs(5);

/// Headers that describe one connection, never forwarded across the proxy.
const HOP_BY_HOP: [&str; 8] = [
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
];

/// Headers that describe the client; kept only from a trusted proxy.
const FORWARDING: [&str; 5] = [
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
];

/// An address range, `10.0.0.0/8` or a single address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cidr {
    network: IpAddr,
    prefix: u8,
}

impl FromStr for Cidr {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let invalid = || format!("invalid address range {text:?}");
        let (address, prefix) = text.split_once('/').unwrap_or((text, ""));
        let network: IpAddr = address.parse().map_err(|_| invalid())?;
        let bits = if network.is_ipv4() { 32 } else { 128 };
        let prefix = if prefix.is_empty() {
            bits
        } else {
            prefix.parse::<u8>().map_err(|_| invalid())?
        };
        if prefix > bits {
            return Err(invalid());
        }
        Ok(Self { network, prefix })
    }
}

impl Cidr {
    #[must_use]
    pub fn contains(&self, address: IpAddr) -> bool {
        let address = match address {
            IpAddr::V6(v6) => v6.to_ipv4_mapped().map_or(address, IpAddr::V4),
            IpAddr::V4(_) => address,
        };
        match (self.network, address) {
            (IpAddr::V4(network), IpAddr::V4(address)) => {
                let mask = u32::MAX
                    .checked_shl(32 - u32::from(self.prefix))
                    .unwrap_or(0);
                u32::from(network) & mask == u32::from(address) & mask
            }
            (IpAddr::V6(network), IpAddr::V6(address)) => {
                let mask = u128::MAX
                    .checked_shl(128 - u32::from(self.prefix))
                    .unwrap_or(0);
                u128::from(network) & mask == u128::from(address) & mask
            }
            _ => false,
        }
    }
}

/// The origin behind the front, and how far it is trusted.
pub struct Upstream {
    client: reqwest::Client,
    origin: String,
    timeout: Duration,
    max_body: u64,
    trusted: Vec<Cidr>,
}

/// `url` as a bare `http(s)://host:port` origin, without its trailing slash.
///
/// # Errors
/// When `url` has credentials, a path, a query or a fragment.
pub fn origin(url: &str) -> Result<String, String> {
    let parsed = reqwest::Url::parse(url).map_err(|error| format!("{url}: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.path() != "/"
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(format!(
            "must be an http(s)://host:port origin, not {url:?}"
        ));
    }
    Ok(parsed.as_str().trim_end_matches('/').to_owned())
}

impl Upstream {
    /// An upstream at `url`, an `http(s)://host:port` origin with no path.
    ///
    /// # Errors
    /// When `url` is not a bare origin or the client cannot be built.
    pub fn new(
        url: &str,
        timeout: Duration,
        max_body: u64,
        trusted: Vec<Cidr>,
    ) -> Result<Self, String> {
        let origin = origin(url).map_err(|error| format!("upstream {error}"))?;
        if timeout.is_zero() {
            return Err("upstream_timeout_seconds must be positive".into());
        }
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .no_proxy()
            .connect_timeout(timeout)
            .read_timeout(timeout)
            .build()
            .map_err(|error| format!("upstream client: {error}"))?;
        Ok(Self {
            client,
            origin,
            timeout,
            max_body,
            trusted,
        })
    }

    /// Whether the upstream answers its own `/readyz` with 200, so the
    /// single listener is ready only when what stands behind it is.
    pub async fn ready(&self) -> bool {
        let probe = self
            .client
            .get(format!("{}/readyz", self.origin))
            .timeout(READY_TIMEOUT.min(self.timeout))
            .send();
        matches!(probe.await, Ok(answer) if answer.status() == reqwest::StatusCode::OK)
    }

    /// Streams `request` upstream and its answer back.
    ///
    /// # Errors
    /// `CapacityExceeded` (413) past the body bound, `UpstreamUnavailable`
    /// (502) when the upstream cannot be reached or fails, and
    /// `UpstreamTimeout` (504) when it does not answer in time.
    pub async fn forward(
        &self,
        request: Request,
        peer: Option<SocketAddr>,
    ) -> Result<Response, ApiError> {
        let (parts, body) = request.into_parts();
        let too_large = || {
            ApiError::new(
                "CapacityExceeded",
                format!("request body exceeds {} bytes", self.max_body),
                413,
            )
        };
        let length = match parts.headers.get(header::CONTENT_LENGTH) {
            Some(value) => Some(
                value
                    .to_str()
                    .ok()
                    .and_then(|length| length.parse::<u64>().ok())
                    .ok_or_else(|| ApiError::new("ValidationError", "invalid HTTP message", 400))?,
            ),
            None => None,
        };
        if length.is_some_and(|length| length > self.max_body) {
            return Err(too_large());
        }
        let target = parts
            .uri
            .path_and_query()
            .map_or("/", |target| target.as_str());
        if !target.starts_with('/') {
            return Err(ApiError::new(
                "ValidationError",
                "invalid request target",
                400,
            ));
        }
        let chunked = parts.headers.contains_key(header::TRANSFER_ENCODING);
        let headers = forwarded(&parts.headers, peer, &self.trusted);
        let mut builder = self
            .client
            .request(parts.method, format!("{}{target}", self.origin))
            .headers(headers);
        let exceeded = Arc::new(AtomicBool::new(false));
        if length.is_some_and(|length| length > 0) || chunked {
            let limit = self.max_body;
            let flag = exceeded.clone();
            let mut seen = 0u64;
            let stream = body.into_data_stream().map(move |chunk| {
                let chunk = chunk.map_err(std::io::Error::other)?;
                seen += chunk.len() as u64;
                if seen > limit {
                    flag.store(true, Ordering::Relaxed);
                    return Err(std::io::Error::other("request body too large"));
                }
                Ok(chunk)
            });
            builder = builder.body(reqwest::Body::wrap_stream(stream));
        }
        let answered = tokio::time::timeout(self.timeout, builder.send()).await;
        let timed_out = || {
            ApiError::new(
                "UpstreamTimeout",
                "the upstream did not answer in time",
                504,
            )
        };
        let answer = match answered {
            Err(_) => return Err(timed_out()),
            Ok(Err(_)) if exceeded.load(Ordering::Relaxed) => return Err(too_large()),
            Ok(Err(error)) if error.is_timeout() => return Err(timed_out()),
            Ok(Err(_)) => {
                return Err(ApiError::new(
                    "UpstreamUnavailable",
                    "the upstream is unavailable",
                    502,
                ));
            }
            Ok(Ok(answer)) => answer,
        };
        let mut response = Response::new(Body::empty());
        *response.status_mut() = answer.status();
        *response.headers_mut() = without_hop_by_hop(answer.headers());
        *response.body_mut() = Body::from_stream(answer.bytes_stream());
        Ok(response)
    }
}

/// `headers` less the hop-by-hop ones and any their `Connection` names.
fn without_hop_by_hop(headers: &HeaderMap) -> HeaderMap {
    let named: Vec<String> = headers
        .get_all(header::CONNECTION)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .map(|name| name.trim().to_ascii_lowercase())
        .collect();
    let mut kept = HeaderMap::with_capacity(headers.len());
    for (name, value) in headers {
        let name_text = name.as_str();
        if HOP_BY_HOP.contains(&name_text) || named.iter().any(|named| named == name_text) {
            continue;
        }
        kept.append(name.clone(), value.clone());
    }
    kept
}

/// The headers sent upstream: the client's, less hop-by-hop ones, with
/// `X-Forwarded-For`, `-Proto` and `-Host` describing the client. A peer
/// outside `trusted` cannot claim forwarding headers of its own.
fn forwarded(headers: &HeaderMap, peer: Option<SocketAddr>, trusted: &[Cidr]) -> HeaderMap {
    let mut kept = without_hop_by_hop(headers);
    let trusted_peer =
        peer.is_some_and(|peer| trusted.iter().any(|range| range.contains(peer.ip())));
    if !trusted_peer {
        for name in FORWARDING {
            kept.remove(name);
        }
    }
    if let Some(peer) = peer {
        let prior: Vec<&str> = kept
            .get_all("x-forwarded-for")
            .iter()
            .filter_map(|value| value.to_str().ok())
            .collect();
        let chain = if prior.is_empty() {
            peer.ip().to_string()
        } else {
            format!("{}, {}", prior.join(", "), peer.ip())
        };
        if let Ok(value) = HeaderValue::from_str(&chain) {
            kept.insert(HeaderName::from_static("x-forwarded-for"), value);
        }
    }
    if !kept.contains_key("x-forwarded-host")
        && let Some(host) = headers.get(header::HOST)
    {
        kept.insert(HeaderName::from_static("x-forwarded-host"), host.clone());
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer(address: &str) -> Option<SocketAddr> {
        Some(SocketAddr::new(address.parse().unwrap(), 40000))
    }

    #[test]
    fn ranges_parse_and_contain() {
        let range: Cidr = "10.0.0.0/8".parse().unwrap();
        assert!(range.contains("10.2.3.4".parse().unwrap()));
        assert!(!range.contains("11.0.0.1".parse().unwrap()));
        assert!(range.contains("::ffff:10.0.0.1".parse().unwrap()));
        let one: Cidr = "127.0.0.1".parse().unwrap();
        assert!(one.contains("127.0.0.1".parse().unwrap()));
        assert!(!one.contains("127.0.0.2".parse().unwrap()));
        let all: Cidr = "0.0.0.0/0".parse().unwrap();
        assert!(all.contains("8.8.8.8".parse().unwrap()));
        let v6: Cidr = "fd00::/8".parse().unwrap();
        assert!(v6.contains("fd12::1".parse().unwrap()));
        assert!(!v6.contains("10.0.0.1".parse().unwrap()));
        for bad in ["10.0.0.0/33", "nope", "10.0.0.0/x", "::/129"] {
            assert!(bad.parse::<Cidr>().is_err(), "{bad}");
        }
    }

    #[test]
    fn hop_by_hop_headers_and_those_connection_names_are_dropped() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "connection",
            HeaderValue::from_static("keep-alive, x-private"),
        );
        headers.insert("keep-alive", HeaderValue::from_static("timeout=5"));
        headers.insert("x-private", HeaderValue::from_static("1"));
        headers.insert("te", HeaderValue::from_static("trailers"));
        headers.insert("upgrade", HeaderValue::from_static("websocket"));
        headers.append("set-cookie", HeaderValue::from_static("a=1"));
        headers.append("set-cookie", HeaderValue::from_static("b=2"));
        headers.insert(
            "location",
            HeaderValue::from_static("/auth/callback?code=x"),
        );
        let kept = without_hop_by_hop(&headers);
        assert_eq!(kept.len(), 3);
        assert_eq!(kept.get_all("set-cookie").iter().count(), 2);
        assert!(kept.contains_key("location"));
    }

    #[test]
    fn untrusted_peers_cannot_claim_forwarding_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(header::HOST, HeaderValue::from_static("gf.example"));
        headers.insert("x-forwarded-for", HeaderValue::from_static("1.2.3.4"));
        headers.insert("x-forwarded-proto", HeaderValue::from_static("https"));
        headers.insert("forwarded", HeaderValue::from_static("for=1.2.3.4"));
        let sent = forwarded(&headers, peer("203.0.113.9"), &[]);
        assert_eq!(sent["x-forwarded-for"], "203.0.113.9");
        assert!(!sent.contains_key("x-forwarded-proto"));
        assert!(!sent.contains_key("forwarded"));
        assert_eq!(sent["x-forwarded-host"], "gf.example");
        assert_eq!(sent[header::HOST], "gf.example");

        let trusted = ["35.191.0.0/16".parse().unwrap()];
        let sent = forwarded(&headers, peer("35.191.1.2"), &trusted);
        assert_eq!(sent["x-forwarded-for"], "1.2.3.4, 35.191.1.2");
        assert_eq!(sent["x-forwarded-proto"], "https");
    }

    #[test]
    fn upstreams_are_bare_origins() {
        let timeout = Duration::from_secs(1);
        assert!(Upstream::new("http://127.0.0.1:8001", timeout, 1, vec![]).is_ok());
        assert!(Upstream::new("http://127.0.0.1:8001/", timeout, 1, vec![]).is_ok());
        for bad in [
            "ftp://127.0.0.1",
            "http://127.0.0.1:8001/console",
            "http://u:p@127.0.0.1:8001",
            "http://127.0.0.1:8001?x=1",
            "127.0.0.1:8001",
        ] {
            assert!(Upstream::new(bad, timeout, 1, vec![]).is_err(), "{bad}");
        }
        assert!(Upstream::new("http://127.0.0.1:1", Duration::ZERO, 1, vec![]).is_err());
    }
}
