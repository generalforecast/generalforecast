// SPDX-FileCopyrightText: General Forecast, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Who a request's bearer key belongs to and what it may do: one static key,
//! or the control plane's introspection behind a short cache.

use super::*;

/// How long introspection answers are trusted: valid keys, and refused ones.
const POSITIVE_TTL: Duration = Duration::from_secs(30);
const NEGATIVE_TTL: Duration = Duration::from_secs(5);
/// The most introspection answers kept; past it the cache starts over.
const CACHED_KEYS: usize = 65_536;

/// Who a bearer key belongs to, and what it may do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Principal {
    /// The one project the key reaches, or None for every project.
    pub project: Option<String>,
    /// What the key may do, or None for everything.
    pub permissions: Option<Vec<String>>,
}

impl Principal {
    /// A principal that reaches everything, as a server without keys has.
    #[must_use]
    pub fn operator() -> Self {
        Self {
            project: None,
            permissions: None,
        }
    }

    pub(super) fn authorize(&self, project: &str, permission: &str) -> Result<(), ApiError> {
        let scoped = self.project.as_deref().is_none_or(|own| own == project);
        let allowed = self
            .permissions
            .as_ref()
            .is_none_or(|permissions| permissions.iter().any(|held| held == permission));
        if scoped && allowed {
            Ok(())
        } else {
            Err(ApiError::new(
                "Forbidden",
                "permission denied for this scope",
                403,
            ))
        }
    }
}

/// Checks the bearer key a request presents.
pub trait KeyStore: Send + Sync {
    /// The principal behind the `Authorization` header's value, if any.
    fn verify<'a>(
        &'a self,
        authorization: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Principal, ApiError>>;

    /// What clients must present: `bearer`, or `none` without a key.
    fn auth(&self) -> &'static str;
}

/// The key a `Bearer` header presents.
fn presented(authorization: Option<&str>) -> Result<&str, ApiError> {
    let (scheme, key) = authorization
        .unwrap_or_default()
        .split_once(' ')
        .unwrap_or_default();
    if scheme != "Bearer" || key.is_empty() || key.contains(' ') {
        return Err(ApiError::unauthorized("application bearer required"));
    }
    Ok(key)
}

/// One key for every project, as `gf serve` takes it from `GF_API_KEY`; or
/// none, where only this machine, or a network the operator trusts, reaches
/// the server.
pub struct Static {
    /// The key's digest: comparing digests takes the same time for every guess.
    key: Option<[u8; 32]>,
    /// Who the key is.
    principal: Principal,
}

impl Static {
    #[must_use]
    pub fn new(key: Option<&str>) -> Self {
        Self::scoped(key, Principal::operator())
    }

    /// One key that reaches only what `principal` may; without a key,
    /// every request is `principal`.
    #[must_use]
    pub fn scoped(key: Option<&str>, principal: Principal) -> Self {
        Self {
            key: key.map(digest),
            principal,
        }
    }
}

impl KeyStore for Static {
    fn verify<'a>(
        &'a self,
        authorization: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Principal, ApiError>> {
        Box::pin(async move {
            let Some(key) = &self.key else {
                return Ok(self.principal.clone());
            };
            let given = digest(presented(authorization)?);
            let differs = given
                .iter()
                .zip(key)
                .fold(0u8, |acc, (left, right)| acc | (left ^ right));
            if differs == 0 {
                Ok(self.principal.clone())
            } else {
                Err(ApiError::unauthorized("invalid API key"))
            }
        })
    }

    fn auth(&self) -> &'static str {
        if self.key.is_some() { "bearer" } else { "none" }
    }
}

/// The control plane's keys, checked through its introspection endpoint
/// under a service token, with answers cached briefly: valid keys for 30
/// seconds or until they expire, refused ones for 5. Concurrent requests
/// presenting one key share one introspection call.
pub struct Introspect {
    client: reqwest::Client,
    url: Arc<str>,
    token: Arc<str>,
    cache: Arc<Mutex<HashMap<[u8; 32], Cached>>>,
    asking: Arc<Mutex<HashMap<[u8; 32], Asking>>>,
}

/// An introspection answer, and until when it is trusted.
type Cached = (Option<Principal>, Instant);

/// An introspection call under way, which every request presenting its key
/// awaits. It runs on a task of its own, so it ends, caches its answer and
/// leaves the calls under way however many of its waiters go away.
type Asking = Shared<BoxFuture<'static, Result<(Option<Principal>, Duration), ApiError>>>;

/// What the introspection endpoint answers for a valid key.
#[derive(Deserialize)]
struct Introspected {
    /// Whether the key is live; a refused key answers `active: false` alone.
    active: bool,
    #[serde(default)]
    project: String,
    #[serde(default)]
    permissions: Vec<String>,
    /// When the key expires, in seconds since 1970, if it does.
    #[serde(default)]
    expires: Option<f64>,
}

impl Introspect {
    /// Checks keys against the control plane at `base`, such as
    /// `https://api.generalforecast.com`, under `token`.
    ///
    /// # Errors
    ///
    /// If the HTTP client cannot be built.
    pub fn new(base: &str, token: String) -> Result<Self, String> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|error| error.to_string())?;
        Ok(Self {
            client,
            url: format!("{}/internal/v1/keys:introspect", base.trim_end_matches('/')).into(),
            token: token.into(),
            cache: Arc::default(),
            asking: Arc::default(),
        })
    }

    fn cached(&self, verifier: &[u8; 32]) -> Option<Option<Principal>> {
        let cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        cache
            .get(verifier)
            .filter(|(_, until)| *until > Instant::now())
            .map(|(principal, _)| principal.clone())
    }

    /// The introspection call for `verifier`: the one under way, or a new
    /// one, which caches its answer and forgets itself once it ends.
    fn asking(&self, verifier: [u8; 32]) -> Asking {
        let mut asking = self.asking.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(call) = asking.get(&verifier) {
            return call.clone();
        }
        let (client, url, token) = (self.client.clone(), self.url.clone(), self.token.clone());
        let (cache, calls) = (self.cache.clone(), self.asking.clone());
        // Forgetting the call waits for this lock, so it follows the insert below.
        let task = tokio::spawn(async move {
            let answered = ask(client, url, token, verifier).await;
            if let Ok((principal, ttl)) = &answered {
                remember(&cache, verifier, principal.clone(), *ttl);
            }
            calls
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&verifier);
            answered
        });
        let call = async move {
            task.await.unwrap_or_else(|_| {
                Err(ApiError::new(
                    "CredentialUnavailable",
                    "cannot check API keys now",
                    503,
                ))
            })
        }
        .boxed()
        .shared();
        asking.insert(verifier, call.clone());
        call
    }
}

/// Caches an answer. A full cache sheds what has expired, then the refused
/// keys, which a caller presenting random keys fills it with, and only then
/// the valid keys it still holds.
fn remember(
    cache: &Mutex<HashMap<[u8; 32], Cached>>,
    verifier: [u8; 32],
    principal: Option<Principal>,
    ttl: Duration,
) {
    let mut cache = cache.lock().unwrap_or_else(PoisonError::into_inner);
    if cache.len() >= CACHED_KEYS {
        let now = Instant::now();
        cache.retain(|_, (_, until)| *until > now);
    }
    if cache.len() >= CACHED_KEYS {
        cache.retain(|_, (principal, _)| principal.is_some());
    }
    if cache.len() >= CACHED_KEYS {
        cache.clear();
    }
    cache.insert(verifier, (principal, Instant::now() + ttl));
}

/// Asks the control plane who `verifier`'s key is, and for how long to trust the answer.
pub(super) async fn ask(
    client: reqwest::Client,
    url: Arc<str>,
    token: Arc<str>,
    verifier: [u8; 32],
) -> Result<(Option<Principal>, Duration), ApiError> {
    let unavailable = || ApiError::new("CredentialUnavailable", "cannot check API keys now", 503);
    let response = client
        .post(&*url)
        .bearer_auth(&token)
        .json(&json!({"verifier": request::hex(&verifier)}))
        .send()
        .await
        .map_err(|_| unavailable())?;
    // Unknown keys answer 200 `{"active": false}`; any other status says
    // the control plane or this server's service token is at fault, not
    // the key, so it is neither cached nor answered as a bad key.
    if response.status() != StatusCode::OK {
        return Err(unavailable());
    }
    let found: Introspected = response.json().await.map_err(|_| unavailable())?;
    if !found.active {
        return Ok((None, NEGATIVE_TTL));
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let left = match found.expires {
        Some(expires) if expires <= now.as_secs_f64() => return Ok((None, NEGATIVE_TTL)),
        // An expiry too far off for a duration is as good as none.
        Some(expires) => {
            Duration::try_from_secs_f64(expires - now.as_secs_f64()).unwrap_or(POSITIVE_TTL)
        }
        None => POSITIVE_TTL,
    };
    let principal = Principal {
        project: Some(found.project),
        permissions: Some(found.permissions),
    };
    Ok((Some(principal), left.min(POSITIVE_TTL)))
}

impl KeyStore for Introspect {
    fn verify<'a>(
        &'a self,
        authorization: Option<&'a str>,
    ) -> BoxFuture<'a, Result<Principal, ApiError>> {
        Box::pin(async move {
            let verifier = digest(presented(authorization)?);
            let principal = match self.cached(&verifier) {
                Some(principal) => principal,
                None => self.asking(verifier).await?.0,
            };
            principal.ok_or_else(|| ApiError::unauthorized("invalid API key"))
        })
    }

    fn auth(&self) -> &'static str {
        "bearer"
    }
}
