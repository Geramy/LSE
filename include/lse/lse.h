/*
 * lse.h — the Lemon Seed Engine as an in-process library.
 *
 * One engine holds one loaded model (plus its MTP or DFlash2 drafter) on the
 * device set it opened. Requests are the OpenAI-shaped JSON requests the HTTP
 * server takes, answered with the same JSON the HTTP server returns; the HTTP
 * server itself (lse_http_start) is an optional adapter over the same engine,
 * and the lse-server executable is exactly lse_open + lse_http_start.
 *
 * Plain C, no C++ types: Swift and other C callers import it directly.
 *
 * Threading: every function may be called from any thread. Response callbacks
 * run on an engine-owned worker thread, one per request; a callback must not
 * call lse_close on its own engine.
 */
#ifndef LSE_LSE_H
#define LSE_LSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define LSE_API
#else
#define LSE_API __attribute__((visibility("default")))
#endif

/* Bumped whenever a struct below changes layout or a function changes
 * meaning. lse_config carries the value it was initialized with. */
#define LSE_ABI_VERSION 1u

/* Engine version string, e.g. "0.5.0". */
LSE_API const char *lse_version(void);
/* LSE_ABI_VERSION of the library actually linked. */
LSE_API uint32_t lse_abi_version(void);

/* Result codes. */
typedef enum lse_result {
  LSE_OK = 0,
  /* The configuration or the request is invalid; nothing was changed. The
   * lse-server executable exits with status 2 for these. */
  LSE_ERR_INVALID_ARGUMENT = 1,
  /* Opening the device, the model or a listener failed. Exit status 1. */
  LSE_ERR_FAILED = 2,
  /* The engine is closing, or the request is unknown or already finished. */
  LSE_ERR_STATE = 3,
} lse_result;

/* Every option the lse-server command line takes, one field per flag. Fill it
 * with lse_config_init, then set what differs. Strings are borrowed for the
 * duration of the call that takes the config; NULL means "the default". */
typedef struct lse_config {
  uint32_t abi_version;          /* set by lse_config_init */
  uint32_t struct_size;          /* set by lse_config_init */

  const char *model;             /* -m/--model: checkpoint dir, .safetensors, or HF repo id */
  const char *served_name;       /* --served-name (default: model) */
  const char *tokenizer_repo;    /* --tokenizer */

  /* Speculative decoding. */
  const char *mtp_path;          /* --mtp */
  uint32_t mtp_depth;            /* --mtp-depth, 1..7 (default 3) */
  int32_t no_mtp;                /* --no-mtp */
  int32_t dflash2;               /* --dflash2=on */
  const char *dflash2_model;     /* --dflash2-model */

  /* Attention and prefill. */
  int32_t flashprefill_v2;       /* --FlashPrefillV2: -1 default, 0 off, 1 on */
  const char *attention_prefill; /* --attention-prefill: dense|blasst|flashprefill-v2 */
  const char *attention_decode;  /* --attention-decode: dense|blasst */
  const char *attention_calibration; /* --attention-calibration FILE */
  const char *kv_cache_dtype;    /* --kv-cache-dtype: fp32|fp16|bf16|fp8|bf8 */
  int32_t kv_len;                /* --kv-len (0: the model's) */
  uint32_t batch_size;           /* --batch-size (default 1024) */
  uint32_t ubatch_size;          /* --ubatch-size (default 1024) */

  /* Sampling and limits. */
  int32_t has_temperature;       /* nonzero: temperature overrides the model's */
  float temperature;             /* --temperature, 0..2 */
  int32_t max_tokens;            /* --max-tokens cap per request (default 4096) */

  /* Devices and kernels. */
  const char *pool;              /* --pool, e.g. "hrx:0" */
  const char *dialect;           /* --dialect: hip|loom */
  const char *cache_dir;         /* --cache-dir (default: the platform cache) */

  /* HTTP adapter (lse_http_start only). */
  const char *host;              /* --host (default 127.0.0.1) */
  int32_t port;                  /* --port (default 8080) */
  const char *api_key;           /* --api-key: require Authorization: Bearer KEY */
  int32_t shutdown_grace_seconds;/* --shutdown-grace-seconds, 1..600 (default 30) */

  /* Sessions: a request naming a "session_id" continues that session's KV;
   * one without runs in a session released when it ends. Beyond either
   * limit, least recently used idle sessions are evicted (their next request
   * prefills again); under memory pressure idle sessions are evicted
   * whatever the limits. 0 means no limit. */
  uint32_t max_sessions;          /* --max-sessions (default 8) */
  uint64_t session_memory_budget; /* --session-memory-budget BYTES (default 0) */

  /* --no-cpu-fallback: a group the device cannot run fails the request
   * (naming the cause) instead of running on the CPU. Either way every CPU
   * fallback is logged, counted in lse_status (engine.cpu_fallback) and
   * reported to the client as "lse_warnings". */
  int32_t disable_cpu_fallback;
} lse_config;

LSE_API void lse_config_init(lse_config *cfg);

/* Log lines (what lse-server prints on stderr), one call per line without the
 * trailing newline. Process-wide: the engine's diagnostics are written to
 * stderr, which is redirected through the callback while one is set (and
 * still forwarded to the original stderr). NULL restores plain stderr. Once
 * this returns, no call into the previous callback is running. A callback
 * must not call lse_set_log_callback itself. */
typedef enum lse_log_level { LSE_LOG_INFO = 0, LSE_LOG_ERROR = 1 } lse_log_level;
typedef void (*lse_log_cb)(void *user, lse_log_level level, const char *line);
LSE_API void lse_set_log_callback(lse_log_cb cb, void *user);

typedef struct lse_engine lse_engine;

/* Opens the device set and loads the model, blocking until it is ready to
 * serve. Progress is visible meanwhile through lse_status(NULL, ...). On
 * failure returns NULL and, when err is not NULL, a message to release with
 * lse_free; lse_last_error() then tells an invalid configuration from a
 * failure to load. One engine per process. */
LSE_API lse_engine *lse_open(const lse_config *cfg, char **err);
/* The lse_result of the last lse_open on this thread. */
LSE_API lse_result lse_last_error(void);
/* Cancels every request, stops HTTP if it runs, waits for the workers, then
 * releases the model and the device runtime. */
LSE_API void lse_close(lse_engine *engine);

/* Requests. */
typedef uint64_t lse_request_id;
typedef enum lse_event {
  /* A complete non-streaming 2xx response body. Final. */
  LSE_EVENT_RESPONSE = 0,
  /* One chunk of a "stream": true response: exactly the JSON object an SSE
   * "data:" line carries, without the framing. */
  LSE_EVENT_CHUNK = 1,
  /* The stream finished (the SSE "[DONE]"). data is NULL. Final. */
  LSE_EVENT_DONE = 2,
  /* The request failed: data is the same {"error": {...}} JSON the HTTP
   * server sends, status the HTTP status it would have used (a failure in the
   * middle of a stream reports 500; a cancelled request 499). Final. */
  LSE_EVENT_ERROR = 3,
} lse_event;
/* data is valid only during the call. Exactly one final event per request. */
typedef void (*lse_response_cb)(void *user, lse_request_id id, lse_event event,
                                int status, const char *data, size_t len);

/* Starts one request, e.g. ("POST", "/v1/chat/completions", body) or
 * ("GET", "/v1/models", NULL). Returns at once; the callback carries the
 * answer. Requests run concurrently and queue for the device exactly as they
 * do behind the HTTP server. */
LSE_API lse_result lse_request(lse_engine *engine, const char *method,
                               const char *path, const char *json_body,
                               size_t len, lse_response_cb cb, void *user,
                               lse_request_id *out_id);
/* Stops a queued or running request. Its callback receives LSE_EVENT_ERROR
 * (status 499) instead of any further output. */
LSE_API lse_result lse_cancel(lse_engine *engine, lse_request_id id);

/* Releases a session: its KV, recurrent state and everything the engine
 * built for it go back to the device, waiting for a generation in flight in
 * it to finish. LSE_ERR_STATE when there is no such session. The same as
 * DELETE /v1/lse/sessions/{id}; GET /v1/lse/sessions and lse_status list the
 * live sessions and the bytes each holds. */
LSE_API lse_result lse_session_close(lse_engine *engine, const char *session_id);

/* Device power, for hosts that suspend the app or the machine (iPadOS
 * background, macOS sleep), with a device runtime that tracks it (the
 * mac_linuxgpu HSA runtime). lse_status reports the state under "power":
 * active, suspending, suspended, resuming, lost, or unknown when untracked.
 *
 * lse_power_prepare: before the host stops using the GPU for a while. Waits
 * up to drain_timeout_ms for work already submitted, then pauses every
 * queue; device memory is kept and a generation in flight pauses and
 * continues on resume. Meanwhile new completion requests are answered 503
 * {"error": {"type": "engine_suspended", "code": "suspended"},
 * "retry_after": 1} (over HTTP with Retry-After) having started nothing.
 * lse_power_resume: when the host comes back.
 *
 * Lost: the host slept and the device's memory went with it. Requests are
 * answered 503 {"error": {"type": "device_lost"}}, lse_status reports state
 * "lost", and lse_power_resume returns LSE_ERR_STATE. The engine cannot
 * continue: lse_close it and lse_open again (the device is probed afresh
 * and the model loads again; sessions are gone).
 *
 * Both take the engine (NULL acts on the device runtime alone, once an
 * engine has opened it), fill *json_out with the power state as lse_status
 * reports it (release with lse_free), and return LSE_ERR_FAILED with a
 * message in *err when the runtime has no power control or the driver
 * refused. */
LSE_API lse_result lse_power_prepare(lse_engine *engine, uint32_t drain_timeout_ms,
                                     char **json_out, char **err);
LSE_API lse_result lse_power_resume(lse_engine *engine, char **json_out, char **err);

/* Engine status as JSON (release with lse_free): load phase and progress
 * (also before lse_open returns, with engine NULL), the model being served,
 * request counters, the timings of the last completed generation, and the
 * device bytes the engine holds ("memory": live and peak). */
LSE_API lse_result lse_status(const lse_engine *engine, char **json_out);

/* Model inspection and memory planning.
 *
 * Neither function opens a device or reads a tensor's payload: config.json
 * and the safetensors headers are enough. Both are fast, allocate no GPU
 * memory and may be called at any time, from any thread, with or without an
 * engine open -- e.g. to fill a model picker or size a context before
 * lse_open. Results are JSON strings released with lse_free; on failure
 * *json_out is NULL and *err (when err is not NULL) holds a message to
 * release with lse_free.
 *
 * lse_model_info describes a model directory, .safetensors file or HF repo id
 * in the local cache: what LSE's loader detects it as (from its tensor names,
 * not only model_type), dense or MoE, layer count and widths, which layers
 * hold KV and which are linear-attention (Gated DeltaNet) layers with fixed
 * recurrent state, max context, vocab, quantization, bytes on disk and the
 * weights' estimated VRAM, MTP and DFlash2 facts (including whether it is
 * itself a DFlash2 draft and the fields a draft must match), and, for every
 * KV cache format LSE supports (fp32, fp16, bf16, fp8, bf8), the KV bytes per
 * token and per 16-token block as the paged allocator sizes them. */
LSE_API lse_result lse_model_info(const char *model, char **json_out, char **err);

/* What lse_open with `cfg` would allocate on the device, by component:
 * weights (packed the way the loader packs them), KV (with the allocator's
 * paging granularity), recurrent state, RoPE tables, program workspace, the
 * prefill activation peak, and the MTP or DFlash2 draft, with resident and
 * peak totals. Reads the fields lse_open reads: model, mtp_path/no_mtp/
 * mtp_depth, dflash2/dflash2_model, kv_cache_dtype, kv_len, batch_size,
 * ubatch_size and dialect (Loom stores K/V in fragments, HIP in contiguous
 * pools). `options_json` may be NULL or an object with any of:
 *   "context_tokens":      tokens held when estimating (default: kv_len)
 *   "sequences":           sequences decoded together (default 1)
 *   "device_arch":         e.g. "gfx1201"; decides whether the packed Q8
 *                          weight copies are counted (default: assumed)
 *   "kv_storage":          "fragmented" or "contiguous" (overrides dialect)
 *   "device_memory_bytes": also report "fits" and "max_kv_len", the largest
 *                          kv_len whose estimate fits in this many bytes
 * Weights, KV, recurrent state, RoPE and the DFlash2 ring are computed
 * exactly; activation and workspace are modelled and flagged approximate. */
LSE_API lse_result lse_estimate(const lse_config *cfg, const char *options_json,
                                char **json_out, char **err);

/* Optional HTTP surface over the same engine. host NULL and port <= 0 take
 * the config's. Returns once the socket is bound and listening. */
LSE_API lse_result lse_http_start(lse_engine *engine, const char *host,
                                  int32_t port, char **err);
/* Asks the listener to stop and in-flight HTTP requests to end. Safe to call
 * repeatedly and from a signal-driven thread; does not block. */
LSE_API void lse_http_stop(lse_engine *engine);
/* Blocks until the listener has stopped and every HTTP worker has drained. */
LSE_API lse_result lse_http_wait(lse_engine *engine, char **err);

LSE_API void lse_free(void *p);

#ifdef __cplusplus
}
#endif

#endif /* LSE_LSE_H */
