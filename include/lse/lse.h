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

/* Engine version string, e.g. "0.4.24". */
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
} lse_config;

LSE_API void lse_config_init(lse_config *cfg);

/* Log lines (what lse-server prints on stderr), one call per line without the
 * trailing newline. Process-wide: the engine's diagnostics are written to
 * stderr, which is redirected through the callback while one is set (and
 * still forwarded to the original stderr). NULL restores plain stderr. */
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

/* Engine status as JSON (release with lse_free): load phase and progress
 * (also before lse_open returns, with engine NULL), the model being served,
 * request counters, and the timings of the last completed generation. */
LSE_API lse_result lse_status(const lse_engine *engine, char **json_out);

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
