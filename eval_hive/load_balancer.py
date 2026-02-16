"""Async least-connections reverse proxy for multi-server inference.

Proxies HTTP requests from lm-eval to a pool of vLLM backends, routing
each request to the backend with the fewest in-flight requests.

Backends are automatically removed from rotation after consecutive failures
and re-added when periodic health checks succeed.

Usage::

    python -m eval_hive.load_balancer \
        --listen-port 30042 \
        --backends node1:30043,node2:30043,localhost:30043
"""

import argparse
import asyncio
import logging
import time

import aiohttp
from aiohttp import web

logger = logging.getLogger("eval_hive.load_balancer")

# After this many consecutive forward errors, mark backend as down
FAILURE_THRESHOLD = 10
# How often (seconds) to health-check backends and potentially restore them
HEALTH_CHECK_INTERVAL = 60


class BackendPool:
    """Manages a pool of backends with least-connections routing and health tracking."""

    def __init__(self, backends: list[str], failure_threshold: int = FAILURE_THRESHOLD):
        self._backends = [f"http://{b}" for b in backends]
        self._in_flight: dict[str, int] = {b: 0 for b in self._backends}
        self._consecutive_failures: dict[str, int] = {b: 0 for b in self._backends}
        self._alive: dict[str, bool] = {b: True for b in self._backends}
        self._failure_threshold = failure_threshold
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._health_task: asyncio.Task | None = None

    async def start(self) -> None:
        connector = aiohttp.TCPConnector(limit_per_host=50, keepalive_timeout=30)
        self._session = aiohttp.ClientSession(connector=connector)
        self._health_task = asyncio.create_task(self._periodic_health_check())

    async def stop(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        if self._session:
            await self._session.close()

    async def _pick(self) -> str:
        async with self._lock:
            alive = {b: c for b, c in self._in_flight.items() if self._alive[b]}
            if not alive:
                # All backends down — fall back to least-connections across all
                logger.warning("All backends marked down, routing to least-loaded anyway")
                alive = self._in_flight
            chosen = min(alive, key=alive.get)  # type: ignore[arg-type]
            self._in_flight[chosen] += 1
            return chosen

    async def _release(self, backend: str) -> None:
        async with self._lock:
            self._in_flight[backend] = max(0, self._in_flight[backend] - 1)

    async def _record_success(self, backend: str) -> None:
        async with self._lock:
            if self._consecutive_failures[backend] > 0:
                self._consecutive_failures[backend] = 0
            if not self._alive[backend]:
                self._alive[backend] = True
                logger.info("Backend %s recovered (marked alive via successful request)", backend)

    async def _record_failure(self, backend: str) -> None:
        async with self._lock:
            self._consecutive_failures[backend] += 1
            count = self._consecutive_failures[backend]
            if count >= self._failure_threshold and self._alive[backend]:
                self._alive[backend] = False
                n_alive = sum(1 for v in self._alive.values() if v)
                logger.warning(
                    "Backend %s marked DOWN after %d consecutive failures (%d/%d alive)",
                    backend, count, n_alive, len(self._backends),
                )

    def _short_name(self, backend: str) -> str:
        """Extract host from backend URL, e.g. 'http://lrdn1970:31437' -> 'lrdn1970'."""
        return backend.split("//", 1)[-1].split(":")[0]

    async def forward(self, request: web.Request) -> web.StreamResponse:
        backend = await self._pick()
        node = self._short_name(backend)
        t0 = time.monotonic()
        try:
            url = f"{backend}{request.path_qs}"
            body = await request.read()
            headers = {
                k: v
                for k, v in request.headers.items()
                if k.lower() not in ("host", "transfer-encoding")
            }
            async with self._session.request(
                method=request.method,
                url=url,
                headers=headers,
                data=body,
                timeout=aiohttp.ClientTimeout(total=600),
            ) as resp:
                response = web.StreamResponse(
                    status=resp.status,
                    headers={
                        k: v
                        for k, v in resp.headers.items()
                        if k.lower()
                        not in ("transfer-encoding", "content-encoding", "content-length")
                    },
                )
                await response.prepare(request)
                async for chunk in resp.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                elapsed = time.monotonic() - t0
                logger.info(
                    "%s %s -> %s %d (%.1fs)",
                    request.method, request.path, node, resp.status, elapsed,
                )
                await self._record_success(backend)
                return response
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.error(
                "%s %s -> %s FAILED (%.1fs): %s",
                request.method, request.path, node, elapsed, e,
            )
            await self._record_failure(backend)
            return web.Response(status=502, text=f"Backend error: {e}")
        finally:
            await self._release(backend)

    async def check_health(self) -> tuple[bool, dict[str, bool]]:
        details: dict[str, bool] = {}
        for backend in self._backends:
            try:
                async with self._session.get(
                    f"{backend}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    details[backend] = resp.status == 200
            except Exception:
                details[backend] = False
        return any(details.values()), details

    async def _periodic_health_check(self) -> None:
        """Periodically probe all backends and update alive status."""
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            try:
                _, details = await self.check_health()
                async with self._lock:
                    for backend, healthy in details.items():
                        was_alive = self._alive[backend]
                        if healthy and not was_alive:
                            self._alive[backend] = True
                            self._consecutive_failures[backend] = 0
                            logger.info("Backend %s recovered (health check passed)", backend)
                        elif not healthy and was_alive:
                            self._alive[backend] = False
                            logger.warning("Backend %s marked DOWN (health check failed)", backend)
                    n_alive = sum(1 for v in self._alive.values() if v)
                    logger.info(
                        "Health check: %d/%d backends alive", n_alive, len(self._backends)
                    )
            except Exception as e:
                logger.error("Periodic health check error: %s", e)


async def _health_handler(request: web.Request) -> web.Response:
    pool: BackendPool = request.app["pool"]
    healthy, details = await pool.check_health()
    return web.json_response(details, status=200 if healthy else 503)


async def _proxy_handler(request: web.Request) -> web.StreamResponse:
    pool: BackendPool = request.app["pool"]
    return await pool.forward(request)


def create_app(backends: list[str]) -> web.Application:
    pool = BackendPool(backends)
    app = web.Application()
    app["pool"] = pool
    app.router.add_get("/health", _health_handler)
    app.router.add_route("*", "/{path_info:.*}", _proxy_handler)
    app.on_startup.append(lambda _: pool.start())
    app.on_cleanup.append(lambda _: pool.stop())
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="eval-hive load balancer")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument(
        "--backends",
        type=str,
        required=True,
        help="Comma-separated host:port list",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    backends = [b.strip() for b in args.backends.split(",")]
    logger.info("Starting load balancer on :%d -> %s", args.listen_port, backends)

    app = create_app(backends)
    web.run_app(
        app,
        host="0.0.0.0",
        port=args.listen_port,
        access_log=None,  # we log per-request in forward() with backend info
        print=lambda msg: logger.info(msg),
    )


if __name__ == "__main__":
    main()
