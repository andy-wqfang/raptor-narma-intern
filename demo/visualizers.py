"""Serve both RAPTOR visualizers as subpages of one loopback website."""

from __future__ import annotations

import argparse
import asyncio
from argparse import Namespace
from pathlib import Path
from typing import Any


async def _wait_for_listener(port: int, timeout_s: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_running_loop().time() >= deadline:
                raise RuntimeError(
                    f"visualizer backend did not listen on port {port}"
                )
            await asyncio.sleep(0.05)


async def start_gateway(
    port: int,
    *,
    quad_backend_port: int,
    hex_backend_port: int,
) -> tuple[Any, Any, int]:
    """Start the single-origin HTTP/WebSocket visualizer gateway."""

    import aiohttp
    from aiohttp import WSMsgType, web

    app = web.Application()
    proxy_session = aiohttp.ClientSession(auto_decompress=True)
    app["proxy_session"] = proxy_session

    async def close_proxy_session(_app: web.Application) -> None:
        await proxy_session.close()

    app.on_cleanup.append(close_proxy_session)

    async def redirect_to_quad(_request: web.Request) -> web.Response:
        raise web.HTTPFound("/quadcopter/?L2FDisplayActions=true")

    async def redirect_quad_slash(_request: web.Request) -> web.Response:
        raise web.HTTPFound("/quadcopter/?L2FDisplayActions=true")

    async def redirect_hex_slash(_request: web.Request) -> web.Response:
        raise web.HTTPFound("/hexacopter/?L2FDisplayActions=true")

    async def proxy_websocket(
        request: web.Request,
        backend_port: int,
    ) -> web.WebSocketResponse:
        upstream = await proxy_session.ws_connect(
            f"http://127.0.0.1:{backend_port}/ui",
            max_msg_size=100_000_000,
        )
        browser = web.WebSocketResponse(max_msg_size=100_000_000)
        await browser.prepare(request)

        async def forward(source: Any, target: Any) -> None:
            async for message in source:
                if message.type is WSMsgType.TEXT:
                    await target.send_str(message.data)
                elif message.type is WSMsgType.BINARY:
                    await target.send_bytes(message.data)
                elif message.type in {
                    WSMsgType.CLOSE,
                    WSMsgType.CLOSED,
                    WSMsgType.ERROR,
                }:
                    break

        browser_to_upstream = asyncio.create_task(
            forward(browser, upstream)
        )
        upstream_to_browser = asyncio.create_task(
            forward(upstream, browser)
        )
        tasks = {browser_to_upstream, upstream_to_browser}
        try:
            _done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await upstream.close()
            await browser.close()
        return browser

    async def proxy_http(
        request: web.Request,
        backend_port: int,
    ) -> web.Response:
        tail = request.match_info.get("tail", "")
        upstream_url = f"http://127.0.0.1:{backend_port}/{tail}"
        if request.query_string:
            upstream_url += f"?{request.query_string}"

        async with proxy_session.request(
            request.method,
            upstream_url,
        ) as upstream:
            body = await upstream.read()
            content_type = upstream.headers.get("Content-Type", "")
            if "text/html" in content_type:
                text = body.decode(upstream.charset or "utf-8")
                text = text.replace(
                    'window.location.host + "/ui"',
                    (
                        "window.location.host + "
                        'window.location.pathname.replace(/\\/$/, "") '
                        '+ "/ui"'
                    ),
                )
                body = text.encode("utf-8")

            forwarded_headers = {
                key: value
                for key, value in upstream.headers.items()
                if key.lower()
                in {
                    "cache-control",
                    "content-type",
                    "etag",
                    "last-modified",
                }
            }
            return web.Response(
                body=body,
                status=upstream.status,
                headers=forwarded_headers,
            )

    async def quad_websocket(request: web.Request) -> web.WebSocketResponse:
        return await proxy_websocket(request, quad_backend_port)

    async def hex_websocket(request: web.Request) -> web.WebSocketResponse:
        return await proxy_websocket(request, hex_backend_port)

    async def quad_http(request: web.Request) -> web.Response:
        return await proxy_http(request, quad_backend_port)

    async def hex_http(request: web.Request) -> web.Response:
        return await proxy_http(request, hex_backend_port)

    app.add_routes(
        [
            web.get("/", redirect_to_quad),
            web.get("/quadcopter", redirect_quad_slash),
            web.get("/hexacopter", redirect_hex_slash),
            web.get("/quadcopter/ui", quad_websocket),
            web.get("/hexacopter/ui", hex_websocket),
            web.route("*", "/quadcopter/{tail:.*}", quad_http),
            web.route("*", "/hexacopter/{tail:.*}", hex_http),
        ]
    )

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    sockets = site._server.sockets if site._server is not None else []
    if not sockets:
        await runner.cleanup()
        raise RuntimeError("visualizer gateway did not create a listener")
    actual_port = int(sockets[0].getsockname()[1])
    return runner, site, actual_port


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serve the RAPTOR quadcopter and hexacopter visualizers as "
            "subpages of one loopback website"
        )
    )
    parser.add_argument("--port", type=int, default=13337)
    parser.add_argument("--quad-backend-port", type=int, default=13338)
    parser.add_argument("--hex-backend-port", type=int, default=13339)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="run both simulations as fast as possible",
    )
    return parser


async def async_main(args: argparse.Namespace) -> int:
    ports = {
        args.port,
        args.quad_backend_port,
        args.hex_backend_port,
    }
    if len(ports) != 3:
        raise ValueError("gateway and backend ports must be different")

    from demo import hexacopter, position_hold

    quad_args = Namespace(
        check=False,
        port=args.quad_backend_port,
        hex_href="/hexacopter/?L2FDisplayActions=true",
        seed=args.seed,
        loop=True,
        no_realtime=args.no_realtime,
        output_dir=Path("artifacts/position_hold"),
        announce=False,
    )
    hex_args = Namespace(
        check=False,
        port=args.hex_backend_port,
        quad_href="/quadcopter/?L2FDisplayActions=true",
        loop=True,
        no_realtime=args.no_realtime,
        output_dir=Path("artifacts/hexacopter"),
        announce=False,
    )

    quad_task = asyncio.create_task(
        position_hold.async_main(quad_args),
        name="raptor-quadcopter-backend",
    )
    hex_task = asyncio.create_task(
        hexacopter.async_main(hex_args),
        name="raptor-hexacopter-backend",
    )
    tasks = {quad_task, hex_task}
    gateway_runner = None
    try:
        await asyncio.gather(
            _wait_for_listener(args.quad_backend_port),
            _wait_for_listener(args.hex_backend_port),
        )
        gateway_runner, _site, actual_port = await start_gateway(
            args.port,
            quad_backend_port=args.quad_backend_port,
            hex_backend_port=args.hex_backend_port,
        )
        print(
            "RAPTOR visualizer site listening on "
            f"http://127.0.0.1:{actual_port}"
        )
        print(
            "Quadcopter: "
            f"http://127.0.0.1:{actual_port}/quadcopter/"
            "?L2FDisplayActions=true"
        )
        print(
            "Hexacopter: "
            f"http://127.0.0.1:{actual_port}/hexacopter/"
            "?L2FDisplayActions=true"
        )
        print(
            "Forward it with: ssh -N "
            f"-L {actual_port}:127.0.0.1:{actual_port} "
            "user@remote-host"
        )
        results = await asyncio.gather(quad_task, hex_task)
        return 0 if all(result == 0 for result in results) else 1
    finally:
        if gateway_runner is not None:
            await gateway_runner.cleanup()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main() -> int:
    args = build_parser().parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
