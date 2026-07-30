import asyncio
import unittest

import aiohttp

from demo.hexacopter import _hexacopter_dashboard_html
from demo.position_hold import _dashboard_html, start_loopback_server
from demo.visualizers import start_gateway


class CombinedVisualizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_subpages_and_websocket_isolation(self) -> None:
        quad_runner, _quad_site, quad_port = await start_loopback_server(
            0,
            index_transform=lambda original: _dashboard_html(
                original,
                peer_href="/hexacopter/?L2FDisplayActions=true",
            ),
        )
        hex_runner, _hex_site, hex_port = await start_loopback_server(
            0,
            index_transform=lambda original: _hexacopter_dashboard_html(
                original,
                peer_href="/quadcopter/?L2FDisplayActions=true",
            ),
        )
        gateway_runner, _gateway_site, gateway_port = await start_gateway(
            0,
            quad_backend_port=quad_port,
            hex_backend_port=hex_port,
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{gateway_port}/",
                    allow_redirects=False,
                ) as response:
                    self.assertEqual(response.status, 302)
                    self.assertTrue(
                        response.headers["Location"].startswith(
                            "/quadcopter/"
                        )
                    )

                async with session.get(
                    f"http://127.0.0.1:{gateway_port}/quadcopter/"
                ) as response:
                    self.assertEqual(response.status, 200)
                    quad_body = await response.text()
                    self.assertIn("RAPTOR position keeping", quad_body)
                    self.assertIn(
                        'href="/hexacopter/?L2FDisplayActions=true"',
                        quad_body,
                    )
                    self.assertIn(
                        "window.location.pathname.replace",
                        quad_body,
                    )

                async with session.get(
                    f"http://127.0.0.1:{gateway_port}/hexacopter/"
                ) as response:
                    self.assertEqual(response.status, 200)
                    hex_body = await response.text()
                    self.assertIn(
                        "RAPTOR on a Firefly hexacopter",
                        hex_body,
                    )
                    self.assertIn(
                        'href="/quadcopter/?L2FDisplayActions=true"',
                        hex_body,
                    )

                backend = await session.ws_connect(
                    f"http://127.0.0.1:{quad_port}/backend"
                )
                handshake = await backend.receive_json()
                self.assertEqual(handshake["channel"], "handshake")
                browser = await session.ws_connect(
                    f"http://127.0.0.1:{gateway_port}/quadcopter/ui"
                )
                message = {
                    "channel": "gatewayTest",
                    "data": {"isolated": True},
                }
                await backend.send_json(message)
                received = await asyncio.wait_for(
                    browser.receive_json(),
                    timeout=2,
                )
                self.assertEqual(received, message)
                await browser.close()
                await backend.close()
        finally:
            await gateway_runner.cleanup()
            await hex_runner.cleanup()
            await quad_runner.cleanup()


if __name__ == "__main__":
    unittest.main()
