import asyncio
import json
import unittest

import aiohttp
import numpy as np
import l2f

from demo.position_hold import (
    KICK_STEP,
    N_DRONES,
    _dashboard_html,
    build_runtime,
    parameters_message,
    run_episode,
    start_loopback_server,
    state_message,
)


class ConfigurationTests(unittest.TestCase):
    def test_sampled_configuration_invariants(self) -> None:
        runtime = build_runtime(42)
        self.assertEqual(len(runtime.configurations), N_DRONES)
        self.assertEqual(runtime.observations.shape[0], N_DRONES)
        self.assertGreaterEqual(runtime.observations.shape[1], 22)

        for config, metadata in zip(
            runtime.configurations,
            runtime.configuration_metadata,
        ):
            dynamics = config["dynamics"]
            self.assertGreaterEqual(metadata["mass_kg"], 0.02)
            self.assertLessEqual(metadata["mass_kg"], 5.0)
            self.assertGreaterEqual(metadata["thrust_to_weight"], 1.5)
            self.assertLessEqual(metadata["thrust_to_weight"], 5.0)
            self.assertGreaterEqual(metadata["torque_to_inertia"], 40.0)
            self.assertLessEqual(metadata["torque_to_inertia"], 1200.0)
            self.assertTrue(np.all(np.diag(dynamics["J"]) > 0))
            product = np.asarray(dynamics["J"]) @ np.asarray(dynamics["J_inv"])
            np.testing.assert_allclose(product, np.eye(3), rtol=0.08, atol=0.01)
            self.assertEqual(len(dynamics["rotor_positions"]), 4)
            self.assertEqual(len(dynamics["rotor_thrust_coefficients"]), 4)
            expected_xy_signs = [(1, -1), (-1, -1), (-1, 1), (1, 1)]
            actual_xy_signs = [
                (int(np.sign(position[0])), int(np.sign(position[1])))
                for position in dynamics["rotor_positions"]
            ]
            self.assertEqual(actual_xy_signs, expected_xy_signs)
            self.assertFalse(config["mdp"]["termination"]["enabled"])

    def test_dashboard_telemetry_payloads(self) -> None:
        runtime = build_runtime(42)
        ui = l2f.UI()
        ui.ns = "test"
        parameters = json.loads(parameters_message(runtime, ui))
        self.assertTrue(parameters["latch"])
        self.assertEqual(len(parameters["data"]), N_DRONES)
        self.assertEqual(parameters["data"][0]["demo"]["label"], "Drone 1")
        self.assertIn("thrust_to_weight", parameters["data"][0]["demo"])

        state = json.loads(
            state_message(
                runtime,
                ui,
                np.zeros((N_DRONES, 4), dtype=np.float32),
                KICK_STEP,
                3,
            )
        )
        demo = state["data"]["demo"]
        self.assertTrue(state["latch"])
        self.assertEqual(demo["episode"], 3)
        self.assertEqual(demo["phase_id"], "kick_recovery")
        self.assertEqual(demo["event"], "Velocity kick applied")
        self.assertEqual(len(demo["position_errors_m"]), N_DRONES)


class ScenarioTests(unittest.IsolatedAsyncioTestCase):
    async def test_robust_scenario_passes(self) -> None:
        metrics, trajectory = await run_episode(42, realtime=False)
        self.assertTrue(metrics["passed"], json.dumps(metrics, indent=2))
        self.assertTrue(np.isfinite(metrics["run"]["maximum_absolute_action"]))
        self.assertEqual(trajectory.shape, (1600, 8, 3))
        self.assertTrue(np.isfinite(trajectory).all())

    async def test_loopback_server_and_latching(self) -> None:
        runner, site, port = await start_loopback_server(
            0,
            index_transform=lambda original: _dashboard_html(
                original,
                peer_href="/hexacopter/?L2FDisplayActions=true",
            ),
        )
        try:
            sockets = site._server.sockets
            self.assertEqual(sockets[0].getsockname()[0], "127.0.0.1")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://127.0.0.1:{port}/") as response:
                    self.assertEqual(response.status, 200)
                    body = await response.text()
                    self.assertIn("three.module.js", body)
                    self.assertIn("RAPTOR position keeping", body)
                    self.assertIn("Episode phase", body)
                    self.assertIn("Drone configurations", body)
                    self.assertIn("Display grid", body)
                    self.assertIn("3 × 3 cells", body)
                    self.assertIn("0.90 m spacing", body)
                    self.assertIn("1.80 × 1.80 m", body)
                    self.assertIn("Firefly hexacopter", body)
                    self.assertIn(
                        'href="/hexacopter/?L2FDisplayActions=true"',
                        body,
                    )

                backend = await session.ws_connect(
                    f"http://127.0.0.1:{port}/backend"
                )
                handshake = await backend.receive_json()
                self.assertEqual(handshake["channel"], "handshake")
                message = {
                    "channel": "testLatch",
                    "latch": True,
                    "data": {"ready": True},
                }
                await backend.send_json(message)

                ui = await session.ws_connect(f"http://127.0.0.1:{port}/ui")
                received = await asyncio.wait_for(ui.receive_json(), timeout=2)
                self.assertEqual(received, message)
                await ui.close()
                await backend.close()
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
