import json
import unittest

import aiohttp
import numpy as np

from demo.hexacopter import (
    BoundedAllocator,
    TOTAL_STEPS,
    _hexacopter_dashboard_html,
    build_runtime,
    firefly_model,
    parameters_message,
    run_episode,
    state_message,
    ui_message,
)
from demo.position_hold import start_loopback_server


class FireflyModelTests(unittest.TestCase):
    def test_published_model_and_allocator(self) -> None:
        model = firefly_model()
        self.assertEqual(model.rotor_positions_m.shape, (6, 3))
        self.assertAlmostEqual(model.mass_kg, 1.5)
        self.assertAlmostEqual(model.arm_length_m, 0.215)
        self.assertAlmostEqual(model.max_rotor_speed_rad_s, 838.0)
        self.assertGreater(model.thrust_to_weight, 2.4)
        self.assertLess(model.thrust_to_weight, 2.5)

        allocator = BoundedAllocator(model)
        self.assertEqual(allocator.physical_matrix.shape, (4, 6))
        self.assertEqual(np.linalg.matrix_rank(allocator.physical_matrix), 4)
        hover_wrench = np.array([model.mass_kg * 9.81, 0, 0, 0])
        thrusts, residual = allocator.allocate(hover_wrench)
        np.testing.assert_allclose(
            thrusts,
            np.full(6, model.mass_kg * 9.81 / 6),
            rtol=1e-10,
            atol=1e-10,
        )
        self.assertLess(residual, 1e-12)

    def test_six_rotor_browser_payloads(self) -> None:
        runtime = build_runtime()
        parameters = json.loads(parameters_message(runtime, "test"))
        self.assertEqual(parameters["channel"], "setParameters")
        self.assertEqual(
            len(parameters["data"]["dynamics"]["rotor_positions"]),
            6,
        )

        ui = json.loads(ui_message("test"))
        self.assertEqual(ui["channel"], "setUI")
        self.assertIn(
            "state.rotor_speeds_normalized",
            ui["data"]["render_function"],
        )
        self.assertFalse(ui["data"]["options"]["showAxes"])
        self.assertIn(
            "i < this.rotors.length",
            ui["data"]["render_function"],
        )

        state = json.loads(
            state_message(
                runtime,
                np.zeros(4),
                0.0,
                0,
                1,
                1,
            )
        )
        self.assertEqual(state["channel"], "setState")
        self.assertEqual(
            len(state["data"]["state"]["rotor_speeds_normalized"]),
            6,
        )


class FireflyScenarioTests(unittest.IsolatedAsyncioTestCase):
    async def test_unchanged_policy_holds_firefly_position(self) -> None:
        metrics, trajectory = await run_episode(realtime=False)
        self.assertTrue(metrics["passed"], json.dumps(metrics, indent=2))
        self.assertEqual(trajectory["position"].shape, (TOTAL_STEPS, 3))
        self.assertEqual(
            trajectory["motor_speed_rad_s"].shape,
            (TOTAL_STEPS, 6),
        )
        self.assertTrue(
            np.isfinite(trajectory["allocation_residual"]).all()
        )

    async def test_hexacopter_dashboard(self) -> None:
        runner, _site, port = await start_loopback_server(
            0,
            index_transform=lambda original: _hexacopter_dashboard_html(
                original,
                peer_href="/quadcopter/?L2FDisplayActions=true",
            ),
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://127.0.0.1:{port}/"
                ) as response:
                    self.assertEqual(response.status, 200)
                    body = await response.text()
                    self.assertIn(
                        "RAPTOR on a Firefly hexacopter",
                        body,
                    )
                    self.assertIn("Bounded allocator", body)
                    self.assertIn("Six-motor geometry", body)
                    self.assertIn("Front left", body)
                    self.assertIn("6.003 N", body)
                    self.assertIn("Eight quadcopters", body)
                    self.assertIn(
                        'href="/quadcopter/?L2FDisplayActions=true"',
                        body,
                    )
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
