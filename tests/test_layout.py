"""The packages import. (The paper specs and their tests live in the private `.pawtrace/` folder.)"""


def test_packages_import():
    import adapters  # noqa: F401
    import controllers.parkinsons.baselines  # noqa: F401
    import controllers.parkinsons.rl.ddpg  # noqa: F401
    import controllers.parkinsons.rl.sea_dbs  # noqa: F401
    import controllers.parkinsons.rl.snn  # noqa: F401
    import plants.parkinsons.kumaravelu2016  # noqa: F401
