"""Smoke tests: editable install and package layout."""


def test_packages_import() -> None:
    import envs
    import controllers
    from controllers import ddpg, sea_dbs, snn

    assert envs is not None
    assert controllers is not None
    assert ddpg is not None
    assert snn is not None
    assert sea_dbs is not None
