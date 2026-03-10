from rd_sbi.utils.seed import set_global_seed


def test_set_global_seed_non_negative_roundtrip() -> None:
    applied = set_global_seed(12345)
    assert applied == 12345


def test_set_global_seed_rejects_negative() -> None:
    try:
        set_global_seed(-1)
        assert False, "expected ValueError"
    except ValueError:
        assert True
