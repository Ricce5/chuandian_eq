from src.utils.catalog_pathing import build_tpp_catalog_init_kwargs


class _BasicCatalog:
    def __init__(self, root_dir, data_dir=None, freq="1h", normalize=True):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.freq = freq
        self.normalize = normalize


class _ExtraKwargCatalog:
    def __init__(self, root_dir, data_dir=None, **kwargs):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.kwargs = kwargs


class InducedTripletGroupedCatalog:
    pass


class _GroupedCatalog(InducedTripletGroupedCatalog):
    def __init__(self, root_dir, data_dir=None, split_groups=None):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.split_groups = split_groups


def test_build_tpp_catalog_init_kwargs_filters_unknown_and_dedupes_root_dir(tmp_path):
    base_dir = tmp_path / "PNR"
    init_kwargs = build_tpp_catalog_init_kwargs(
        catalog_ds_class=_BasicCatalog,
        base_dir=base_dir,
        catalog_cfg={
            "root_dir": tmp_path / "custom_root",
            "freq": "1D",
            "normalize": False,
            "unused_flag": "ignored",
        },
    )

    assert init_kwargs == {
        "root_dir": str((tmp_path / "custom_root").resolve()),
        "data_dir": str(base_dir.resolve()),
        "freq": "1D",
        "normalize": False,
    }


def test_build_tpp_catalog_init_kwargs_keeps_explicit_data_dir_for_grouped_catalog(tmp_path):
    base_dir = tmp_path / "SSFS"
    explicit_data_dir = tmp_path / "grouped_data"
    init_kwargs = build_tpp_catalog_init_kwargs(
        catalog_ds_class=_GroupedCatalog,
        base_dir=base_dir,
        catalog_cfg={
            "data_dir": explicit_data_dir,
            "split_groups": {"train": ("A",), "val": ("B",), "test": ("C",)},
            "unused_flag": "ignored",
        },
    )

    assert init_kwargs == {
        "root_dir": str((base_dir / "catalogs").resolve()),
        "data_dir": explicit_data_dir,
        "split_groups": {"train": ("A",), "val": ("B",), "test": ("C",)},
    }


def test_build_tpp_catalog_init_kwargs_preserves_kwargs_for_flexible_catalog(tmp_path):
    base_dir = tmp_path / "Flexible"
    init_kwargs = build_tpp_catalog_init_kwargs(
        catalog_ds_class=_ExtraKwargCatalog,
        base_dir=base_dir,
        catalog_cfg={"freq": "1D", "normalize": False},
    )

    assert init_kwargs == {
        "root_dir": str((base_dir / "catalogs").resolve()),
        "data_dir": str(base_dir.resolve()),
        "freq": "1D",
        "normalize": False,
    }
