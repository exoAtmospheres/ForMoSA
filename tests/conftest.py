from pathlib import Path

import pytest

from ForMoSA.core import config as _config

# Tests construct real PhotometryFilter objects (e.g. Keck/NIRC2.Lp). By default
# that hits the SVO Filter Profile Service over the network on a cache miss,
# which is unreliable in CI. Point the filter cache at a fixture directory
# committed to the repo so the whole test suite resolves filters locally,
# with no network dependency.
FIXTURE_FILTER_PATH = Path(__file__).parent / "fixtures" / "filters"


@pytest.fixture(autouse=True, scope="session")
def _use_fixture_filter_path():
    original = _config.FILTER_PATH
    _config.set_filter_path(FIXTURE_FILTER_PATH)
    yield
    _config.set_filter_path(original)
