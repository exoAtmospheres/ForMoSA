import pytest

from ForMoSA.config.global_config import ConfigParameters
from ForMoSA.parameter.parameter_set import ParameterSet
from ForMoSA.core.enums import ParameterKind

# Regression tests for GitHub issue #34: ConfigParameters had no `av` field,
# so extinction (documented, and fully implemented in the transform pipeline
# via ParameterKind.AV) could never actually be fit through the Python API.


def test_config_parameters_accepts_av():
    cp = ConfigParameters(av=["uniform", "0", "10"])
    assert cp.av == ["uniform", "0", "10"]


def test_av_parses_to_the_right_parameter_kind():
    cp = ConfigParameters(av=["uniform", "0", "10"])
    name, kind, scope, obs_index = cp._parse_param_name("av")
    assert kind == ParameterKind.AV


def test_av_reaches_the_parameter_set():
    cp = ConfigParameters(par1=["uniform", "500", "3000"], av=["uniform", "0", "10"])
    ps = ParameterSet.from_config(cp)

    av_params = [p for p in ps.parameters if p.kind == ParameterKind.AV]
    assert len(av_params) == 1
    assert av_params[0].name == "av"


def test_av_left_unset_does_not_appear():
    cp = ConfigParameters(par1=["uniform", "500", "3000"])
    ps = ParameterSet.from_config(cp)

    assert not any(p.kind == ParameterKind.AV for p in ps.parameters)
