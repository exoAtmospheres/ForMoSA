import pytest
import numpy as np
import pandas as pd

from ForMoSA.parameter.parameter_set import ParameterSet
from ForMoSA.parameter.parameter import Parameter
from ForMoSA.parameter.prior import UniformPrior, ConstantPrior
from ForMoSA.core.enums import ParameterKind
from ForMoSA.core.errors import ForMoSAError


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def uniform_prior():
    return UniformPrior(0.0, 10.0)


@pytest.fixture
def constant_prior():
    return ConstantPrior(5.0)


@pytest.fixture
def free_parameter(uniform_prior):
    return Parameter(
        name="teff",
        kind=ParameterKind.R,
        prior=uniform_prior,
        scope='global',
    )


@pytest.fixture
def fixed_parameter(constant_prior):
    return Parameter(
        name="logg",
        kind=ParameterKind.D,
        prior=constant_prior,
        scope='global',
    )


@pytest.fixture
def parameter_set(free_parameter, fixed_parameter):
    ps = ParameterSet()
    ps.add_parameter(free_parameter)
    ps.add_parameter(fixed_parameter)
    return ps


# ==========================================================
# Tests de base
# ==========================================================

def test_empty_parameterset():
    ps = ParameterSet()
    assert len(ps.parameters) == 0
    assert ps.free_parameters == []
    assert ps.fixed_parameters == []


def test_add_parameter(parameter_set):
    assert len(parameter_set.parameters) == 2
    assert parameter_set.names == ["teff", "logg"]


def test_add_duplicate_parameter_raises(free_parameter):
    ps = ParameterSet()
    ps.add_parameter(free_parameter)

    with pytest.raises(ForMoSAError):
        ps.add_parameter(free_parameter)


def test_add_wrong_type_raises():
    ps = ParameterSet()
    with pytest.raises(ForMoSAError):
        ps.add_parameter("not_a_parameter")


# ==========================================================
# Propriétés
# ==========================================================

def test_free_and_fixed_parameters(parameter_set):
    free = parameter_set.free_parameters
    fixed = parameter_set.fixed_parameters

    assert len(free) == 1
    assert len(fixed) == 1
    assert free[0].name == "teff"
    assert fixed[0].name == "logg"


def test_names_and_kinds(parameter_set):
    assert parameter_set.names == ["teff", "logg"]
    assert all(isinstance(k, ParameterKind) for k in parameter_set.kinds)


# ==========================================================
# prior_transform
# ==========================================================

def test_prior_transform_ok(parameter_set):
    theta = np.array([0.5])  # un seul paramètre libre
    values = parameter_set.prior_transform(theta)

    assert isinstance(values, list)
    assert len(values) == 1
    assert values[0] == pytest.approx(5.0)


def test_prior_transform_wrong_type(parameter_set):
    with pytest.raises(ForMoSAError):
        parameter_set.prior_transform("not_a_list")


def test_prior_transform_wrong_length(parameter_set):
    with pytest.raises(ForMoSAError):
        parameter_set.prior_transform(np.array([0.1, 0.2]))


# ==========================================================
# summary
# ==========================================================

def test_summary_string(parameter_set):
    summary = parameter_set.summary(as_dataframe=False)
    assert isinstance(summary, str)
    assert "teff" in summary
    assert "logg" in summary


def test_summary_dataframe(parameter_set):
    df = parameter_set.summary(as_dataframe=True)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["name", "kind", "prior", "value", "fixed", "scope", "obs_index"]
    assert len(df) == 2

    teff_row = df[df["name"] == "teff"].iloc[0]
    logg_row = df[df["name"] == "logg"].iloc[0]

    assert teff_row["fixed"] == False
    assert logg_row["fixed"] == True
