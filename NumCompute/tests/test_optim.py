import numpy as np
import pytest

from numcompute import optim


# ---------------------------------------------------------------------
# grad: basic / expected behaviour
# ---------------------------------------------------------------------

@pytest.mark.parametrize("method", ["central", "forward"])
def test_grad_quadratic(method):
    """f(x) = x . x has the exact gradient 2x."""
    x = np.array([1.0, -2.0, 3.0])
    g = optim.grad(lambda v: v @ v, x, method=method)
    np.testing.assert_allclose(g, 2 * x, atol=1e-4)


def test_grad_known_function():
    """f(x, y) = x^2 * y + sin(y) has gradient (2xy, x^2 + cos(y))."""
    def f(v):
        return v[0] ** 2 * v[1] + np.sin(v[1])

    x = np.array([1.5, 0.3])
    expected = np.array([2 * 1.5 * 0.3, 1.5 ** 2 + np.cos(0.3)])
    np.testing.assert_allclose(optim.grad(f, x), expected, rtol=1e-7)


def test_central_more_accurate_than_forward():
    """Central difference error ~ h^2, forward ~ h, so central should win clearly."""
    x = np.array([1.0])
    exact = np.exp(1.0)
    central_err = abs(optim.grad(lambda v: np.exp(v[0]), x, h=1e-3, method="central")[0] - exact)
    forward_err = abs(optim.grad(lambda v: np.exp(v[0]), x, h=1e-3, method="forward")[0] - exact)
    assert central_err < forward_err / 100


def test_grad_scalar_input():
    g = optim.grad(lambda v: v ** 3, 2.0)
    assert isinstance(g, float)
    assert g == pytest.approx(12.0, rel=1e-6)


def test_grad_keeps_shape_of_x():
    x = np.arange(6.0).reshape(2, 3)
    g = optim.grad(lambda v: np.sum(v ** 2), x)
    assert g.shape == (2, 3)
    np.testing.assert_allclose(g, 2 * x, atol=1e-4)


# ---------------------------------------------------------------------
# grad: edge cases
# ---------------------------------------------------------------------

def test_grad_integer_input():
    """Integer x must be converted to float, otherwise x + h would be truncated to x."""
    g = optim.grad(lambda v: v @ v, [1, 2])
    np.testing.assert_allclose(g, [2.0, 4.0], atol=1e-4)


def test_grad_non_contiguous_input():
    x = np.array([1.0, 99.0, 2.0, 99.0, 3.0])[::2]   # strided view [1, 2, 3]
    np.testing.assert_allclose(optim.grad(lambda v: v @ v, x), [2.0, 4.0, 6.0], atol=1e-4)


def test_grad_does_not_mutate_input():
    x = np.array([1.0, 2.0])
    x_copy = x.copy()
    optim.grad(lambda v: v @ v, x)
    np.testing.assert_array_equal(x, x_copy)


def test_grad_constant_function_is_zero():
    np.testing.assert_allclose(optim.grad(lambda v: 5.0, np.ones(4)), np.zeros(4), atol=1e-12)


# ---------------------------------------------------------------------
# grad: error handling
# ---------------------------------------------------------------------

def test_grad_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be 'central' or 'forward'"):
        optim.grad(lambda v: v @ v, [1.0], method="backward")


@pytest.mark.parametrize("h", [0, -1e-5, np.nan, np.inf, "small"])
def test_grad_invalid_h_raises(h):
    with pytest.raises(ValueError, match="h must be a positive finite number"):
        optim.grad(lambda v: v @ v, [1.0], h=h)


def test_grad_empty_x_raises():
    with pytest.raises(ValueError, match="x must not be empty."):
        optim.grad(lambda v: 0.0, np.array([]))


def test_grad_nan_x_raises():
    with pytest.raises(ValueError, match="x must contain only finite values."):
        optim.grad(lambda v: v @ v, [1.0, np.nan])


def test_grad_non_numeric_x_raises():
    with pytest.raises(ValueError, match="x must be numeric"):
        optim.grad(lambda v: 0.0, ["a", "b"])


def test_grad_vector_valued_f_raises():
    with pytest.raises(ValueError, match="f must return a single number"):
        optim.grad(lambda v: v * 2, [1.0, 2.0])


# ---------------------------------------------------------------------
# jacobian
# ---------------------------------------------------------------------

def test_jacobian_linear_map():
    """For F(x) = A x the Jacobian is exactly A."""
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    J = optim.jacobian(lambda v: A @ v, np.array([0.5, -1.0]))
    assert J.shape == (3, 2)
    np.testing.assert_allclose(J, A, atol=1e-8)


@pytest.mark.parametrize("method", ["central", "forward"])
def test_jacobian_nonlinear(method):
    """F(x, y) = (x * y, sin(x)) -> J = [[y, x], [cos(x), 0]]."""
    def F(v):
        return np.array([v[0] * v[1], np.sin(v[0])])

    x = np.array([0.7, 2.0])
    expected = np.array([[2.0, 0.7], [np.cos(0.7), 0.0]])
    np.testing.assert_allclose(optim.jacobian(F, x, method=method), expected, atol=1e-4)


def test_jacobian_scalar_function_is_row_gradient():
    J = optim.jacobian(lambda v: v @ v, np.array([1.0, 2.0]))
    assert J.shape == (1, 2)
    np.testing.assert_allclose(J, [[2.0, 4.0]], atol=1e-6)


def test_jacobian_matches_grad():
    def f(v):
        return np.sum(np.sin(v) * v)

    x = np.array([0.1, 0.5, 2.0])
    np.testing.assert_allclose(optim.jacobian(f, x)[0], optim.grad(f, x), rtol=1e-12)


def test_jacobian_changing_output_size_raises():
    def F(v):
        return np.ones(3) if v[0] > 1.0 else np.ones(2)

    with pytest.raises(ValueError, match="expected 2"):
        optim.jacobian(F, np.array([1.0]))


def test_jacobian_invalid_method_raises():
    with pytest.raises(ValueError, match="method must be 'central' or 'forward'"):
        optim.jacobian(lambda v: v, [1.0], method="sideways")
