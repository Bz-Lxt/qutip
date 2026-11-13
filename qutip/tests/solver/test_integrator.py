from qutip.solver.sesolve import SESolver
from qutip.solver.mesolve import MESolver
from qutip.solver.mcsolve import MCSolver
from qutip.solver.solver_base import Solver
from qutip.solver.integrator import *
from qutip.solver.integrator._rhs import RHS
import qutip
import qutip.core.data as _data

import functools
import gc
import weakref

import numpy as np
from numpy.testing import assert_allclose
import pytest

# Deactivate warning for test without cython
from qutip.core.coefficient import WARN_MISSING_MODULE
WARN_MISSING_MODULE[0] = 0


class TestIntegratorCallable:

    @pytest.fixture(autouse=True)
    def setup(self, random_generator):
        def derivative_1(t, state, out=None):
            x = state.to_array()
            der = np.array([
                [1, t],
                [x[1, 0] * -0.1, t**2],
                [-0.1j * t * x[2, 0], t**3],
            ])
            der = _data.Dense(der)
            if out is not None:
                out += der
                der = out
            return der

        def analytical_1(t, x0):
            x0 = x0.to_array()
            out = np.array([
                [x0[0, 0] + t, x0[0, 1] + t**2/2],
                [np.exp(t * -0.1) * x0[1, 0] , x0[1, 1] + t**3/3],
                [np.exp(-0.1j * t**2/2) * x0[2, 0], x0[2, 1] + t**4/4],
            ])
            return out

        mat = qutip.rand_herm(3, density=0.75, seed=random_generator) * -0.5j

        derivative_2 = qutip.QobjEvo(mat).matmul_data

        def analytical_2(t, x0):
            return ((t * mat).expm() @ qutip.Qobj(x0)).full()

        derivative_3 = RHS(derivative_1, inplace=True)

        self.tests_cases = {
            "function": (derivative_1, analytical_1),
            "QobjEvo": (derivative_2, analytical_2),
            "RHS": (derivative_3, analytical_1),
        }

    @pytest.fixture(params=[
        method
        for method, integrator in Solver.avail_integrators().items()
        if integrator.rhs_format == "callable"
    ])
    def method(self, request):
        # The callables returned are the general method registered with
        # the base Solver class.
        return request.param

    def _check_integrator(self, integrator_instance, state0, analytical):
        tlist = np.linspace(0, 2, 21)
        integrator_instance.set_state(tlist[0], state0)
        for (t, state), t_in in zip(integrator_instance.run(tlist), tlist[1:]):
            assert t == t_in
            assert_allclose(analytical(t, state0), state.to_array(), atol=5e-5)
            assert state.shape == state0.shape
        t, state = integrator_instance.integrate(3)
        assert t == 3.
        assert_allclose(analytical(t, state0), state.to_array(), atol=5e-5)

        t_front, state = integrator_instance.mcstep(5)
        t_target = (3 + t_front) / 2
        t_in, state = integrator_instance.mcstep(t_target)
        assert t_in == t_target
        assert_allclose(analytical(t_in, state0), state.to_array(), atol=5e-5)

    @pytest.mark.parametrize("case", ["function", "QobjEvo", "RHS"])
    def test_integration_func(self, case, method):
        derivative, analytical = self.tests_cases[case]
        integrator = Solver.avail_integrators()[method]
        integrator_instance = integrator(derivative, {})
        state0 = _data.Dense(np.array([
            [1, 1j],
            [1., 0.25 + 0.75j],
            [0.5 - 0.5j, 0.],
        ]))
        self._check_integrator(integrator_instance, state0, analytical)


class TestIntegratorMatrix:
    @pytest.fixture
    def hermitian(self, random_generator):
        return qutip.rand_herm(10, seed=random_generator).data

    @pytest.fixture
    def non_hermitian(self, random_generator):
        return (
            qutip.rand_stochastic(10, density=0.4, seed=random_generator) +
            qutip.rand_stochastic(10, density=0.4, seed=random_generator) * 1j
        ).data

    @pytest.fixture(params=[
        (IntegratorKrylov, {"krylov_dim": 5}),
        (IntegratorDiag, {}),
    ], ids=["krylov", "diag"])
    def integrator(self, request):
        # Matrix could be limited to hermitian only and not registered with
        # all solver.
        return request.param

    @staticmethod
    def analytical(matrix, t, state0):
        return ((qutip.Qobj(matrix) * t).expm() @ qutip.Qobj(state0)).full()

    def _check_integrator(self, integrator_instance, state0, analytical):
        integrator_instance.set_state(0, state0)
        tlist = np.linspace(0, 2, 21)
        for (t, state), t_in in zip(integrator_instance.run(tlist), tlist[1:]):
            assert t == t_in
            assert_allclose(analytical(t, state0), state.to_array(), atol=2e-5)
            assert state.shape == state0.shape
        t, state = integrator_instance.integrate(3)
        assert t == 3.
        assert_allclose(analytical(t, state0), state.to_array(), atol=2e-5)

    def test_integration_hermitian(
        self, integrator, hermitian, random_generator
    ):
        integrator, options = integrator
        integrator_instance = integrator(hermitian, options)
        state0 = qutip.rand_ket(10, seed=random_generator).data
        self._check_integrator(
            integrator_instance, state0,
            functools.partial(self.analytical, hermitian)
        )

    def test_integration_non_hermitian(
        self, integrator, non_hermitian, random_generator
    ):
        integrator, options = integrator
        integrator_instance = integrator(non_hermitian, options)
        state0 = qutip.rand_ket(10, seed=random_generator).data
        self._check_integrator(
            integrator_instance, state0,
            functools.partial(self.analytical, non_hermitian)
        )


class TestIntegratorMCstep():
    analytical = lambda _, t: np.cos(t * np.pi)
    system = qutip.QobjEvo(-1j * qutip.sigmax() * np.pi)

    @pytest.fixture(params=list(MCSolver.avail_integrators().keys()))
    def mc_method(self, request):
        return request.param

    def test_mc_integration(self, mc_method):
        integrator = MCSolver.avail_integrators()[mc_method]
        if integrator.rhs_format == "callable":
            evol = integrator(self.system.matmul_data, {})
        elif integrator.rhs_format == "matrix":
            evol = integrator(self.system(0).data, {})
        else:
            pytest.skip()
        state = qutip.basis(2,0).data
        evol.set_state(0, state)
        t = 0
        for i in range(1, 21):
            t_target = i * 0.05
            while t < t_target:
                t_old, y_old = evol.get_state()
                t, state = evol.mcstep(t_target)
                assert t <= t_target
                assert t > t_old
                assert_allclose(self.analytical(t),
                                state.to_array()[0, 0], atol=2e-5)
                t_back = (t + t_old) / 2
                t_got, bstate = evol.mcstep(t_back)
                assert t_back == t_got
                assert_allclose(self.analytical(t),
                                state.to_array()[0, 0], atol=2e-5)
                t, state = evol.mcstep(t)


    @pytest.mark.parametrize('start', [1, -1])
    def test_mc_integration_mixed(self, start, mc_method):
        system = qutip.QobjEvo(qutip.qeye(1))
        integrator = MCSolver.avail_integrators()[mc_method]
        if integrator.rhs_format == "callable":
            evol = integrator(system.matmul_data, {})
        elif integrator.rhs_format == "matrix":
            evol = integrator(system(0).data, {})
        else:
            pytest.skip()

        state = qutip.basis(1,0).data
        evol.set_state(start, state)
        t = start
        t_target = start + .1
        while t < t_target:
            t, _ = evol.mcstep(start + .1)
        _ = evol.mcstep(start + .2)
        t_target = (start + .1 + t) / 2
        t, state = evol.mcstep(t_target)
        assert (
            state.to_array()[0, 0]
            == pytest.approx(np.exp(t - start), abs=1e-5)
        )


@pytest.mark.parametrize('sizes', [(1, 100), (10, 10), (100, 0)],
                     ids=["large", "multiple subspaces", "diagonal"])
def test_krylov(sizes):
    # Krylov solve act differently for large systems composed tensored
    # sub systems.
    N, M = sizes
    H = qutip.qeye(N)
    if M:
        H = H & (qutip.num(M) + qutip.create(M) + qutip.destroy(M))
    H = -1j * H
    integrator = IntegratorKrylov(H.data, {"krylov_dim": 30})
    ref_integrator = IntegratorDiag(H.data, {})
    psi = qutip.basis(100, 95).data
    integrator.set_state(0, psi)
    ref_integrator.set_state(0, psi)
    for t in np.linspace(0.25, 1, 4):
        out = integrator.integrate(t)[1]
        ref = ref_integrator.integrate(t)[1]
        assert qutip.data.norm.l2(out - ref) == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize('integrator',
    [IntegratorScipyAdams, IntegratorScipyBDF, IntegratorScipylsoda],
    ids=["adams", 'bdf', "lsoda"]
)
def test_concurent_usage(integrator):
    opt = {'atol':1e-10, 'rtol':1e-7}

    sys1 = qutip.QobjEvo(0.5 * qutip.qeye(1))
    inter1 = integrator(sys1.matmul_data, opt)
    inter1.set_state(0, qutip.basis(1,0).data)

    sys2 = qutip.QobjEvo(-0.5 * qutip.qeye(1))
    inter2 = integrator(sys2.matmul_data, opt)
    inter2.set_state(0, qutip.basis(1,0).data)

    for t in np.linspace(0,1,6):
        expected1 = pytest.approx(np.exp(t/2), abs=1e-5)
        assert inter1.integrate(t)[1].to_array()[0, 0] == expected1
        expected2 = pytest.approx(np.exp(-t/2), abs=1e-5)
        assert inter2.integrate(t)[1].to_array()[0, 0] == expected2


@pytest.mark.parametrize(
    "integrator",
    [
        IntegratorScipyAdams,
        IntegratorScipyBDF,
        IntegratorScipyDop853,
        IntegratorScipylsoda,
    ],
    ids=["adams", "bdf", "dop853", "lsoda"],
)
@pytest.mark.parametrize("initialized", [False, True])
def test_scipy_integrator_does_not_form_reference_cycle(integrator, initialized):
    system = qutip.QobjEvo(0.5 * qutip.qeye(1))
    instance = integrator(system.matmul_data, {})
    if initialized:
        instance.set_state(0, qutip.basis(1, 0).data)
        instance.integrate(0.1)
    reference = weakref.ref(instance)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        del instance
        assert reference() is None
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()


@pytest.mark.parametrize('integrator',
    [IntegratorVern7, IntegratorVern9, IntegratorTsit5],
    ids=["vern7", 'vern9', 'tsit5']
)
def test_pickling_rk_methods(integrator):
    """Test whether VernN and Tsitoura's methods can be pickled and"
    " hence used in multiprocessing"""
    opt = {'atol':1e-10, 'rtol':1e-7}

    sys = qutip.QobjEvo(0.5 * qutip.qeye(1))
    inter = integrator(sys.matmul_data, opt)
    inter.set_state(0, qutip.basis(1,0).data)

    import pickle
    pickled = pickle.dumps(inter, -1)
    recreated = pickle.loads(pickled)
    recreated.set_state(0, qutip.basis(1,0).data)

    for t in np.linspace(0, 1, 6):
        expected = pytest.approx(np.exp(t/2), abs=1e-5)
        result1 = inter.integrate(t)[1].to_array()[0, 0]
        result2 = recreated.integrate(t)[1].to_array()[0, 0]
        assert result1 == result2 == expected

@pytest.mark.parametrize('integrator',
    [IntegratorVern7, IntegratorVern9, IntegratorTsit5],
    ids=["vern7", 'vern9', 'tsit5']
)
def test_rk_options(integrator):
    """Test whether VernN and Tsitoura's methods with no dense output."""
    opt = {
        'atol':1e-10, 'rtol':1e-7, 'interpolate':False, 'first_step':0.5
    }

    sys = qutip.QobjEvo(qutip.qeye(1)).matmul_data
    inter = integrator(sys, opt)
    inter.set_state(0, qutip.basis(1,0).data)

    for t in np.linspace(0, 1, 6):
        expected = pytest.approx(np.exp(t), abs=1e-5)
        result1 = inter.integrate(t)[1].to_array()[0, 0]
        assert result1 == expected


def _rk_dense_system():
    # psi(t) = exp(-i pi sigmax t) |0>
    system = qutip.QobjEvo(-1j * qutip.sigmax() * np.pi)
    psi0 = qutip.basis(2, 0).data

    def analytical(t):
        return (
            qutip.Qobj(-1j * qutip.sigmax() * np.pi * t).expm()
            @ qutip.basis(2, 0)
        ).full()

    return system, psi0, analytical


@pytest.mark.parametrize('integrator',
    [IntegratorVern7, IntegratorVern9, IntegratorTsit5],
    ids=["vern7", 'vern9', 'tsit5']
)
class TestRKDenseOutput:
    def _make(self, integrator, extra_options=None):
        system, psi0, analytical = _rk_dense_system()
        options = {
            'atol': 1e-11, 'rtol': 1e-11,
            'store_dense_output': True,
        }
        if extra_options:
            options.update(extra_options)
        inter = integrator(system.matmul_data, options)
        inter.set_state(0, psi0)
        return inter, analytical

    def test_dense_interpolate_matches_analytical(self, integrator):
        inter, analytical = self._make(integrator)
        tlist = np.linspace(0, 1, 6)
        list(inter.run(tlist))

        query_times = [
            tlist[0], tlist[-1],
            0.5 * (tlist[1] + tlist[2]),
            0.237, 0.581, 0.913,
        ]
        for t in query_times:
            t_got, state = inter.dense_interpolate(t)
            assert t_got == t
            assert_allclose(state.to_array(), analytical(t), atol=1e-8)

    def test_dense_interpolate_matches_inserted_tlist(self, integrator):
        # The queried state must be identical to having the time in tlist.
        t_star = 0.413
        tlist = np.linspace(0, 1, 6)

        inter, _ = self._make(integrator)
        list(inter.run(tlist))
        _, queried = inter.dense_interpolate(t_star)

        system, psi0, _ = _rk_dense_system()
        reference = integrator(
            system.matmul_data,
            {'atol': 1e-11, 'rtol': 1e-11, 'store_dense_output': True},
        )
        reference.set_state(0, psi0)
        list(reference.run(sorted([*tlist, t_star])))
        _, direct = reference.dense_interpolate(t_star)

        assert_allclose(
            queried.to_array(), direct.to_array(), atol=1e-13, rtol=0
        )

    def test_interpolate_is_non_destructive(self, integrator):
        inter, analytical = self._make(integrator)
        inter.integrate(0.7)
        t_before, state_before = inter.get_state(copy=True)

        # Query inside the step currently held by the integrator; the
        # integrator state must not move.
        ode = inter._ode_solver
        t_query = 0.5 * (ode.t_prev + min(0.7, ode.t_front))
        t_got, queried = inter.interpolate(t_query)
        assert t_got == t_query
        assert_allclose(queried.to_array(), analytical(t_query), atol=1e-8)
        t_after, state_after = inter.get_state(copy=True)
        assert t_after == t_before
        assert_allclose(
            state_after.to_array(), state_before.to_array(), rtol=0, atol=0
        )

        # Continuing the evolution must match an integrator without queries.
        system, psi0, _ = _rk_dense_system()
        reference = integrator(
            system.matmul_data, {'atol': 1e-11, 'rtol': 1e-11}
        )
        reference.set_state(0, psi0)
        reference.integrate(0.7)
        for t in [0.8, 0.9, 1.1]:
            _, ref_state = reference.integrate(t)
            _, state = inter.integrate(t)
            assert_allclose(
                state.to_array(), ref_state.to_array(), rtol=0, atol=0
            )

    def test_dense_interpolate_is_non_destructive(self, integrator):
        inter, _ = self._make(integrator)
        tlist = np.linspace(0, 1, 6)
        list(inter.run(tlist))

        system, psi0, analytical = _rk_dense_system()
        reference = integrator(
            system.matmul_data,
            {'atol': 1e-11, 'rtol': 1e-11, 'store_dense_output': True},
        )
        reference.set_state(0, psi0)
        list(reference.run(tlist))

        for t in [0.137, 0.482, 0.836]:
            inter.dense_interpolate(t)
        # The integration state and the continuation are untouched.
        assert inter.get_state()[0] == reference.get_state()[0]
        for t in [1.1, 1.3]:
            _, state = inter.integrate(t)
            _, ref_state = reference.integrate(t)
            assert_allclose(
                state.to_array(), ref_state.to_array(), rtol=0, atol=0
            )

    def test_dense_interpolate_range_errors(self, integrator):
        inter, _ = self._make(integrator)
        list(inter.run(np.linspace(0, 1, 6)))

        with pytest.raises(ValueError):
            inter.dense_interpolate(-0.1)
        with pytest.raises(ValueError):
            inter.dense_interpolate(5.0)

        t_front = inter._ode_solver.t_front
        t_prev = inter._ode_solver.t_prev
        with pytest.raises(ValueError):
            inter.interpolate(t_prev - 0.01)
        with pytest.raises(ValueError):
            inter.interpolate(t_front + 0.01)
        # Endpoints of the current step are valid queries.
        inter.interpolate(t_prev)
        inter.interpolate(t_front)

    def test_dense_output_disabled(self, integrator):
        system, psi0, _ = _rk_dense_system()
        inter = integrator(
            system.matmul_data,
            {'atol': 1e-11, 'rtol': 1e-11, 'store_dense_output': False},
        )
        inter.set_state(0, psi0)
        list(inter.run(np.linspace(0, 1, 6)))
        with pytest.raises(ValueError):
            inter.dense_interpolate(0.5)

    def test_store_dense_output_requires_interpolation(self, integrator):
        system, psi0, _ = _rk_dense_system()
        with pytest.raises(ValueError):
            integrator(
                system.matmul_data,
                {
                    'atol': 1e-11, 'rtol': 1e-11,
                    'interpolate': False,
                    'store_dense_output': True,
                },
            )


def test_dense_output_unsupported_integrator():
    system = qutip.QobjEvo(0.5 * qutip.qeye(1))
    inter = IntegratorScipyAdams(system.matmul_data, {})
    inter.set_state(0, qutip.basis(1, 0).data)
    with pytest.raises(IntegratorException):
        inter.interpolate(0.1)
    with pytest.raises(IntegratorException):
        inter.dense_interpolate(0.1)
