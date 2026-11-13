import pytest
import pickle
import qutip
import numpy as np
from qutip.solver.sesolve import sesolve, SESolver
from qutip.solver.krylovsolve import krylovsolve
from qutip.solver.solver_base import Solver
from qutip.solver.integrator import IntegratorException

# Deactivate warning for test without cython
from qutip.core.coefficient import WARN_MISSING_MODULE
WARN_MISSING_MODULE[0] = 0


all_ode_method = [
    method for method, integrator in SESolver.avail_integrators().items()
    if integrator.rhs_format == "callable"
]

def _analytic(t, alpha):
    return ((1 - np.exp(-alpha * t)) / alpha)


class TestSeSolve():
    H0 = 0.2 * np.pi * qutip.sigmaz()
    H1 = np.pi * qutip.sigmax()
    tlist = np.linspace(0, 20, 200)
    args = {'alpha': 0.5}
    w_a = 0.35
    a = 0.5

    @pytest.mark.parametrize(['unitary_op'], [
        pytest.param(None, id="state"),
        pytest.param(qutip.qeye(2), id="unitary"),
    ])
    @pytest.mark.parametrize(['H', 'analytical'], [
        pytest.param(H1, lambda t, _: t, id='const_H'),
        pytest.param(lambda t, alpha: (np.pi * qutip.sigmax()
                                       * np.exp(-alpha * t)),
                     _analytic, id='func_H'),
        pytest.param([[H1, lambda t, alpha: np.exp(-alpha * t)]],
                     _analytic, id='list_func_H'),
        pytest.param([[H1, 'exp(-alpha*t)']],
                     _analytic, id='list_str_H'),
        pytest.param([[H1, np.exp(-args['alpha'] * tlist)]],
                     _analytic, id='list_array_H'),
        pytest.param(qutip.QobjEvo([[H1, 'exp(-alpha*t)']], args=args),
                     _analytic, id='QobjEvo_H'),
    ])
    def test_sesolve(self, H, analytical, unitary_op):
        """
        Compare integrated evolution with analytical result
        If U0 is not None then operator evo is checked
        Otherwise state evo
        """
        tol = 5e-3
        psi0 = qutip.basis(2, 0)
        options = {"progress_bar": None}

        if unitary_op is None:
            output = sesolve(
                H, psi0, self.tlist,
                e_ops=[qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()],
                args=self.args, options=options
            )
            sx, sy, sz = output.expect[0], output.expect[1], output.expect[2]
        else:
            output = sesolve(H, unitary_op, self.tlist, args=self.args,
                             options=options)
            sx = [qutip.expect(qutip.sigmax(), U * psi0)
                  for U in output.states]
            sy = [qutip.expect(qutip.sigmay(), U * psi0)
                  for U in output.states]
            sz = [qutip.expect(qutip.sigmaz(), U * psi0)
                  for U in output.states]

        sx_analytic = np.zeros(np.shape(self.tlist))
        sy_analytic = np.array(
            [-np.sin(2 * np.pi * analytical(t, self.args['alpha']))
             for t in self.tlist]
        )
        sz_analytic = np.array(
            [np.cos(2 * np.pi * analytical(t, self.args['alpha']))
             for t in self.tlist]
        )

        np.testing.assert_allclose(sx, sx_analytic, atol=tol)
        np.testing.assert_allclose(sy, sy_analytic, atol=tol)
        np.testing.assert_allclose(sz, sz_analytic, atol=tol)

    @pytest.mark.parametrize(['state_type'], [
        pytest.param("ket", id="ket"),
        pytest.param("unitary", id="unitary"),
    ])
    def test_sesolve_normalization(self, state_type):
        # non-hermitean H causes state to evolve non-unitarily
        H = qutip.Qobj([[1, -0.1j], [-0.1j, 1]])
        psi0 = qutip.basis(2, 0)
        options = {"normalize_output": True, "progress_bar": None}

        if state_type == "ket":
            output = sesolve(H, psi0, self.tlist, e_ops=[], options=options)
            norms = [state.norm() for state in output.states]
            np.testing.assert_allclose(
                norms, [1.0 for _ in self.tlist], atol=1e-15,
            )
        else:
            # evolution of unitaries should not be normalized
            U = qutip.qeye(2)
            output = sesolve(H, U, self.tlist, e_ops=[], options=options)
            norms = [state.norm() for state in output.states]
            assert all(norm > 2 for norm in norms[1:])

    @pytest.mark.parametrize(['unitary_op'], [
        pytest.param(None, id="state"),
        pytest.param(qutip.qeye(2), id="unitary"),
    ])
    @pytest.mark.parametrize('method', all_ode_method, ids=all_ode_method)
    def test_sesolve_method(self, method, unitary_op):
        """
        Compare integrated evolution with analytical result
        If U0 is not None then operator evo is checked
        Otherwise state evo
        """
        tol = 5e-3
        psi0 = qutip.basis(2, 0)
        options = {"method": method, "progress_bar": None}
        H = [[self.H1, 'exp(-alpha*t)']]

        if unitary_op is None:
            output = sesolve(
                H, psi0, self.tlist,
                e_ops=[qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()],
                args=self.args, options=options
            )
            sx, sy, sz = output.expect[0], output.expect[1], output.expect[2]
        else:
            output = sesolve(H, unitary_op, self.tlist,
                             args=self.args, options=options)
            sx = [qutip.expect(qutip.sigmax(), U * psi0)
                  for U in output.states]
            sy = [qutip.expect(qutip.sigmay(), U * psi0)
                  for U in output.states]
            sz = [qutip.expect(qutip.sigmaz(), U * psi0)
                  for U in output.states]

        sx_analytic = np.zeros(np.shape(self.tlist))
        sy_analytic = np.array(
            [np.sin(-2*np.pi * _analytic(t, self.args['alpha']))
             for t in self.tlist]
        )
        sz_analytic = np.array(
            [np.cos(2*np.pi *_analytic(t, self.args['alpha']))
             for t in self.tlist]
        )

        np.testing.assert_allclose(sx, sx_analytic, atol=tol)
        np.testing.assert_allclose(sy, sy_analytic, atol=tol)
        np.testing.assert_allclose(sz, sz_analytic, atol=tol)

    @pytest.mark.parametrize('normalize', [True, False],
                             ids=['Normalized', ''])
    @pytest.mark.parametrize(['H', 'args'],
        [pytest.param(H0 + H1, {}, id='const_H'),
         pytest.param(lambda t, a, w_a: (
             a * t * 0.2 * np.pi * qutip.sigmaz() +
             np.cos(w_a * t) * np.pi * qutip.sigmax()
         ), {'a':a, 'w_a':w_a}, id='func_H'),
         pytest.param([
             [H0, lambda t, a: a * t],
             [H1, lambda t, w_a: np.cos(w_a * t)]
         ], {'a':a, 'w_a':w_a}, id='list_func_H'),
         pytest.param([H0, [H1, 'cos(w_a*t)']], {'w_a':w_a}, id='list_str_H'),
    ])
    def test_compare_evolution(self, H, normalize, args, tol=5e-5):
        """
        Compare integrated evolution of unitary operator with state evo
        """
        psi0 = qutip.basis(2, 0)
        U0 = qutip.qeye(2)

        options = {
            "store_states": True,
            "normalize_output": normalize,
            "progress_bar": None
        }
        out_s = sesolve(
            H, psi0, self.tlist,
            e_ops=[qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()],
            options=options, args=args
        )
        xs, ys, zs = out_s.expect[0], out_s.expect[1], out_s.expect[2]
        xss = [qutip.expect(qutip.sigmax(), U) for U in out_s.states]
        yss = [qutip.expect(qutip.sigmay(), U) for U in out_s.states]
        zss = [qutip.expect(qutip.sigmaz(), U) for U in out_s.states]

        np.testing.assert_allclose(xs, xss, atol=tol)
        np.testing.assert_allclose(ys, yss, atol=tol)
        np.testing.assert_allclose(zs, zss, atol=tol)

        if normalize:
            # propagator evolution is not normalized (yet?)
            tol = 5e-4
        out_u = sesolve(H, U0, self.tlist, options=options, args=args)
        xu = [qutip.expect(qutip.sigmax(), U * psi0) for U in out_u.states]
        yu = [qutip.expect(qutip.sigmay(), U * psi0) for U in out_u.states]
        zu = [qutip.expect(qutip.sigmaz(), U * psi0) for U in out_u.states]

        np.testing.assert_allclose(xs, xu, atol=tol)
        np.testing.assert_allclose(ys, yu, atol=tol)
        np.testing.assert_allclose(zs, zu, atol=tol)

    def test_sesolver_args(self):
        options = {"progress_bar": None}
        solver_obj = SESolver(qutip.QobjEvo([self.H0, [self.H1,'a']],
                                            args={'a': 1}),
                              options=options)
        res = solver_obj.run(qutip.basis(2,1), [0, 1, 2, 3],
                             e_ops=[qutip.num(2)], args={'a':0})
        np.testing.assert_allclose(res.expect[0], 1)

    def test_sesolver_pickling(self):
        e_ops = [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
        options = {"progress_bar": None}
        solver_obj = SESolver(self.H0 + self.H1,
                              options=options)
        solver_copy = pickle.loads(pickle.dumps(solver_obj))
        sx, sy, sz = solver_obj.run(qutip.basis(2,1), [0, 1, 2, 3],
                                    e_ops=e_ops).expect
        csx, csy, csz = solver_copy.run(qutip.basis(2,1), [0, 1, 2, 3],
                                       e_ops=e_ops).expect
        np.testing.assert_allclose(sx, csx)
        np.testing.assert_allclose(sy, csy)
        np.testing.assert_allclose(sz, csz)

    @pytest.mark.parametrize('method', all_ode_method, ids=all_ode_method)
    def test_sesolver_stepping(self, method):
        options = {
            "method": method,
            "atol": 1e-7,
            "rtol": 1e-8,
            "progress_bar": None
        }
        solver_obj = SESolver(
            qutip.QobjEvo([self.H1, lambda t, a: a], args={"a":0.25}),
            options=options
        )
        solver_obj.start(qutip.basis(2,0), 0)
        sr2 = -(2**.5)/2
        state = solver_obj.step(1)
        np.testing.assert_allclose(qutip.expect(qutip.sigmax(), state), 0.,
                                   atol=2e-6)
        np.testing.assert_allclose(qutip.expect(qutip.sigmay(), state), -1,
                                   atol=2e-6)
        np.testing.assert_allclose(qutip.expect(qutip.sigmaz(), state), 0.,
                                   atol=2e-6)

        state = solver_obj.step(2, args={"a":0.125})
        np.testing.assert_allclose(qutip.expect(qutip.sigmax(), state), 0.,
                                   atol=2e-6)
        np.testing.assert_allclose(qutip.expect(qutip.sigmay(), state), sr2,
                                   atol=2e-6)
        np.testing.assert_allclose(qutip.expect(qutip.sigmaz(), state), sr2,
                                   atol=2e-6)

        solver_obj.options = {
            "method": "adams",
            "atol": 1e-7,
            "rtol": 1e-8,
            "progress_bar": None
        }
        state = solver_obj.step(3, args={"a":0})
        np.testing.assert_allclose(qutip.expect(qutip.sigmax(), state), 0.,
                                   atol=2e-6)
        np.testing.assert_allclose(qutip.expect(qutip.sigmay(), state), sr2,
                                   atol=2e-6)
        np.testing.assert_allclose(qutip.expect(qutip.sigmaz(), state), sr2,
                                   atol=2e-6)


def test_sesolve_bad_H():
    with pytest.raises(TypeError):
        SESolver(np.eye(3))
    with pytest.raises(ValueError):
        SESolver(qutip.basis(3,1))


def test_sesolve_bad_state():
    solver = SESolver(qutip.qeye(4))
    with pytest.raises(TypeError):
        solver.start(qutip.basis(4,1).dag(), 0)
    with pytest.raises(TypeError):
        solver.start(qutip.basis(2,1) & qutip.basis(2,0), 0)


def test_sesolve_step_no_start():
    solver = SESolver(qutip.qeye(4))
    with pytest.raises(RuntimeError):
        solver.step(1)


@pytest.mark.parametrize("algorithm", ['lanczos', 'lanczos_fro'])
def test_krylovsolve_pure_state(algorithm, random_generator):
    H = qutip.rand_herm([2] * 8, seed=random_generator)
    psi0 = qutip.basis([2] * 8, [1] * 8)
    e_op = qutip.num(256)
    e_op.dims = H.dims
    tlist = np.linspace(0, 1, 11)
    ref = sesolve(H, psi0, tlist, e_ops=[e_op]).expect[0]
    options = {"store_states": True, "algorithm": algorithm}
    krylov_sol = krylovsolve(H, psi0, tlist, 20, e_ops=[e_op], options=options)
    np.testing.assert_allclose(np.ones(len(krylov_sol.states)),
                               [s.norm() for s in krylov_sol.states])
    np.testing.assert_allclose(ref, krylov_sol.expect[0], atol=1e-7, rtol=1e-6)


def test_krylovsolve_error():
    H = qutip.rand_herm(10, density=0.2)
    psi0 = qutip.basis([10], [9])
    tlist = np.linspace(0, 1, 11)
    with pytest.raises(ValueError) as err:
        krylovsolve(H, psi0, tlist, 8, algorithm="foo_bar")
    assert "The requested algorithm" in str(err.value)


def test_feedback():

    def f(t, A, qobj=None):
        return (A-2.)

    N = 4
    tol = 1e-14
    psi0 = qutip.basis(N, N-1)
    a = qutip.destroy(N)
    H = qutip.QobjEvo(
        [qutip.num(N), [a+a.dag(), f]],
        args={"A": SESolver.ExpectFeedback(qutip.num(N), default=3.)}
    )
    solver = qutip.SESolver(H)
    result = solver.run(psi0, np.linspace(0, 30, 301), e_ops=[qutip.num(N)])
    assert np.all(result.expect[0] > 2 - tol)


@pytest.mark.parametrize("method", ["vern7", "vern9", "tsit5"])
class TestSESolverDenseOutput:
    H = 0.2 * np.pi * qutip.sigmaz() + np.pi * qutip.sigmax()
    psi0 = qutip.basis(2, 0)
    e_ops = [qutip.sigmax(), qutip.sigmaz()]
    tlist = np.linspace(0, 2, 7)
    query_times = [0.137, 0.582, 1.236, 1.841]

    def _options(self, method):
        return {
            "method": method,
            "store_dense_output": True,
            "atol": 1e-11,
            "rtol": 1e-11,
            "progress_bar": None,
        }

    def test_query_matches_inserted_tlist(self, method):
        solver = SESolver(self.H, options=self._options(method))
        result = solver.run(self.psi0, self.tlist, e_ops=self.e_ops)

        for t in self.query_times:
            state, expect = result.query(t)

            direct = sesolve(
                self.H, self.psi0, sorted([*self.tlist, t]),
                e_ops=self.e_ops,
                options={
                    "method": method, "atol": 1e-11, "rtol": 1e-11,
                    "store_states": True, "progress_bar": None,
                },
            )
            idx = list(direct.times).index(t)
            np.testing.assert_allclose(
                state.full(), direct.states[idx].full(), atol=1e-12, rtol=0
            )
            for i, e_op in enumerate(self.e_ops):
                np.testing.assert_allclose(
                    expect[i], direct.expect[i][idx], atol=1e-12, rtol=0
                )

    def test_query_without_e_ops_returns_state(self, method):
        solver = SESolver(self.H, options=self._options(method))
        result = solver.run(self.psi0, self.tlist)
        state = result.query(0.42)
        assert isinstance(state, qutip.Qobj)
        assert state.dims == self.psi0.dims

        direct = sesolve(
            self.H, self.psi0, [0, 0.42],
            options={"method": method, "atol": 1e-11, "rtol": 1e-11},
        )
        np.testing.assert_allclose(
            state.full(), direct.states[-1].full(), atol=1e-12, rtol=0
        )

    def test_solver_query_matches_result_query(self, method):
        solver = SESolver(self.H, options=self._options(method))
        result = solver.run(self.psi0, self.tlist, e_ops=self.e_ops)
        for t in self.query_times:
            state_solver = solver.query(t)
            state_result, _ = result.query(t)
            np.testing.assert_allclose(
                state_solver.full(), state_result.full(), rtol=0, atol=0
            )

    def test_query_does_not_affect_subsequent_step(self, method):
        options = self._options(method)
        solver_ref = SESolver(self.H, options=options)
        solver_ref.start(self.psi0, self.tlist[0])
        solver_query = SESolver(self.H, options=options)
        solver_query.start(self.psi0, self.tlist[0])

        for t in self.tlist[1:]:
            ref_state = solver_ref.step(t).full()
            query_state = solver_query.step(t).full()
            np.testing.assert_allclose(
                query_state, ref_state, rtol=0, atol=0
            )
            # Queries between steps must leave the evolution untouched.
            solver_query.query(t - 0.031)

        # And the next step beyond the original tlist is still identical.
        np.testing.assert_allclose(
            solver_query.step(2.3).full(),
            solver_ref.step(2.3).full(),
            rtol=0, atol=0,
        )

    def test_query_without_option_raises(self, method):
        result = sesolve(
            self.H, self.psi0, self.tlist,
            options={"method": method, "progress_bar": None},
        )
        with pytest.raises(RuntimeError):
            result.query(0.5)

        solver = SESolver(
            self.H, options={"method": method, "progress_bar": None}
        )
        solver.start(self.psi0, self.tlist[0])
        solver.step(self.tlist[1])
        # RK integrator without recorded dense output raises ValueError.
        with pytest.raises(ValueError):
            solver.query(0.05)

        # Integrators without any dense-output support raise
        # IntegratorException instead.
        solver_scipy = SESolver(
            self.H, options={"method": "dop853", "progress_bar": None}
        )
        solver_scipy.start(self.psi0, self.tlist[0])
        solver_scipy.step(self.tlist[1])
        with pytest.raises(IntegratorException):
            solver_scipy.query(0.05)

    def test_query_outside_tlist_raises(self, method):
        solver = SESolver(self.H, options=self._options(method))
        result = solver.run(self.psi0, self.tlist)
        with pytest.raises(ValueError):
            result.query(self.tlist[0] - 0.1)
        with pytest.raises(ValueError):
            result.query(self.tlist[-1] + 0.1)

    def test_query_at_tlist_points(self, method):
        solver = SESolver(self.H, options=self._options(method))
        result = solver.run(self.psi0, self.tlist, e_ops=self.e_ops)
        direct = sesolve(
            self.H, self.psi0, self.tlist,
            e_ops=self.e_ops,
            options={
                "method": method, "atol": 1e-11, "rtol": 1e-11,
                "store_states": True, "progress_bar": None,
            },
        )
        for i, t in enumerate(self.tlist):
            state, expect = result.query(t)
            np.testing.assert_allclose(
                state.full(), direct.states[i].full(), atol=1e-12, rtol=0
            )
            for k in range(len(self.e_ops)):
                np.testing.assert_allclose(
                    expect[k], direct.expect[k][i], atol=1e-12, rtol=0
                )
