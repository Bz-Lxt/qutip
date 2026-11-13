#cython: language_level=3
from qutip.core.data cimport Data
from qutip.solver.integrator._rhs cimport RHS

cpdef enum Status:
    AT_FRONT = 2
    INTERPOLATED = 1
    NORMAL = 0
    TOO_MUCH_WORK = -1
    DT_UNDERFLOW = -2
    OUTSIDE_RANGE = -3
    NOT_INITIATED = -4

cdef class Explicit_RungeKutta:
    cdef RHS deriv

    # Ode state data, set in set_initial_value
    cdef list k
    cdef Data _y_temp, _y, _y_prev, _y_front, _k_fsal
    cdef double _dt_safe, _dt_int
    cdef double _t, _t_prev, _t_front
    cdef Status _status
    cdef dict status_messages

    # options: set in init
    cdef readonly double rtol, atol, first_step, min_step, max_step
    cdef readonly int max_numsteps
    cdef readonly bint interpolate
    # Whether the dense-output polynomial of every accepted step is kept
    # so that ``dense_interpolate`` can query the whole evolved interval.
    cdef readonly bint store_dense_output
    # Recorded dense-output polynomials, one tuple per accepted step:
    # (t0, dt, y_at_t0, [Q_0, ..., Q_{denseout_order - 1}]) with
    # y(t0 + tau * dt) = y_at_t0 + tau * (Q_0 + tau * (Q_1 + ... )) and
    # Q_j = dt * sum_i bi[i, j] * k_i.
    cdef list dense_steps
    cdef list _dense_t0

    # Runge Kutta tableau and info
    cdef int rk_step, rk_extra_step, order, denseout_order
    # Whether variable step according to tolerance are supported
    cdef bint adaptative_step
    # Whether the output can be computed everywhere along the step
    cdef bint can_interpolate
    # Whether the first derivative is the same as the last one of the prevous step
    cdef bint first_same_as_last
    cdef object b_factor_np
    cdef dict butcher_tableau
    cdef double [:] b
    cdef double [:] b_factor
    cdef double [:] c
    cdef double [:] e
    cdef double [:, ::1] a
    cdef double [:, ::1] bi

    cpdef void integrate(Explicit_RungeKutta self, double t, bint step=*) except *

    cpdef Data interpolate_state(Explicit_RungeKutta self, double t)

    cpdef Data dense_interpolate(Explicit_RungeKutta self, double t)

    cpdef void set_initial_value(self, Data y0, double t) except *

    cdef int _step_in_err(self, double t, int max_step) except -1

    cdef double _compute_step(self, double dt) except -1

    cdef double _error(self, Data y_new, double dt) except -1

    cdef void _prep_dense_out(self) except *

    cdef Data _interpolate_step(self, double t, Data out)

    cdef void _record_dense_step(self) except *

    cdef Data _eval_dense(self, tuple record, double t)

    cdef inline Data _accumulate(self, Data target, double[:] factors,
                                 double dt, int size)

    cdef double _estimate_first_step(self, double t, Data y0) except -1

    cdef double _get_timestep(self, double t)

    cdef void _recompute_safe_step(self, double err, double dt)
