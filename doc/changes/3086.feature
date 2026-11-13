Add dense-output interpolation to the Vern7, Vern9 and Tsit5 integrators.
When a solver is run with the new ``store_dense_output`` option, the returned
``Result`` supports ``result.query(t)`` (and the solver supports
``solver.query(t)``) at any time within the integrated interval, returning the
interpolated state (and expectation values when ``e_ops`` are given) without
re-integrating. Queries are non-destructive: they do not change the integrator
state and subsequent ``step`` calls continue exactly as if no query had been
made. Interpolated values agree with adding the queried time directly to
``tlist`` within the integrator tolerance.
