import numpy as np
from WF_HenonHeiles_parallel import WF, dx


def test_full_rothe_error_implementation():
    dim = 3
    cvalsv = [-4.531396 - 1.810139j, 2.505948 - 0.34591j, 2.030252 + 1.668j, 1.47]

    params1 = [
        6.929664e-01,
        3.909027e-02,
        7.515107e-01,
        3.050364e-02,
        3.847112e-02,
        7.227180e-01,
        1.506672e-02,
        2.774190e-02,
        -1.415030e-02,
        -1.766089e-02,
        4.679727e-02,
        -3.541376e-02,
        2.214731e00,
        2.080399e00,
        2.042120e00,
        -6.505042e-03,
        2.629362e-02,
        -3.654459e-03,
    ]
    params2 = [
        7.387447e-01,
        9.086572e-02,
        7.616151e-01,
        6.371750e-02,
        4.517546e-02,
        7.266459e-01,
        -1.223442e-02,
        -2.018877e-02,
        -3.551452e-02,
        -4.674869e-02,
        7.833209e-03,
        -4.309553e-02,
        1.755156e00,
        1.754918e00,
        1.832218e00,
        2.333246e-02,
        5.743388e-02,
        -1.032477e-02,
    ]

    params = np.concatenate((np.array(params1), np.array(params2), params1, params2)).reshape(
        4, -1
    )
    num = len(params) // 2
    h = 0.25

    cvals = np.zeros(2 * num, dtype=complex)
    cvals[:num] = cvalsv[:num]

    wf = WF(params, np.asarray(cvals))
    re = wf.rothe_error(num, h=h)
    rj = wf.rothe_jacobian(num, h=h)

    finite_diff_errors = []
    for i in range(dim * (dim + 3)):
        for j in range(num):
            dparams = params.copy()
            dparams[num + j][i] += dx
            dwf = WF(dparams, np.asarray(cvals))
            numerical = (dwf.rothe_error(num, h=h) - re) / dx
            analytical = rj[dim * (dim + 3) * j + i]
            finite_diff_errors.append(abs(numerical - analytical))

    max_error = float(np.max(finite_diff_errors))
    assert np.isfinite(max_error)
    assert max_error < 1e-2
