import numpy as np
import matplotlib.pyplot as plt

modifiedEuler = True

g = 9.8
mb = 4455.0 + 200 * 8
mw = 325.5
M = mb + 4 * mw
Izz = 34823.2 + 335 + 285 + 1.3**2 * 285 + 3.80**2 * 367.5 + 100 * 8
Ispinf = 10.0
Ispinr = 20.0
d = 1.9
a = 1.25
b = 3.75
Rf = 0.51
Rr = 0.51
base = a + b
h = 1.287  # in doubt

Fzf = mw * g + mb * g * b / 2 / base
Fzr = mw * g + mb * g * a / 2 / base

# B0 = 2.37272
# B1 = -9.46
# B2 = 1490
# B3 = 130
# B4 = 276
# B5 = 0.0886
# B6 = 0.00402
# B7 = -0.0615
# B8 = 1.2
# B9 = 0.0299
# B10 = -0.176

# A0 = 1.65
# A1 = -34
# A2 = 1250
# A3 = 3036
# A4 = 12.8
# A5 = 0.00501
# A6 = -0.02103
# A7 = 0.77394
# A8 = 0.002289
# A9 = 0.013442
# A10 = 0.003709
# A11 = 19.1656
# A12 = 1.21356
# A13 = 6.26206
# A11 = 0.0
# A12 = 0.0
# A13 = 0.0

B0 = 1.765
B1 = -4.184
B2 = 923.8
B3 = 0
B4 = 138.5
B5 = 0
B6 = 0
B7 = -0.003864
B8 = 0.6197
B9 = 0
B10 = 0


A0 = 2.496
A1 = -3.986
A2 = 886.4
A3 = 9319
A4 = 114.4
A5 = 0.00501
A6 = 0
A7 = 0.997
A8 = 0
A9 = 0
A10 = 0
A11 = 0
A12 = 0
A13 = 0


def pacFx(Fz, kappa):
    """
    Fz-kN

    kappa-%
    """
    C = B0
    D = B1 * Fz**2 + B2 * Fz
    BCD = (B3 * Fz**2 + B4 * Fz) * np.exp(-B5 * Fz)
    B = BCD / C / D
    E = B6 * Fz**2 + B7 * Fz + B8
    Sh = B9 * Fz + B10
    Sv = 0
    X1 = kappa + Sh
    return D * np.sin(C * np.arctan(B * X1 - E * (B * X1 - np.arctan(B * X1)))) + Sv


def pacFy(Fz, alpha, gamma=0.0):
    """
    Fz-kN

    alpha-deg
    """
    C = A0
    D = A1 * Fz**2 + A2 * Fz
    BCD = A3 * np.sin(2 * np.arctan(Fz / A4)) * (1 - A5 * abs(gamma))
    B = BCD / C / D
    E = A6 * Fz + A7
    Sh = A8 * gamma + A9 * Fz + A10
    Sv = A11 * Fz * gamma + A12 * Fz + A13
    X1 = alpha + Sh
    return -D * np.sin(C * np.arctan(B * X1 - E * (B * X1 - np.arctan(B * X1)))) + Sv


def pacFxyCombine(kappa, alpha, Fx0, Fy0, eps=1e-3):
    """
    kappa-[-1, 1]

    alpha-rad
    """
    delt_x = abs(kappa / (1 + kappa + eps))
    delt_y = abs(np.tan(alpha) / (1 + kappa + eps))
    delt = (delt_x**2 + delt_y**2) ** 0.5 + eps
    Fx = delt_x / delt * Fx0
    Fy = delt_y / delt * Fy0
    return Fx, Fy


def kappa(R, q, u, eps=1e-3):
    return np.clip((R * q - u) / (max(abs(R * q), abs(u)) + eps), -1, 1)


def alpha(u, v):
    return np.arctan2(v, abs(u))


def init(u, v, r, qfl, qfr, qrl, qrr):
    u[0] = 9.44
    v[0] = 0.0
    r[0] = 0.0
    qfl[0] = u[0] / Rf
    qfr[0] = u[0] / Rf
    qrl[0] = u[0] / Rr
    qrr[0] = u[0] / Rr


endTime = 10.0
timeStep = 0.002
timeSeries = np.arange(0, timeStep + endTime, timeStep, dtype=float)
stepSeries = np.arange(0, timeSeries.size - 1, dtype=int)

u = np.zeros(timeSeries.shape)
v = np.zeros(timeSeries.shape)
r = np.zeros(timeSeries.shape)
qfl = np.zeros(timeSeries.shape)
qfr = np.zeros(timeSeries.shape)
qrl = np.zeros(timeSeries.shape)
qrr = np.zeros(timeSeries.shape)

delta = 0.0 * np.ones(timeSeries.shape)
Tfl = 0.0 * np.ones(timeSeries.shape)
Tfr = 0.0 * np.ones(timeSeries.shape)
Trl = 0.0 * np.ones(timeSeries.shape)
Trr = 0.0 * np.ones(timeSeries.shape)

delta[int(2 / timeStep) : int(3 / timeStep)] = np.linspace(
    0.0, 0.1257, num=int(1 / timeStep)
)
delta[int(3 / timeStep) :] = 0.1257

init(u, v, r, qfl, qfr, qrl, qrr)

du = 0.0
dv = 0.0
dr = 0.0
dqfl = 0.0
dqfr = 0.0
dqrl = 0.0
dqrr = 0.0

u_desired = 0.5 * u[0]

for step in stepSeries:

    # predictor
    ufl = (u[step] - d / 2 * r[step]) * np.cos(delta[step]) + (
        v[step] + a * r[step]
    ) * np.sin(delta[step])
    ufr = (u[step] + d / 2 * r[step]) * np.cos(delta[step]) + (
        v[step] + a * r[step]
    ) * np.sin(delta[step])
    url = u[step] - d / 2 * r[step]
    urr = u[step] + d / 2 * r[step]

    vfl = -(u[step] - d / 2 * r[step]) * np.sin(delta[step]) + (
        v[step] + a * r[step]
    ) * np.cos(delta[step])
    vfr = -(u[step] + d / 2 * r[step]) * np.sin(delta[step]) + (
        v[step] + a * r[step]
    ) * np.cos(delta[step])
    vrl = v[step] - b * r[step]
    vrr = v[step] - b * r[step]

    alphafl = alpha(ufl, vfl)
    alphafr = alpha(ufr, vfr)
    alpharl = alpha(url, vrl)
    alpharr = alpha(urr, vrr)

    kappafl = kappa(Rf, qfl[step], ufl)
    kappafr = kappa(Rf, qfr[step], ufr)
    kapparl = kappa(Rr, qrl[step], url)
    kapparr = kappa(Rr, qrr[step], urr)

    Fzfl = max(
        Fzf
        - mb * h * (du - v[step] * r[step]) / 2 / base
        - mb * h * (dv + u[step] * r[step]) * b / d / base,
        0,
    )
    Fzfr = max(
        Fzf
        - mb * h * (du - v[step] * r[step]) / 2 / base
        + mb * h * (dv + u[step] * r[step]) * b / d / base,
        0,
    )
    Fzrl = max(
        Fzr
        + mb * h * (du - v[step] * r[step]) / 2 / base
        - mb * h * (dv + u[step] * r[step]) * a / d / base,
        0,
    )
    Fzrr = max(
        Fzr
        + mb * h * (du - v[step] * r[step]) / 2 / base
        + mb * h * (dv + u[step] * r[step]) * a / d / base,
        0,
    )

    Fxfl = pacFx(Fzfl / 1000, kappafl * 100)
    Fxfr = pacFx(Fzfr / 1000, kappafr * 100)
    Fxrl = pacFx(Fzrl / 1000, kapparl * 100)
    Fxrr = pacFx(Fzrr / 1000, kapparr * 100)

    Fyfl = pacFy(Fzfl / 1000, alphafl * 180 / np.pi)
    Fyfr = pacFy(Fzfr / 1000, alphafr * 180 / np.pi)
    Fyrl = pacFy(Fzrl / 1000, alpharl * 180 / np.pi)
    Fyrr = pacFy(Fzrr / 1000, alpharr * 180 / np.pi)

    Fxfl, Fyfl = pacFxyCombine(kappafl, alphafl, Fxfl, Fyfl)
    Fxfr, Fyfr = pacFxyCombine(kappafr, alphafr, Fxfr, Fyfr)
    Fxrl, Fyrl = pacFxyCombine(kapparl, alpharl, Fxrl, Fyrl)
    Fxrr, Fyrr = pacFxyCombine(kapparr, alpharr, Fxrr, Fyrr)

    du = (
        (Fxfl + Fxfr) * np.cos(delta[step])
        - (Fyfl + Fyfr) * np.sin(delta[step])
        + Fxrl
        + Fxrr
    ) / M + v[step] * r[step]
    dv = (
        (Fxfl + Fxfr) * np.sin(delta[step])
        + (Fyfl + Fyfr) * np.cos(delta[step])
        + Fyrl
        + Fyrr
    ) / M - u[step] * r[step]
    dr = (
        a * ((Fxfl + Fxfr) * np.sin(delta[step]) + (Fyfl + Fyfr) * np.cos(delta[step]))
        - b * (Fyrl + Fyrr)
        + d
        / 2
        * (
            Fxrr
            - Fxrl
            + (Fxfr - Fxfl) * np.cos(delta[step])
            + (Fyfl - Fyfr) * np.sin(delta[step])
        )
    ) / Izz

    Tfl[step] = 500 * (u_desired - u[step])
    Tfr[step] = 500 * (u_desired - u[step])
    Trl[step] = 500 * (u_desired - u[step])
    Trr[step] = 500 * (u_desired - u[step])

    dqfl = (Tfl[step] - Rf * Fxfl) / Ispinf
    dqfr = (Tfr[step] - Rf * Fxfr) / Ispinf
    dqrl = (Trl[step] - Rr * Fxrl) / Ispinr
    dqrr = (Trr[step] - Rr * Fxrr) / Ispinr

    u_bar = u[step] + timeStep * du
    v_bar = v[step] + timeStep * dv
    r_bar = r[step] + timeStep * dr
    qfl_bar = qfl[step] + timeStep * dqfl
    qfr_bar = qfr[step] + timeStep * dqfr
    qrl_bar = qrl[step] + timeStep * dqrl
    qrr_bar = qrr[step] + timeStep * dqrr

    if not modifiedEuler:
        u[step + 1] = u_bar
        v[step + 1] = v_bar
        r[step + 1] = r_bar
        qfl[step + 1] = qfl_bar
        qfr[step + 1] = qfr_bar
        qrl[step + 1] = qrl_bar
        qrr[step + 1] = qrr_bar
    else:
        # corrector
        ufl = (u_bar - d / 2 * r_bar) * np.cos(delta[step + 1]) + (
            v_bar + a * r_bar
        ) * np.sin(delta[step + 1])
        ufr = (u_bar + d / 2 * r_bar) * np.cos(delta[step + 1]) + (
            v_bar + a * r_bar
        ) * np.sin(delta[step + 1])
        url = u_bar - d / 2 * r_bar
        urr = u_bar + d / 2 * r_bar

        vfl = -(u_bar - d / 2 * r_bar) * np.sin(delta[step + 1]) + (
            v_bar + a * r_bar
        ) * np.cos(delta[step + 1])
        vfr = -(u_bar + d / 2 * r_bar) * np.sin(delta[step + 1]) + (
            v_bar + a * r_bar
        ) * np.cos(delta[step + 1])
        vrl = v_bar - b * r_bar
        vrr = v_bar - b * r_bar

        alphafl = alpha(ufl, vfl)
        alphafr = alpha(ufr, vfr)
        alpharl = alpha(url, vrl)
        alpharr = alpha(urr, vrr)

        kappafl = kappa(Rf, qfl_bar, ufl)
        kappafr = kappa(Rf, qfr_bar, ufr)
        kapparl = kappa(Rr, qrl_bar, url)
        kapparr = kappa(Rr, qrr_bar, urr)

        Fzfl = max(
            Fzf
            - mb * h * (du - v_bar * r_bar) / 2 / base
            - mb * h * (dv + u_bar * r_bar) * b / d / base,
            0,
        )
        Fzfr = max(
            Fzf
            - mb * h * (du - v_bar * r_bar) / 2 / base
            + mb * h * (dv + u_bar * r_bar) * b / d / base,
            0,
        )
        Fzrl = max(
            Fzr
            + mb * h * (du - v_bar * r_bar) / 2 / base
            - mb * h * (dv + u_bar * r_bar) * a / d / base,
            0,
        )
        Fzrr = max(
            Fzr
            + mb * h * (du - v_bar * r_bar) / 2 / base
            + mb * h * (dv + u_bar * r_bar) * a / d / base,
            0,
        )

        Fxfl = pacFx(Fzfl / 1000, kappafl * 100)
        Fxfr = pacFx(Fzfr / 1000, kappafr * 100)
        Fxrl = pacFx(Fzrl / 1000, kapparl * 100)
        Fxrr = pacFx(Fzrr / 1000, kapparr * 100)

        Fyfl = pacFy(Fzfl / 1000, alphafl * 180 / np.pi)
        Fyfr = pacFy(Fzfr / 1000, alphafr * 180 / np.pi)
        Fyrl = pacFy(Fzrl / 1000, alpharl * 180 / np.pi)
        Fyrr = pacFy(Fzrr / 1000, alpharr * 180 / np.pi)

        Fxfl, Fyfl = pacFxyCombine(kappafl, alphafl, Fxfl, Fyfl)
        Fxfr, Fyfr = pacFxyCombine(kappafr, alphafr, Fxfr, Fyfr)
        Fxrl, Fyrl = pacFxyCombine(kapparl, alpharl, Fxrl, Fyrl)
        Fxrr, Fyrr = pacFxyCombine(kapparr, alpharr, Fxrr, Fyrr)

        du_bar = (
            (Fxfl + Fxfr) * np.cos(delta[step + 1])
            - (Fyfl + Fyfr) * np.sin(delta[step + 1])
            + Fxrl
            + Fxrr
        ) / M + v_bar * r_bar
        dv_bar = (
            (Fxfl + Fxfr) * np.sin(delta[step + 1])
            + (Fyfl + Fyfr) * np.cos(delta[step + 1])
            + Fyrl
            + Fyrr
        ) / M - u_bar * r_bar
        dr_bar = (
            a
            * (
                (Fxfl + Fxfr) * np.sin(delta[step + 1])
                + (Fyfl + Fyfr) * np.cos(delta[step + 1])
            )
            - b * (Fyrl + Fyrr)
            + d
            / 2
            * (
                Fxrr
                - Fxrl
                + (Fxfr - Fxfl) * np.cos(delta[step + 1])
                + (Fyfl - Fyfr) * np.sin(delta[step + 1])
            )
        ) / Izz

        Tfl[step + 1] = 500 * (u_desired - u_bar)
        Tfr[step + 1] = 500 * (u_desired - u_bar)
        Trl[step + 1] = 500 * (u_desired - u_bar)
        Trr[step + 1] = 500 * (u_desired - u_bar)

        dqfl_bar = (Tfl[step + 1] - Rf * Fxfl) / Ispinf
        dqfr_bar = (Tfr[step + 1] - Rf * Fxfr) / Ispinf
        dqrl_bar = (Trl[step + 1] - Rr * Fxrl) / Ispinr
        dqrr_bar = (Trr[step + 1] - Rr * Fxrr) / Ispinr

        u[step + 1] = u[step] + timeStep / 2 * (du + du_bar)
        v[step + 1] = v[step] + timeStep / 2 * (dv + dv_bar)
        r[step + 1] = r[step] + timeStep / 2 * (dr + dr_bar)
        qfl[step + 1] = qfl[step] + timeStep / 2 * (dqfl + dqfl_bar)
        qfr[step + 1] = qfr[step] + timeStep / 2 * (dqfr + dqfr_bar)
        qrl[step + 1] = qrl[step] + timeStep / 2 * (dqrl + dqrl_bar)
        qrr[step + 1] = qrr[step] + timeStep / 2 * (dqrr + dqrr_bar)

plt.plot(timeSeries, u, label="u")
plt.plot(timeSeries, v, label="v")
plt.plot(timeSeries, r * 180 / np.pi, label="r")

# plt.plot(np.linspace(-100, 100, num=100), pacFx(7.3575, np.linspace(-100, 100, num=100)))
# plt.plot(np.linspace(-100, 100, num=100), pacFx(14.715, np.linspace(-100, 100, num=100)))
# plt.plot(np.linspace(-100, 100, num=100), pacFx(29.43, np.linspace(-100, 100, num=100)))
# plt.plot(np.linspace(-100, 100, num=100), pacFx(44.145, np.linspace(-100, 100, num=100)))
# plt.plot(np.linspace(-100, 100, num=100), pacFx(58.86, np.linspace(-100, 100, num=100)))
# plt.plot(np.linspace(-90, 90, num=100), pacFy(7.3575, np.linspace(-90, 90, num=100)))
# plt.plot(np.linspace(-90, 90, num=100), pacFy(14.715, np.linspace(-90, 90, num=100)))
# plt.plot(np.linspace(-90, 90, num=100), pacFy(29.43, np.linspace(-90, 90, num=100)))
# plt.plot(np.linspace(-90, 90, num=100), pacFy(44.145, np.linspace(-90, 90, num=100)))
# plt.plot(np.linspace(-90, 90, num=100), pacFy(58.86, np.linspace(-90, 90, num=100)))

plt.legend()
plt.grid()
plt.show()
