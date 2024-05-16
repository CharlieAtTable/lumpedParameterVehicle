import numpy as np
import matplotlib.pyplot as plt

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
h = 0.5  # in doubt

Fzf = mw * g + mb * g * b / 2 / base
Fzr = mw * g + mb * g * a / 2 / base

# kf = -1.5
# kr = -1.5
# muf = 0.7188
# mur = 0.7188

B0 = 2.37272
B1 = -9.46
B2 = 1490
B3 = 130
B4 = 276
B5 = 0.0886
B6 = 0.00402
B7 = -0.0615
B8 = 1.2
B9 = 0.0299
B10 = -0.176

A0 = 1.65
A1 = -34
A2 = 1250
A3 = 3036
A4 = 12.8
A5 = 0.00501
A6 = -0.02103
A7 = 0.77394
A8 = 0.002289
A9 = 0.013442
A10 = 0.003709
A11 = 19.1656
A12 = 1.21356
A13 = 6.26206


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


def pacFy(Fz, alpha, gamma=0):
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


def kappa(R, q, u, eps=1e-3):
    return np.clip((R * q - u) / (max(abs(R * q), abs(u)) + eps), -1, 1)


def alpha(u, v):
    return np.arctan2(v, abs(u))


def init(u, v, r, qfl, qfr, qrl, qrr):
    u[0] = 10.0
    v[0] = 0.0
    r[0] = 0.0
    qfl[0] = u[0] / Rf
    qfr[0] = u[0] / Rf
    qrl[0] = u[0] / Rr
    qrr[0] = u[0] / Rr


endTime = 10.0
timeStep = 0.001
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

delta[int(1 / timeStep) : int(2 / timeStep)] = np.linspace(
    0.0, 0.2, num=int(1 / timeStep)
)
delta[int(2 / timeStep) :] = 0.2

init(u, v, r, qfl, qfr, qrl, qrr)

du = 0.0
dv = 0.0
dr = 0.0
dqfl = 0.0
dqfr = 0.0
dqrl = 0.0
dqrr = 0.0

for step in stepSeries:

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

    Fzfl = (
        Fzf
        - mb * h * (du - v[step] * r[step]) / 2 / base
        - mb * h * (dv + u[step] * r[step]) * b / d / base
    )
    Fzfr = (
        Fzf
        - mb * h * (du - v[step] * r[step]) / 2 / base
        + mb * h * (dv + u[step] * r[step]) * b / d / base
    )
    Fzrl = (
        Fzr
        + mb * h * (du - v[step] * r[step]) / 2 / base
        - mb * h * (dv + u[step] * r[step]) * a / d / base
    )
    Fzrr = (
        Fzr
        + mb * h * (du - v[step] * r[step]) / 2 / base
        + mb * h * (dv + u[step] * r[step]) * a / d / base
    )

    Fxfl = pacFx(Fzfl / 1000 / 5, kappafl * 100)
    Fxfr = pacFx(Fzfr / 1000 / 5, kappafr * 100)
    Fxrl = pacFx(Fzrl / 1000 / 5, kapparl * 100)
    Fxrr = pacFx(Fzrr / 1000 / 5, kapparr * 100)

    Fyfl = pacFy(Fzfl / 1000 / 5, alphafl * 180 / np.pi)
    Fyfr = pacFy(Fzfr / 1000 / 5, alphafr * 180 / np.pi)
    Fyrl = pacFy(Fzrl / 1000 / 5, alpharl * 180 / np.pi)
    Fyrr = pacFy(Fzrr / 1000 / 5, alpharr * 180 / np.pi)

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
    dqfl = (Tfl[step] - Rf * Fxfl) / Ispinf
    dqfr = (Tfr[step] - Rf * Fxfr) / Ispinf
    dqrl = (Trl[step] - Rr * Fxrl) / Ispinr
    dqrr = (Trr[step] - Rr * Fxrr) / Ispinr

    u[step + 1] = u[step] + timeStep * du
    v[step + 1] = v[step] + timeStep * dv
    r[step + 1] = r[step] + timeStep * dr
    qfl[step + 1] = qfl[step] + timeStep * dqfl
    qfr[step + 1] = qfr[step] + timeStep * dqfr
    qrl[step + 1] = qrl[step] + timeStep * dqrl
    qrr[step + 1] = qrr[step] + timeStep * dqrr

plt.plot(timeSeries, u)
plt.plot(timeSeries, v)
plt.plot(timeSeries, r * 180 / np.pi)
plt.show()
