import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d

# Ackman steering
df_steerL = pd.read_csv("LeftSteer.csv")
deltaL_delta = interp1d(
    df_steerL.iloc[:, 0], df_steerL.iloc[:, 1], kind="linear", fill_value="extrapolate"
)
df_steerR = pd.read_csv("RightSteer.csv")
deltaR_delta = interp1d(
    df_steerR.iloc[:, 0], df_steerR.iloc[:, 1], kind="linear", fill_value="extrapolate"
)

rad2deg = 180 / np.pi
deg2rad = np.pi / 180

wheelNum = 4  # num of wheels
g = 9.8  # gravity
mb = 4455.0 + 200 * 8  # sprung mass
mw = np.array([350.0 / 2, 350 / 2, 760.0 / 2, 760 / 2])  # unsprung mass of single wheel
M = mb + sum(mw)  # total mass
Izz = (
    34823.2 + 335 + 285 + 1.3**2 * 285 + 3.80**2 * 367.5 + 100 * 8
)  # total moment of inertia
Ispin = np.array([10.0, 10.0, 20.0, 20.0])  # spin inertia of wheels
d = 1.9  # wheel track
a = 1.25  # mass center to front axle group
b = 3.75  # mass center to rear axle group
R = np.array([0.51, 0.51, 0.51, 0.51])  # radius of wheel
base = a + b  # wheel base
h = 1.287  # in doubt, mass center height?

Fz_static = (
    np.array(
        [
            mb * g * b / 2 / base,
            mb * g * b / 2 / base,
            mb * g * a / 2 / base,
            mb * g * a / 2 / base,
        ]
    )
    + mw * g
)

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
# A8 = 0.002289
# A9 = -0.0005892
# A10 = -0.1766
# A11 = 19.1656
# A12 = 25.87
# A13 = 104.6
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
    return np.clip(((R * q - u) / (np.maximum(abs(R * q), abs(u)) + eps)), -1, 1)


def alpha(u, v):
    return np.arctan2(v, abs(u))


def init(u, v, r, q):
    u[0] = 9.44
    v[0] = 0.0
    r[0] = 0.0
    q[:, 0] = u[0] / R


endTime = 10.0
timeStep = 0.01
timeSeries = np.arange(0, timeStep + endTime, timeStep, dtype=float)
stepSeries = np.arange(0, timeSeries.size - 1, dtype=int)

u = np.zeros(timeSeries.shape)
v = np.zeros(timeSeries.shape)
r = np.zeros(timeSeries.shape)
q = np.zeros((4, timeSeries.size))

delta = 0.0 * np.ones(timeSeries.shape)
T = 0.0 * np.ones(q.shape)

delta[int(2 / timeStep) : int(3 / timeStep)] = np.linspace(
    0.0, 0.1257, num=int(1 / timeStep)
)
delta[int(3 / timeStep) :] = 0.1257

init(u, v, r, q)

du = 0.0
dv = 0.0
dr = 0.0
dq = np.array([0.0, 0.0, 0.0, 0.0])

u_desired = u[0]

for step in stepSeries:

    deltaL = deltaL_delta(delta[step] * rad2deg) * deg2rad
    deltaR = deltaR_delta(delta[step] * rad2deg) * deg2rad

    uw = np.array(
        [
            [np.cos(deltaL), 0, np.sin(deltaL), 0],
            [0, np.cos(deltaR), 0, np.sin(deltaR)],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
    ) @ np.array(
        [
            u[step] - d / 2 * r[step],
            u[step] + d / 2 * r[step],
            v[step] + a * r[step],
            v[step] + a * r[step],
        ]
    )

    vw = np.array(
        [
            [-np.sin(deltaL), 0, np.cos(deltaL), 0],
            [0, -np.sin(deltaR), np.cos(deltaR), 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ]
    ) @ np.array(
        [
            u[step] - d / 2 * r[step],
            u[step] + d / 2 * r[step],
            v[step] + a * r[step],
            v[step] - b * r[step],
        ]
    )

    alphaw = alpha(uw, vw)

    kappaw = kappa(R, q[:, step], uw)

    Fz = np.maximum(
        Fz_static
        + (mb * h * (du - v[step] * r[step]) / 2 / base) * np.array([-1, -1, 1, 1])
        + (mb * h * (dv + u[step] * r[step]) / d / base) * np.array([-b, b, -a, a]),
        np.zeros(wheelNum),
    )

    Fx = pacFx(Fz / 1000, kappaw * 100)

    Fy = pacFy(Fz / 1000, alphaw * rad2deg)

    Fx, Fy = pacFxyCombine(kappaw, alphaw, Fx, Fy)

    du = (
        np.array([np.cos(deltaL), np.cos(deltaR), 1, 1]) @ Fx
        + np.array([-np.sin(deltaL), -np.sin(deltaR), 0, 0]) @ Fy
    ) / M + v[step] * r[step]

    dv = (
        np.array([np.sin(deltaL), np.sin(deltaR), 0, 0]) @ Fx
        + np.array([np.cos(deltaL), np.cos(deltaR), 1, 1]) @ Fy
    ) / M - u[step] * r[step]

    dr = (
        np.array(
            [
                a * np.sin(deltaL) - d / 2 * np.cos(deltaL),
                a * np.sin(deltaR) + d / 2 * np.cos(deltaR),
                -d / 2,
                d / 2,
            ]
        )
        @ Fx
        + np.array(
            [
                a * np.cos(deltaL) + d / 2 * np.sin(deltaL),
                a * np.cos(deltaR) - d / 2 * np.sin(deltaR),
                -b,
                -b,
            ]
        )
        @ Fy
    ) / Izz

    T[:, step] = 1000 * (u_desired - u[step])

    dq = (T[:, step] - R * Fx) / Ispin

    u[step + 1] = u[step] + timeStep * du
    v[step + 1] = v[step] + timeStep * dv
    r[step + 1] = r[step] + timeStep * dr
    q[:, step + 1] = q[:, step] + timeStep * dq


plt.plot(timeSeries, u, label="u")
plt.plot(timeSeries, v, label="v")
plt.plot(timeSeries, r * 180 / np.pi, label="r")

plt.legend()
plt.grid()
plt.show()
