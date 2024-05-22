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
timeStep = 0.001
timeSeries = np.arange(0, timeStep + endTime, timeStep, dtype=float)
stepSeries = np.arange(0, timeSeries.size - 1, dtype=int)

u = np.zeros(timeSeries.shape)
v = np.zeros(timeSeries.shape)
r = np.zeros(timeSeries.shape)
q = np.zeros((wheelNum, timeSeries.size))

delta = 0.0 * np.ones(timeSeries.shape)
T = 0.0 * np.ones(q.shape)
delta[int(2 / timeStep) : int(3 / timeStep)] = np.linspace(
    0.0, 0.1257, num=int(1 / timeStep)
)
delta[int(3 / timeStep) :] = 0.1257

init(u, v, r, q)

du = np.zeros(timeSeries.shape)
dv = np.zeros(timeSeries.shape)
dr = np.zeros(timeSeries.shape)
dq = np.zeros((wheelNum, timeSeries.size))

u_p = np.zeros(timeSeries.shape)
v_p = np.zeros(timeSeries.shape)
r_p = np.zeros(timeSeries.shape)
q_p = np.zeros((wheelNum, timeSeries.size))
u_pm = np.zeros(timeSeries.shape)
v_pm = np.zeros(timeSeries.shape)
r_pm = np.zeros(timeSeries.shape)
q_pm = np.zeros((wheelNum, timeSeries.size))
du_pm = np.zeros(timeSeries.shape)
dv_pm = np.zeros(timeSeries.shape)
dr_pm = np.zeros(timeSeries.shape)
dq_pm = np.zeros((wheelNum, timeSeries.size))
u_c = np.zeros(timeSeries.shape)
v_c = np.zeros(timeSeries.shape)
r_c = np.zeros(timeSeries.shape)
q_c = np.zeros((wheelNum, timeSeries.size))

u_desired = u[0]


def advance(delta, u_desired, u, v, r, q, du, dv):
    """
    input: delta, u_desired

    last_state: u, v, r, q, du, dv

    return: du, dv, dr, dq
    """
    deltaL = deltaL_delta(delta * rad2deg) * deg2rad
    deltaR = deltaR_delta(delta * rad2deg) * deg2rad
    uw = np.array(
        [
            [np.cos(deltaL), 0, np.sin(deltaL), 0],
            [0, np.cos(deltaR), 0, np.sin(deltaR)],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ]
    ) @ np.array(
        [
            u - d / 2 * r,
            u + d / 2 * r,
            v + a * r,
            v + a * r,
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
            u - d / 2 * r,
            u + d / 2 * r,
            v + a * r,
            v - b * r,
        ]
    )
    alphaw = alpha(uw, vw)
    kappaw = kappa(R, q, uw)
    Fz = np.maximum(
        Fz_static
        + (mb * h * (du - v * r) / 2 / base) * np.array([-1, -1, 1, 1])
        + (mb * h * (dv + u * r) / d / base) * np.array([-b, b, -a, a]),
        np.zeros(wheelNum),
    )
    Fx = pacFx(Fz / 1000, kappaw * 100)
    Fy = pacFy(Fz / 1000, alphaw * rad2deg)
    Fx, Fy = pacFxyCombine(kappaw, alphaw, Fx, Fy)
    du = (
        np.array([np.cos(deltaL), np.cos(deltaR), 1, 1]) @ Fx
        + np.array([-np.sin(deltaL), -np.sin(deltaR), 0, 0]) @ Fy
    ) / M + v * r
    dv = (
        np.array([np.sin(deltaL), np.sin(deltaR), 0, 0]) @ Fx
        + np.array([np.cos(deltaL), np.cos(deltaR), 1, 1]) @ Fy
    ) / M - u * r
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

    # control
    T = [0, 0, 1000 * (u_desired - u), 1000 * (u_desired - u)]

    dq = (T - R * Fx) / Ispin
    return du, dv, dr, dq


# init
for step in range(4):
    du[step], dv[step], dr[step], dq[:, step] = advance(
        delta[step],
        0.5 * u_desired,
        u[step],
        v[step],
        r[step],
        q[:, step],
        du[step],
        dv[step],
    )
    u[step + 1] = u[step] + timeStep * du[step]
    v[step + 1] = v[step] + timeStep * dv[step]
    r[step + 1] = r[step] + timeStep * dr[step]
    q[:, step + 1] = q[:, step] + timeStep * dq[:, step]

    du[step + 1] = du[step]
    dv[step + 1] = dv[step]


for step in stepSeries[0:-3]:
    # PMECME
    # P
    u_p[step + 4] = u[step] + 4 / 3 * timeStep * (
        2 * du[step + 3] - du[step + 2] + 2 * du[step + 1]
    )
    v_p[step + 4] = v[step] + 4 / 3 * timeStep * (
        2 * dv[step + 3] - dv[step + 2] + 2 * dv[step + 1]
    )
    r_p[step + 4] = r[step] + 4 / 3 * timeStep * (
        2 * dr[step + 3] - dr[step + 2] + 2 * dr[step + 1]
    )
    q_p[:, step + 4] = q[:, step] + 4 / 3 * timeStep * (
        2 * dq[:, step + 3] - dq[:, step + 2] + 2 * dq[:, step + 1]
    )
    # M
    u_pm[step + 4] = u_p[step + 4] + 112 / 121 * (u_c[step + 3] - u_p[step + 3])
    v_pm[step + 4] = v_p[step + 4] + 112 / 121 * (v_c[step + 3] - v_p[step + 3])
    r_pm[step + 4] = r_p[step + 4] + 112 / 121 * (r_c[step + 3] - r_p[step + 3])
    q_pm[:, step + 4] = q_p[:, step + 4] + 112 / 121 * (
        q_c[:, step + 3] - q_p[:, step + 3]
    )
    # E
    temp = advance(
        delta[step + 4],
        0.5 * u_desired,
        u_pm[step + 4],
        v_pm[step + 4],
        r_pm[step + 4],
        q_pm[:, step + 4],
        du[step + 3],
        dv[step + 3],
    )
    du_pm[step + 4] = temp[0]
    dv_pm[step + 4] = temp[1]
    dr_pm[step + 4] = temp[2]
    dq_pm[:, step + 4] = temp[3]
    # C
    u_c[step + 4] = (9 * u[step + 3] - u[step + 1]) / 8 + 3 / 8 * timeStep * (
        du_pm[step + 4] + 2 * du[step + 3] - du[step + 2]
    )
    v_c[step + 4] = (9 * v[step + 3] - v[step + 1]) / 8 + 3 / 8 * timeStep * (
        dv_pm[step + 4] + 2 * dv[step + 3] - dv[step + 2]
    )
    r_c[step + 4] = (9 * r[step + 3] - r[step + 1]) / 8 + 3 / 8 * timeStep * (
        dr_pm[step + 4] + 2 * dr[step + 3] - dr[step + 2]
    )
    q_c[:, step + 4] = (9 * q[:, step + 3] - q[:, step + 1]) / 8 + 3 / 8 * timeStep * (
        dq_pm[:, step + 4] + 2 * dq[:, step + 3] - dq[:, step + 2]
    )
    # M
    u[step + 4] = u_c[step + 4] - 9 / 121 * (u_c[step + 4] - u_p[step + 4])
    v[step + 4] = v_c[step + 4] - 9 / 121 * (v_c[step + 4] - v_p[step + 4])
    r[step + 4] = r_c[step + 4] - 9 / 121 * (r_c[step + 4] - r_p[step + 4])
    q[:, step + 4] = q_c[:, step + 4] - 9 / 121 * (q_c[:, step + 4] - q_p[:, step + 4])
    # E
    temp = advance(
        delta[step + 4],
        0.5 * u_desired,
        u[step + 4],
        v[step + 4],
        r[step + 4],
        q[:, step + 4],
        du_pm[step + 4],
        dv_pm[step + 4],
    )
    du[step + 4] = temp[0]
    dv[step + 4] = temp[1]
    dr[step + 4] = temp[2]
    dq[:, step + 4] = temp[3]


plt.plot(timeSeries, u, label="u")
plt.plot(timeSeries, v, label="v")
plt.plot(timeSeries, r * 180 / np.pi, label="r")

plt.legend()
plt.grid()
plt.show()
