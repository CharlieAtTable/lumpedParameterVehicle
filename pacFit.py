import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# 读取CSV文件
df_Fx = pd.read_csv("Fx.csv", index_col=0)
df_Fy = pd.read_csv("Fy.csv", index_col=0)

# 将数据转换为numpy数组
kappa = np.repeat(df_Fx.index.values, len(df_Fx.columns))
Fz = np.tile(df_Fx.columns.astype(float), len(df_Fx.index))
Fx = df_Fx.values.flatten()
alpha = np.repeat(df_Fy.index.values, len(df_Fy.columns))
Fz1 = np.tile(df_Fy.columns.astype(float), len(df_Fy.index))
Fy = df_Fy.values.flatten()


# 定义拟合的函数模型
def pacFx(Fz_kappa, B0, B1, B2, B3, B4, B5, B6, B7, B8, B9, B10):
    """
    Fz-kN

    kappa-%
    """
    Fz, kappa = Fz_kappa
    Fz /= 1000
    kappa *= 100
    C = B0
    D = B1 * Fz**2 + B2 * Fz
    BCD = (B3 * Fz**2 + B4 * Fz) * np.exp(-B5 * Fz)
    B = BCD / C / D
    E = B6 * Fz**2 + B7 * Fz + B8
    Sh = B9 * Fz + B10
    Sv = 0
    X1 = kappa + Sh
    return D * np.sin(C * np.arctan(B * X1 - E * (B * X1 - np.arctan(B * X1)))) + Sv


def pacFy(Fz_alpha, A0, A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13):
    """
    Fz-kN

    alpha-deg
    """
    Fz, alpha = Fz_alpha
    Fz /= 1000
    gamma = 0.0
    C = A0
    D = A1 * Fz**2 + A2 * Fz
    BCD = A3 * np.sin(2 * np.arctan(Fz / A4)) * (1 - A5 * abs(gamma))
    B = BCD / C / D
    E = A6 * Fz + A7
    Sh = A8 * gamma + A9 * Fz + A10
    Sv = A11 * Fz * gamma + A12 * Fz + A13
    X1 = alpha + Sh
    return D * np.sin(C * np.arctan(B * X1 - E * (B * X1 - np.arctan(B * X1)))) + Sv


# 拟合模型
popt_Fx, pcov_Fx = curve_fit(
    pacFx,
    (Fz, kappa),
    Fx,
    p0=[1, -10, 1500, 100, 276, 0.0886, 0.00402, -0.0615, 1.2, 0.0, 0],
    method="dogbox",
)

popt_Fy, pcov_Fy = curve_fit(
    pacFy,
    (Fz1, alpha),
    Fy,
    p0=[
        5,
        -5,
        1250,
        3036,
        12.8,
        0.0,
        -0.02,
        0.8,
        0.0,
        0.01,
        0.004,
        0.0,
        1.2,
        6.26,
    ],
    method="dogbox",
)

# 输出拟合结果
print("Fx: ", popt_Fx)
print("Fy: ", popt_Fy)


# def D(Fz, B1, B2):
#     return B1 * Fz**2 + B2 * Fz


# popt, pcov = curve_fit(
#     D, [7.3575, 14.715, 29.43, 44.145, 58.86], [6587, 12703.5, 23525, 32464.4, 39992.4]
# )
# print(popt)

# print("kappa=", kappa)
# print("Fz=", Fz)
# print("Fx=", Fx)
# print("alpha=", alpha)
# print("Fz1=", Fz1)
# print("Fy=", Fy)
