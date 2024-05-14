import numpy as np


class Tire:
    lonCoef = 0.0
    cornStiff = 0.0


class Axle:
    """axle, contains springs, dampers, tires, position"""

    mw = 0.0
    """mass of half axle"""
    springK = 0.0
    damperC = 0.0
    leftTire = Tire()
    rightTire = Tire()
    position = [0.0, 0.0, 0.0]
    """axle centre location in sprung frame"""
    base = 0.0
    Ispin = 0.0
    """half axle"""
    radius = 510.0
    """tire radius"""


class Vehicle:
    mb = 4455.0
    """sprung mass"""
    Ixx = 2286.8
    Iyy = 35408.7
    Izz = 34823.2
    Ixy = 0.0
    Ixz = 1626.0
    Iyz = 0.0

    axlesNum = 2
    axles = [Axle() for i in range(axlesNum)]
    axles[0].position = [1250.0, 0.0, -575.0]
    axles[0].base = 2030.0
    axles[0].mw = 285.0
    axles[0].Ispin = 10.0

    axles[1].position = [-3750.0, 0.0, -555.0]
    axles[1].base = 1863.0
    axles[1].mw = 367.5
    axles[1].Ispin = 20.0


def main():
    gravity = 9.8
    vehicle = Vehicle()
    endTime = 10.0
    timeStep = 0.01
    timeSeries = np.arange(0, timeStep + endTime, timeStep, dtype=float)
    stepSeries = np.arange(1, timeSeries.size, dtype=int)
    u = np.zeros(timeSeries.shape)
    v = np.zeros(timeSeries.shape)
    u[0] = 0.0
    for step in stepSeries:
        u[step] = timeSeries[step]

    print(u)
    print(timeSeries.shape)
    print(stepSeries.shape)
    return 0


if __name__ == "__main__":
    main()
