from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import sys
# https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767
# https://evgenii.com/blog/two-body-problem-simulator/
animate = False
trail = False
previous = 0


def plotData(x1data,x2data,y1data,y2data, z1data, z2data):
    global animate
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    if animate:
        firstBodyTrail, = ax.plot(x1data, y1data, z1data,'blue',label="body1(t)")
        secondBodyTrail, = ax.plot(x2data, y2data, z2data, '#f5a60a',label="body2(t)")

        firstBody, = ax.plot(x1data, y1data, z1data,'blue', marker="o")
        secondBody, = ax.plot(x2data, y2data, z2data, '#f5a60a',marker="o")
        ax.legend()

        def updateAnimation(num):
            global previous, trail

            if num<len(x1data):
                firstBodyTrail.set_data(x1data[previous:num], y1data[previous:num])
                firstBodyTrail.set_3d_properties(z1data[previous:num])

                firstBody.set_data(x1data[num], y1data[num])
                firstBody.set_3d_properties(z1data[num])


                secondBodyTrail.set_data(x2data[previous:num], y2data[previous:num])
                secondBodyTrail.set_3d_properties(z2data[previous:num])

                secondBody.set_data(x2data[num], y2data[num])
                secondBody.set_3d_properties(z2data[num])

            # Trail

            if trail:
                if (num - previous)<260 and num > 250:
                    previous = previous + 1
            #secondBody.set_color('#9944'+"%02x"%((0x55+num)%0xFF))
            return firstBodyTrail, secondBodyTrail,

        anim = animation.FuncAnimation(fig,updateAnimation, interval=1,blit=False)
    else:
        ax.scatter(x1data, y1data, z1data, label="x1(t)")
        ax.scatter(x2data, y2data, z2data, label="x2(t)")
        ax.legend()
    plt.show()

def calculateTrajectories(t, m1, m2, r, R):


    # Data for a three-dimensional line
    x1data = np.zeros((len(t)))
    y1data = np.zeros((len(t)))
    z1data = np.zeros((len(t)))

    x2data = np.zeros((len(t)))
    y2data = np.zeros((len(t)))
    z2data = np.zeros((len(t)))
    m1 = float(m1)
    m2 = float(m2)
    M = m1 + m2

    for i in range(len(t)):
        #print(r[i][0])
        x1data[i] = float(R[i][0])  + m2/M * float(r[i][0])
        y1data[i] = float(R[i][1])  + m2/M * float(r[i][1])
        z1data[i] = float(R[i][2])  + m2/M * float(r[i][2])

        x2data[i] = float(R[i][0])  - m1/M * float(r[i][0])
        y2data[i] = float(R[i][1])  - m1/M * float(r[i][1])
        z2data[i] = float(R[i][2])  - m1/M * float(r[i][2])

        #print("%-4d %-10s %-10s %-10s %-10s %-10s %-10s"%(i, x1data[i], x2data[i], y1data[i], y2data[i], z1data[i], z2data[i]))

    plotData(x1data,x2data,y1data,y2data,z1data,z2data)

if __name__ == "__main__":

    print(sys.argv)
    if len(sys.argv) == 2:
        if sys.argv[1] == "-animate":
            animate = True
        elif sys.argv[1] == "-animatetrail":
            animate = True
            trail = True

    f = open("data.out","r")
    data = f.readlines()
    f.close()

    if data[0][0:2] == "m1" and data[1][0:2] == "m2" and data[2][0:1] == "t" and data[3][0:2] == "rx" and data[4][0:2] == "ry" and data[5][0:2] == "rz"  and data[6][0:2] == "Rx" and data[7][0:2] == "Ry" and data[8][0:2] == "Rz":
        m1 = data[0].split(" ")[2]
        m2 = data[1].split(" ")[2]
        t  = data[2].split(" ")[2:]
        rx = data[3].split(" ")[2:]
        ry = data[4].split(" ")[2:]
        rz = data[5].split(" ")[2:]

        Rx = data[6].split(" ")[2:]
        Ry = data[7].split(" ")[2:]
        Rz = data[8].split(" ")[2:]


        r = [list(a) for a in zip(rx,ry,rz)]
        R = [list(a) for a in zip(Rx,Ry,Rz)]
        calculateTrajectories(t, m1, m2, r, R)

    elif data[0][0:2] == "m1" and data[1][0:2] == "m2" and data[2][0:1] == "t" and data[3][0:2] == "x1" and data[4][0:2] == "y1" and data[5][0:2] == "z1" and  data[6][0:2] == "x2" and data[7][0:2] == "y2" and data[8][0:2] == "z2":
        m1 = data[0].split(" ")[2]
        m2 = data[1].split(" ")[2]
        t  = data[2].split(" ")[2:]
        x1 = data[3].split(" ")[2:]
        y1 = data[4].split(" ")[2:]
        z1 = data[5].split(" ")[2:]

        x2 = data[6].split(" ")[2:]
        y2 = data[7].split(" ")[2:]
        z2 = data[8].split(" ")[2:]

        x1data = np.zeros((len(t)))
        y1data = np.zeros((len(t)))
        z1data = np.zeros((len(t)))

        x2data = np.zeros((len(t)))
        y2data = np.zeros((len(t)))
        z2data = np.zeros((len(t)))

        for idx in range(len(t)):
            x1data[idx] = float(x1[idx])
            y1data[idx] = float(y1[idx])
            z1data[idx] = float(z1[idx])

            x2data[idx] = float(x2[idx])
            y2data[idx] = float(y2[idx])
            z2data[idx] = float(z2[idx])
        plotData(x1data,x2data,y1data,y2data,z1data,z2data)
