from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import sys
# https://en.wikipedia.org/wiki/Two-body_problem
# https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767
# https://evgenii.com/blog/two-body-problem-simulator/

x1 = []
x2 = []
y1 = []
y2 = []
z1 = []
z2 = []
t = []
R = []
r = []


# default parameters, overwritten by config.ini
PARAMETER_m1 = 1
PARAMETER_m2 = 0.5
PARAMETER_step = 0.1
PARAMETER_points = 1000

G = 6.67408e-11 # N-m2/kg2
m_nd = 1.989e+30 #kg #mass of the sun
r_nd = 5.326e+12 #m #distance between stars in Alpha Centauri
v_nd = 30000 #m/s #relative velocity of earth around the sun
t_nd = 79.91*365*24*3600*0.51 #s #orbital period of Alpha Centauri#Net constants

FACTOR_K1 = 1#G*t_nd*m_nd/(r_nd**2*v_nd)     uncomment for ex2/sci
FACTOR_K2 = 1#v_nd*t_nd/r_nd                 uncomment for ex2/sci
ZERO_R = True                              #R is zeroed


def init(pnumberofvalues):
    global x1,x2,y1,y2,z1,z2,t,R,r
    t = np.zeros(pnumberofvalues)
    x1 = np.zeros([pnumberofvalues,2])
    x2 = np.zeros([pnumberofvalues,2])
    y1 = np.zeros([pnumberofvalues,2])
    y2 = np.zeros([pnumberofvalues,2])
    z1 = np.zeros([pnumberofvalues,2])
    z2 = np.zeros([pnumberofvalues,2])

    R = np.zeros([pnumberofvalues,3])
    r = np.zeros([pnumberofvalues,3])

# Finds value of y for a given t using step size h
# and initial value y0 at x0.
'''
k1 = h * dydt(t0, y)
k2 = h * dydt(t0 + 0.5 * h, y + 0.5 * k1)
k3 = h * dydt(t0 + 0.5 * h, y + 0.5 * k2)
k4 = h * dydt(t0 + h, y + k3)

y[i] = y[i-1] + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4)
'''
def twobodiesExample2(pnumberofpoints, h, K1, K2, m1, m2):
    global x1, x2, y1, y2, z1, z2
    import scipy as sci#Import matplotlib and associated modules for 3D and animations
    # Count number of iterations using step size or
    # step height h
    #Define masses
    t=[]
    r=[]
    R=[]
    r1=[-0.5,0,0] #m
    r2=[0.5,0,0] #m#Convert pos vectors to arrays
    r1=np.array(r1,dtype="float64")
    r2=np.array(r2,dtype="float64")#Find Centre of Mass
    r_com=(m1*r1+m2*r2)/(m1+m2)#Define initial velocities
    v1=[0.01,0.01,0] #m/s
    v2=[-0.05,0,-0.1] #m/s#Convert velocity vectors to arrays
    v1=np.array(v1,dtype="float64")
    v2=np.array(v2,dtype="float64")#Find velocity of COM
    v_com=(m1*v1+m2*v2)/(m1+m2)

    #Package initial parameters
    init_params=np.array([r1,r2,v1,v2]) #create array of initial params
    init_params=init_params.flatten() #flatten array to make it 1D
    time_span=np.linspace(0,1,pnumberofpoints) #8 orbital periods and 500 points#Run the ODE solver
    import scipy.integrate

    #A function defining the equations of motion
    def TwoBodyEquations(w,t,G,m1,m2,K1,K2):
        r1=w[:3]
        r2=w[3:6]
        v1=w[6:9]
        v2=w[9:12]
        r=sci.linalg.norm(r2-r1) #Calculate magnitude or norm of vector
        dv1bydt=K1*m2*(r2-r1)/r**3
        dv2bydt=K1*m1*(r1-r2)/r**3
        dr1bydt=K2*v1
        dr2bydt=K2*v2
        r_derivs=np.concatenate((dr1bydt,dr2bydt))
        derivs=np.concatenate((r_derivs,dv1bydt,dv2bydt))
        return derivs


    two_body_sol=sci.integrate.odeint(TwoBodyEquations,init_params,time_span,args=(G,m1,m2,K1,K2))

    r1_sol=two_body_sol[:,:3]
    r2_sol=two_body_sol[:,3:6]

    init(pnumberofpoints)

    for idx in range(len(x1)):
        x1[idx][0] = r1_sol[idx,0]
        y1[idx][0] = r1_sol[idx,1]
        z1[idx][0] = r1_sol[idx,2]
        x2[idx][0] = r2_sol[idx,0]
        y2[idx][0] = r2_sol[idx,1]
        z2[idx][0] = r2_sol[idx,2]
    '''
    fig=plt.figure(figsize=(15,15))#Create 3D axes
    ax=fig.add_subplot(111,projection="3d")#Plot the orbits
    ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="darkblue")
    ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="tab:red")#Plot the final positions of the stars
    ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=100,label="Alpha Centauri A")
    ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="tab:red",marker="o",s=100,label="Alpha Centauri B")#Add a few more bells and whistles
    ax.set_xlabel("x-coordinate",fontsize=14)
    ax.set_ylabel("y-coordinate",fontsize=14)
    ax.set_zlabel("z-coordinate",fontsize=14)
    ax.set_title("Visualization of orbits of stars in a two-body system\n",fontsize=14)
    ax.legend(loc="upper left",fontsize=14)
    plt.show()
    '''

def twobodiesExample1(pnumberofpoints, h, K1, K2, m1, m2):
    global r,R,x1,x2,y1,y2,z1,z2,t
    # Count number of iterations using step size or
    # step height h
    t0 = 0

    init(pnumberofpoints)

    x1[0][0] = 1
    y1[0][0] = 0
    z1[0][0] = 0
    x2[0][0] = 0
    y2[0][0] = 0
    z2[0][0] = 0

    x1[0][1] = 0
    y1[0][1] = 0
    z1[0][1] = 0
    x2[0][1] = 0
    y2[0][1] = 1.59687
    z2[0][1] = 0


    r[0][0] = (x2[0][0]-x1[0][0])
    r[0][1] = (y2[0][0]-y1[0][0])
    r[0][2] = (z2[0][0]-z1[0][0])

    R[0][0]=(m1*x1[0][0]+m2*x2[0][0])/(m1+m2)   # TODO
    R[0][1]=(m1*y1[0][0]+m2*y2[0][0])/(m1+m2)   # TODO
    R[0][2]=(m1*z1[0][0]+m2*z2[0][0])/(m1+m2)   # TODO


    a = [h/2.0, h/2.0, h, 0]
    b = [h/6.0, h/3.0, h/3.0, h/6.0]


    def deriveVelocityAndAcceleration(px1,py1,pz1,px2,py2,pz2,m1,m2):
        global K1, K2
        rnormed = math.sqrt((px1[0]-px2[0])**2 + (py1[0]-py2[0])**2 + (pz1[0]-pz2[0])**2)
        dx1 = [0,0]
        dy1 = [0,0]
        dz1 = [0,0]

        dx2 = [0,0]
        dy2 = [0,0]
        dz2 = [0,0]

        dx1[0] = px1[1]
        dy1[0] = py1[1]
        dz1[0] = pz1[1]

        dx2[0] = px2[1]
        dy2[0] = py2[1]
        dz2[0] = pz2[1]

        dx1[1] = (m2)  * (px2[0]-px1[0])/ (rnormed**3)
        dy1[1] = (m2) * (py2[0]-py1[0])/ (rnormed**3)
        dz1[1] = (m1) * (pz2[0]-pz1[0]) / (rnormed**3)


        dx2[1] = (m1) * (px1[0]-px2[0]) / (rnormed**3)
        dy2[1] = (m1) * (py1[0]-py2[0]) / (rnormed**3)
        dz2[1] = (m2) * (pz1[0]-pz2[0]) / (rnormed**3)
        return dx1,dy1,dz1,dx2,dy2,dz2

    for i in range(1, len(t)):
        xn1 = np.zeros(2)     # Position and velocity
        xn2 = np.zeros(2)
        yn1 = np.zeros(2)
        yn2 = np.zeros(2)
        zn1 = np.zeros(2)
        zn2 = np.zeros(2)

        # Runge-Kutta
        xn1[0] = x1[i-1][0]
        xn1[1] = x1[i-1][1]

        xn2[0] = x2[i-1][0]
        xn2[1] = x2[i-1][1]

        yn1[0] = y1[i-1][0]
        yn1[1] = y1[i-1][1]

        yn2[0] = y2[i-1][0]
        yn2[1] = y2[i-1][1]

        zn1[0] = z1[i-1][0]
        zn1[1] = z1[i-1][1]

        zn2[0] = z2[i-1][0]
        zn2[1] = z2[i-1][1]

        #vars
        xvar1 = [0,0]
        yvar1 = [0,0]
        zvar1 = [0,0]

        xvar2 = [0,0]
        yvar2 = [0,0]
        zvar2 = [0,0]

        xvar1[0] = xn1[0]
        yvar1[0] = yn1[0]
        zvar1[0] = zn1[0]

        xvar1[1] = xn1[1]
        yvar1[1] = yn1[1]
        zvar1[1] = zn1[1]

        xvar2[0] = xn2[0]
        yvar2[0] = yn2[0]
        zvar2[0] = zn2[0]

        xvar2[1] = xn2[1]
        yvar2[1] = yn2[1]
        zvar2[1] = zn2[1]

        #k
        kx1 = [0,0]
        kx2 = [0,0]
        ky1 = [0,0]
        ky2 = [0,0]
        kz1 = [0,0]
        kz2 = [0,0]



        for idx in range(4):

            dxvar1,dyvar1,dzvar1, dxvar2,dyvar2,dzvar2 = deriveVelocityAndAcceleration(xvar1,yvar1,zvar1,xvar2,yvar2,zvar2,m1, m2)

            dxvar1[0] = K2*dxvar1[0]
            dyvar1[0] = K2*dyvar1[0]
            dzvar1[0] = K2*dzvar1[0]

            dxvar1[1] = K1*dxvar1[1]
            dyvar1[1] = K1*dyvar1[1]
            dzvar1[1] = K1*dzvar1[1]

            kx1[0] = kx1[0] + dxvar1[0]*b[idx]
            ky1[0] = ky1[0] + dyvar1[0]*b[idx]
            kz1[0] = kz1[0] + dzvar1[0]*b[idx]
            kx1[1] = kx1[1] + dxvar1[1]*b[idx]
            ky1[1] = ky1[1] + dyvar1[1]*b[idx]
            kz1[1] = kz1[1] + dzvar1[1]*b[idx]


            kx2[0] = kx2[0] + dxvar2[0]*b[idx]
            ky2[0] = ky2[0] + dyvar2[0]*b[idx]
            kz2[0] = kz2[0] + dzvar2[0]*b[idx]
            kx2[1] = kx2[1] + dxvar2[1]*b[idx]
            ky2[1] = ky2[1] + dyvar2[1]*b[idx]
            kz2[1] = kz2[1] + dzvar2[1]*b[idx]


            # next step
            xvar1[0] = xn1[0]+(dxvar1[0]* a[idx])
            xvar1[1] = xn1[1]+(dxvar1[1] * a[idx])

            yvar1[0] = yn1[0]+(dyvar1[0]* a[idx])
            yvar1[1] = yn1[1]+(dyvar1[1] * a[idx])

            zvar1[0] = zn1[0]+(dzvar1[0]  * a[idx])
            zvar1[1] = zn1[1]+(dzvar1[1] * a[idx])

            xvar2[0] = xn2[0]+(dxvar2[0]* a[idx])
            xvar2[1] = xn2[1]+(dxvar2[1] * a[idx])

            yvar2[0] = yn2[0]+(dyvar2[0]* a[idx])
            yvar2[1] = yn2[1]+(dyvar2[1] * a[idx])

            zvar2[0] = zn2[0]+(dzvar2[0]  * a[idx])
            zvar2[1] = zn2[1]+(dzvar2[1] * a[idx])
            #print(idx)
            #print("-------->  ku::("+str(kx1))
            #print("-------->  ku::("+str(ky1))
            #print("-------->  ku::("+str(kz1))
            #print("-------->  ku::("+str(kx2))
            #print("-------->  ku::("+str(ky2))
            #print("-------->  ku::("+str(kz2))
            #print(str(idx)+")x ---> "+str(xvar1[0] )+" "+str(dxvar1[0]) )


        x1[i][0]  = xn1[0]  + kx1[0]
        y1[i][0]  = yn1[0]  + ky1[0]
        z1[i][0]  = zn1[0]  + kz1[0]

        x2[i][0]  = xn2[0]  + kx2[0]
        y2[i][0]  = yn2[0]  + ky2[0]
        z2[i][0]  = zn2[0]  + kz2[0]

        x1[i][1]  = xn1[1]  + kx1[1]
        y1[i][1]  = yn1[1]  + ky1[1]
        z1[i][1]  = zn1[1]  + kz1[1]

        x2[i][1]  = xn2[1]  + kx2[1]
        y2[i][1]  = yn2[1]  + ky2[1]
        z2[i][1]  = zn2[1]  + kz2[1]

        print(str(i)+" -----------------------------")
        print("u::"+str(x1[i]))
        print("u::"+str(y1[i]))
        print("u::"+str(z1[i]))
        print("u::"+str(x2[i]))
        print("u::"+str(y2[i]))
        print("u::"+str(z2[i]))
        print("----------------------------------")

        R[i][0]=(m1*x1[i][0]+m2*x2[i][0])/(m1+m2)   # TODO
        R[i][1]=(m1*y1[i][0]+m2*y2[i][0])/(m1+m2)   # TODO
        R[i][2]=(m1*z1[i][0]+m2*z2[i][0])/(m1+m2)   # TODO

        t0 = t0 + h
        t[i] = t0

        r[i][0] = (x2[i][0]-x1[i][0])
        r[i][1] = (y2[i][0]-y1[i][0])
        r[i][2] = (z2[i][0]-z1[i][0]) #




def twobodiesExample0(pnumberofpoints, h, K1, K2, m1, m2):
    global r, t, R
    # Count number of iterations using step size or
    # step height h

    init(pnumberofpoints)
    def deriveVelocityAndAcceleration(x1,y1,z1,m1, m2):
        rr = math.sqrt(x1[0]**2 + y1[0]**2)# + z1[0]**2)
        dx1 = [0,0]
        dy1 = [0,0]
        dz1 = [0,0]

        dx1[0] = x1[1]
        dy1[0] = y1[1]
        dz1[0] = z1[1]

        dx1[1] = (-(1+m2)* x1[0]) / (rr**3)
        dy1[1] = (-(1+m2) * y1[0]) / (rr**3)
        dz1[1] = (-(1+m2) * z1[0]) / (rr**3)
        return dx1,dy1,dz1

    t0 = 0
    rx1 = [1,0]
    ry1 = [0, 1.59687]
    rz1 = [0,0]

    r[0][0] = rx1[0]
    r[0][1] = ry1[0]
    r[0][2] = rz1[0]


    a = [h/2.0, h/2.0, h, 0.0]
    b = [h/6.0, h/3.0, h/3.0, h/6.0]
    for i in range(1,len(t)):
        # Runge-Kutta
        xn1 = [0,0]
        yn1 = [0,0]
        zn1 = [0,0]

        xn1[0] = rx1[0]
        xn1[1] = rx1[1]

        yn1[0] = ry1[0]
        yn1[1] = ry1[1]

        zn1[0] = rz1[0]
        zn1[1] = rz1[1]

        #print("####"+str(i)+" x:"+str(xn1)+" y:"+str(yn1))
        kx1 = [0,0]
        ky1 = [0,0]
        kz1 = [0,0]

        xvar1 = xn1[:]
        yvar1 = yn1[:]
        zvar1 = zn1[:]

        for idx in range(4):

            #print(str(xn1)+" "+str(yn1))
            dx1,dy1,dz1 = deriveVelocityAndAcceleration(xvar1,yvar1,zvar1,m1, m2)

            kx1[0] = kx1[0] + (K2*dx1[0]*b[idx])
            ky1[0] = ky1[0] + (K2*dy1[0]*b[idx])
            kz1[0] = kz1[0] + (K2*dz1[0]*b[idx])

            kx1[1] = kx1[1] + (K1*dx1[1]*b[idx])
            ky1[1] = ky1[1] + (K1*dy1[1]*b[idx])
            kz1[1] = kz1[1] + (K1*dz1[1]*b[idx])

            # next step
            xvar1[0] = xn1[0]+(K2*dx1[0]  * a[idx])
            yvar1[0] = yn1[0]+(K2*dy1[0]  * a[idx])
            zvar1[0] = zn1[0]+(K2*dz1[0]  * a[idx])

            xvar1[1] = xn1[1]+(K1*dx1[1]  * a[idx])
            yvar1[1] = yn1[1]+(K1*dy1[1]  * a[idx])
            zvar1[1] = zn1[1]+(K1*dz1[1]  * a[idx])

            #print("-------->("+str(idx)+")")
            #print("-------->  u ::("+str(xn1))
            #print("-------->  u ::("+str(xvar1[0])+"\t"+ str(yvar1[0])+"\t"+str(xvar1[1])+"\t"+str(yvar1[1]))
            #print("-------->  du::("+str(dx1[0])+"\t"+ str(dy1[0])+"\t"+str(dx1[1])+"\t"+str(dy1[1]))
            #print("-------->  ut::("+str(kx1[0])+"\t"+ str(ky1[0])+"\t"+str(kx1[1])+"\t"+str(ky1[1]))

        #print("->kx:"+str(kx1))
        #print("->ky:"+str(kx1))
        rx1[0]  = xn1[0]  + kx1[0]
        ry1[0]  = yn1[0]  + ky1[0]
        rz1[0]  = zn1[0]  + kz1[0]

        rx1[1]  = xn1[1]  + kx1[1]
        ry1[1]  = yn1[1]  + ky1[1]
        rz1[1]  = zn1[1]  + kz1[1]

        r[i][0] = rx1[0]
        r[i][1] = ry1[0]
        r[i][2] = rz1[0]

        t0 = t0 + h
        t[i] = t0


def writeData(path, typeofdata):
    global ZERO_R, PARAMETER_m1,PARAMETER_m2, t, r, R
    print("## %d values generated"%(len(t)))
    f = open(path+"data.out","w+")

    tstring = "t ="

    f.write("m1 = %s\n"%(PARAMETER_m1))
    f.write("m2 = %s\n"%(PARAMETER_m2))

    if typeofdata == 0:
        x1string = "x1 ="
        y1string = "y1 ="
        z1string = "z1 ="

        x2string = "x2 ="
        y2string = "y2 ="
        z2string = "z2 ="

        for i in range(len(t)):
            tstring =  tstring+" "+str(t[i])
            x1string = x1string+" "+str(x1[i][0])
            y1string = y1string+" "+str(y1[i][0])
            z1string = z1string+" "+str(z1[i][0])
            x2string = x2string+" "+str(x2[i][0])
            y2string = y2string+" "+str(y2[i][0])
            z2string = z2string+" "+str(z2[i][0])
        f.write(tstring+"\n")
        f.write (x1string+"\n")
        f.write (y1string+"\n")
        f.write (z1string+"\n")
        f.write (x2string+"\n")
        f.write (y2string+"\n")
        f.write (z2string+"\n")
    elif typeofdata == 1:
        rxstring = "rx ="
        rystring = "ry ="
        rzstring = "rz ="
        Rxstring = "Rx ="
        Rystring = "Ry ="
        Rzstring = "Rz ="
        for i in range(len(t)):
            tstring =  tstring+" "+str(t[i])
            rxstring = rxstring+" "+str(r[i][0])
            rystring = rystring+" "+str(r[i][1])
            rzstring = rzstring+" "+str(r[i][2])

            if ZERO_R:
                Rxstring = Rxstring+" 0"
                Rystring = Rystring+" 0"
                Rzstring = Rzstring+" 0"
            else:
                Rxstring = Rxstring+" "+str(R[i][0])
                Rystring = Rystring+" "+str(R[i][1])
                Rzstring = Rzstring+" "+str(R[i][2])

        f.write (tstring+"\n")
        f.write (rxstring+"\n")
        f.write (rystring+"\n")
        f.write (rzstring+"\n")

        f.write (Rxstring+"\n")
        f.write (Rystring+"\n")
        f.write (Rzstring+"\n")

    f.close()


def readInputData(path):
    global PARAMETER_m1, PARAMETER_m2, PARAMETER_step, PARAMETER_points


    try:
        f = open(path+"config.ini","r")
        data = f.readlines()
        f.close()
        for line in data:
            if line[0] == "#":
                continue

            idx = line.index(":")
            param = line[0:idx]

            if idx != -1:
                if param == "m1":
                    PARAMETER_m1 = float(line[idx+1:].strip())
                elif param == "m2":
                    PARAMETER_m2 = float(line[idx+1:].strip())
                elif param == "step":
                    PARAMETER_step = float(line[idx+1:].strip())
                elif param == "points":
                    PARAMETER_points = int(line[idx+1:].strip())
    except:
        print("config.ini couldn't be interpreted, using default parameters")

if __name__ == "__main__":
    typer = 1
    typex = 0
    path = ""

    if len(sys.argv) > 1:

        if len(sys.argv) == 3:
            path = sys.argv[2]+"/"

        readInputData(path)
        if sys.argv[1] == "-ex0":
            twobodiesExample0(PARAMETER_points, PARAMETER_step, FACTOR_K1, FACTOR_K2, PARAMETER_m1, PARAMETER_m2)
            writeData(path,typer)
        elif sys.argv[1] == "-ex1":
            twobodiesExample1(PARAMETER_points, PARAMETER_step, FACTOR_K1, FACTOR_K2, PARAMETER_m1, PARAMETER_m2)
            writeData(path,typer)
        elif sys.argv[1] == "-ex2":
            twobodiesExample2(PARAMETER_points, PARAMETER_step, FACTOR_K1, FACTOR_K2, PARAMETER_m1, PARAMETER_m2)
            writeData(path, typex)
    else:
        print("# no action")
