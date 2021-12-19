#include "helpers.hpp"
#include <cmath>



extern std::string path;

// derive velocity and acceleration from the input
void derivative(Vector3D * positions, Vector3D * velocities, Vector3D *dpos){

    double r = sqrt(pow(positions[0].x-positions[1].x,2)+pow(positions[0].y-positions[1].y,2)+pow(positions[0].z-positions[1].z,2));
    dpos[0] = velocities[0];
    dpos[1] = (positions[1]-positions[0])/pow(r,3);
    dpos[2] = velocities[1];
    dpos[3] = (positions[0]-positions[1])/pow(r,3);

}

// integrate acceleration and velocity using for each one
// k1 = h * dydt(t0, y)
// k2 = h * dydt(t0 + 0.5 * h, y + 0.5 * k1)
// k3 = h * dydt(t0 + 0.5 * h, y + 0.5 * k2)
// k4 = h * dydt(t0 + h, y + k3)
// y[i] = y0 + (1.0 / 6.0)*(k1 + 2 * k2 + 2 * k3 + k4) whereas y0 = y[i-1]

void rungekutta(const config & simconfig, Vector3D *positions, Vector3D *velocities){
    const double a[]= {simconfig.step/2.0, simconfig.step/2.0, simconfig.step, 0};
    const double b[] = {simconfig.step/6.0, simconfig.step/3.0, simconfig.step/3.0, simconfig.step/6.0};

    // y0
    Vector3D pos1_0(positions[0]);
    Vector3D pos2_0(positions[1]);
    Vector3D vel1_0(velocities[0]);
    Vector3D vel2_0(velocities[1]);

    // k
    Vector3D k_body1[2]={Vector3D(0,0,0),Vector3D(0,0,0)};
    Vector3D k_body2[2]={Vector3D(0,0,0),Vector3D(0,0,0)};

    //dx
    Vector3D dpos[4];

    // k1 k2 k3 k4
    for(unsigned int idx = 0;idx <4;idx++){
        derivative(positions, velocities, dpos);

        dpos[1]=simconfig.mass2 * dpos[1];  //now multiply with masses
        dpos[3]=simconfig.mass1 * dpos[3];

        for(unsigned int dlevels=0;dlevels<2;dlevels++){
            k_body1[dlevels] =k_body1[dlevels]+ b[idx]*dpos[0+dlevels];  // k factors pos1 and vel1
            k_body2[dlevels] =k_body2[dlevels]+ b[idx]*dpos[2+dlevels];  // k factors pos2 and vel2
        }

        // temporary t0,
        positions[0] = pos1_0 +(a[idx]*dpos[0]);
        positions[1] = pos2_0 +(a[idx]*dpos[2]);
        velocities[0] = vel1_0 +(a[idx]*dpos[1]);
        velocities[1] = vel2_0 +(a[idx]*dpos[3]);
    }

    // update velocity and positions
    positions[0] = pos1_0  +  k_body1[0];
    positions[1] = pos2_0  +  k_body2[0];
    velocities[0] = vel1_0 + k_body1[1];
    velocities[1] = vel2_0 + k_body2[1];

    //std::cout<< positions[0].x <<" "<<positions[0].y <<" "<<positions[0].z <<" "<<std::endl;
    //std::cout<< positions[1].x <<" "<<positions[1].y <<" "<<positions[1].z <<" "<<std::endl;
    //std::cout<<"-----------------------------"<<std::endl;
}


void generateTwoBodyData(const config & simconfig, double * tdata, Vector3D * rdata, Vector3D * Rdata){
    Vector3D positions[2];
    Vector3D velocities[2];

    // init [0] - body1 [1]- body2
    positions[0].x = 1;
    positions[0].y = 0;
    positions[0].z = 0;

    positions[1].x = 0;
    positions[1].y = 0;
    positions[1].z = 0;

    velocities[0].x = 0;
    velocities[0].y = 0;
    velocities[0].z = 0;

    velocities[1].x = 0;
    velocities[1].y = 1.59687;
    velocities[1].z = 0;

    rdata[0] =  positions[1] - positions[0];
    bool zero_R = true;
    if(!zero_R)
        Rdata[0] = ((simconfig.mass1*positions[0])+(simconfig.mass2*positions[1]))/(simconfig.mass1+simconfig.mass2);


    // generate
    for(long idx=1; idx<simconfig.points;idx++ ){
        rungekutta(simconfig, positions, velocities);

        rdata[idx] = positions[1] - positions[0];
        tdata[idx] = tdata[idx-1]+ simconfig.step;

        if(!zero_R)
            Rdata[idx] = (simconfig.mass1*positions[0]+simconfig.mass2*positions[1])/(simconfig.mass1+simconfig.mass2);
    }
}


int main(int argc, char**argv)
{
    if(argc>1){
        path = std::string(argv[1])+"/";
    }

    std::cout<<path;
    static config globalconfig = {1,0.5, 0.05,1000};        // default parameters in case config.ini doesn't exist
    static double *tdata;
    static Vector3D *Rdata, *rdata;

    //read config.ini and set parameters
    readConfigIni(&globalconfig);

    // allocate space for values
    tdata =new double[globalconfig.points];
    rdata =new Vector3D[globalconfig.points];
    Rdata =new Vector3D[globalconfig.points];


    // generate data
    generateTwoBodyData(globalconfig, tdata, rdata,Rdata);

    // write data
    writeData(globalconfig, tdata, rdata, Rdata);

    return 0;
}
