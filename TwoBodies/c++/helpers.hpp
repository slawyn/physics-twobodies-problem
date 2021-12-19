
#ifndef HELPERS_H
#define HELPERS_H

#include <iostream>
#include <fstream>
#include <string>
#include "vector.hpp"


typedef struct{
    double mass1;
    double mass2;
    double step;
    long points;
}config;


void readConfigIni(config *);
void writeData(const config &, const double*, const Vector3D *, const Vector3D *);


#endif // HELPERS_H
