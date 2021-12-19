#ifndef VECTOR_HPP
#define VECTOR_HPP

class Vector3D{

public:
    double x;
    double y;
    double z;
    Vector3D(){};
    Vector3D(const double x,const double y,const double z);

    Vector3D operator +(Vector3D b);
    Vector3D operator -(Vector3D b);
    Vector3D operator *(double b) const;
    Vector3D operator /(double b) const;

    /*
    Vector3D operator =( const Vector3D &vec);
    */
};

Vector3D operator *(double b, const Vector3D &vec);

#endif // VECTOR_HPP
