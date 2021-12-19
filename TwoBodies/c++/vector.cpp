#include "vector.hpp"

Vector3D::Vector3D(const double x,const double y,const double z){
        this->x = x;
        this->y = y;
        this->z = z;
}
Vector3D Vector3D::operator+(Vector3D b){
       Vector3D result;
       result.x = this->x + b.x;
       result.y = this->y + b.y;
       result.z =this->z + b.z;

       return result;
}

Vector3D Vector3D::operator-(Vector3D b){
       Vector3D result;
       result.x = this->x - b.x;
       result.y = this->y - b.y;
       result.z =this->z - b.z;

       return result;
}

Vector3D Vector3D::operator*(double b) const{
        Vector3D result;
        result.x = this->x*b;
        result.y = this->y*b;
        result.z = this->z*b;
        return result;

} /**/

Vector3D Vector3D::operator /(double b) const{
        Vector3D result;
        result.x = this->x/b;
        result.y = this->y/b;
        result.z = this->z/b;
        return result;

}

/*
Vector3D Vector3D::operator =( const Vector3D &vec){
         this->x = vec.x;
         this->y = vec.y;
         this->z = vec.z;
         return *this;
 }*/


Vector3D operator *(double b, const Vector3D &vec){
    return vec*b;
}
