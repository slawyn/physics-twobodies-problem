#include "helpers.hpp"

std::string path="";

// for switch-case
const std::string validparameters[] = {"m1","m2","step","points"};
enum {m1=0,m2=1,step=2,points=3};

std::string trim(const std::string& str)
{
    size_t first = str.find_first_not_of(' ');
    if (std::string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}


void readConfigIni(config * configuration){
    std::ifstream configIniFile;
    std::string line;

    configIniFile.open(path+"config.ini");

    // read all lines
    while (std::getline(configIniFile, line)){

        // skip comment lines
        if(line.length()>0 && line[0] !='#'){
            size_t idx = line.find(":");

            // parse parameter
            if(idx != std::string::npos){

                // strip away spaces
                std::string parameter = trim(line.substr(0,idx));
                std::string value = trim(line.substr(idx+1,line.length()));

                unsigned int switcher = 0xFF;

                // find parameter
                for(unsigned int i =0; i<parameter.length();i++){
                    if(parameter.compare(validparameters[i])==0)
                        switcher=i;
                }
                switch(switcher){
                case m1:
                    configuration->mass1 = std::stod(value);
                    break;
                case m2:
                    configuration->mass2 = std::stod(value);
                    break;
                case step:
                    configuration->step = std::stod(value);
                    break;
                case points:
                    configuration->points = std::stol(value);
                    break;

                }
            }
        }
    }

    configIniFile.close();
}

void  writeData(const config &globalconfig, const double *tdata,const Vector3D * rdata, const Vector3D * Rdata){
    std::ofstream dataFile;
    dataFile.open(path+"data.out");

    dataFile<<"m1 = "<<std::to_string(globalconfig.mass1)<<"\n";
    dataFile<<"m2 = "<<std::to_string(globalconfig.mass2)<<"\n";
    dataFile<<"t =";

    for(long idx = 0;idx<globalconfig.points;idx++){
        dataFile<<" "<<std::to_string(tdata[idx]);
    }

    dataFile<<"\nrx =";
    for(long idx = 0;idx<globalconfig.points;idx++){
        dataFile<<" "<<std::to_string(rdata[idx].x);
    }

    dataFile<<"\nry =";
    for(long idx = 0;idx<globalconfig.points;idx++){
           dataFile<<" "<<std::to_string(rdata[idx].y);
    }

    dataFile<<"\nrz =";
    for(long idx = 0;idx<globalconfig.points;idx++){
           dataFile<<" "<<std::to_string(rdata[idx].z);
    }

    dataFile<<"\nRx =";
    for(long idx = 0;idx<globalconfig.points;idx++){
           dataFile<<" "<<std::to_string(Rdata[idx].x);
    }

    dataFile<<"\nRy =";
    for(long idx = 0;idx<globalconfig.points;idx++){
           dataFile<<" "<<std::to_string(Rdata[idx].y);
    }

    dataFile<<"\nRz =";
    for(long idx = 0;idx<globalconfig.points;idx++){
           dataFile<<" "<<std::to_string(Rdata[idx].z);
    }

    dataFile.close();
}
