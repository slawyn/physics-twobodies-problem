TEMPLATE = app
CONFIG += console c++11
#CONFIG -= app_bundle
#CONFIG -= qt

QMAKE_LFLAGS_DEBUG = -static -static-libgcc -static-libstdc++

SOURCES += \
        helpers.cpp \
        main.cpp \
        vector.cpp

HEADERS += \
    helpers.hpp \
    vector.hpp
