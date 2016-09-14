#include <iostream>
using namespace std;
#include "Calibration.h"
#include <QApplication>

int main(int argc,char **argv)
{
    QApplication q(argc, argv);
    Calibration w(argc,argv);
    w.show();
    return q.exec();
}
