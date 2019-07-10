#ifndef LANENET_CLUSTER_H
#define LANENET_CLUSTER_H


#include <Python.h>
#include <iostream>
#include <opencv2/opencv.hpp>

class LaneNetCluster
{
public:
    LaneNetCluster();

    int GetLaneMask(long int binary_seg_ret[1][256][512], float instance_seg_ret[1][256][512][4]);

private:
    int init_numpy();

    void ConvertToArray(PyObject *object);
    void GetLanePts(PyObject *object);
};

#endif // LANENET_CLUSTER_H
