#ifndef TEST_DATA_H
#define TEST_DATA_H

#include <Python.h>
#include "global.h"

class TestData
{
public:
    TestData();
    int test_model(std::vector<cv::Mat>& mat_list);

private:
    int init_numpy();
    PyObject* MatToObject(cv::Mat image);
};

#endif // TEST_DATA_H
