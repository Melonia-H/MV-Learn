#include "test_data.h"
#include <numpy/arrayobject.h>


TestData::TestData()
{

}

PyObject* TestData::MatToObject(cv::Mat image)
{
    int nElem = image.rows * image.cols;
    uchar* m = new uchar[nElem];
    std::memcpy(m, image.data, nElem * sizeof(uchar));
    npy_intp mdims[] = {image.cols, image.rows};
    PyObject* mat = PyArray_SimpleNewFromData(2, mdims, NPY_UINT8, (void*) m);
    return mat;
}

int TestData::init_numpy()
{
    import_array();
}

int TestData::test_model(std::vector<cv::Mat>& mat_list)
{
    Py_Initialize();// 导入模块
    init_numpy();

    if (!Py_IsInitialized())
        return -1;

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../python/')");

    PyObject* pModule = PyImport_ImportModule("test_model"); //文件名
    if (!pModule) {
        std::cout << "Cant open python file!" << std::endl;
        return -1;
    }
    PyObject* pArgs1 = PyTuple_New(1);
    PyTuple_SetItem(pArgs1, 0, Py_BuildValue("s", "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/data/0000.png"));

    PyObject* pFunc = PyObject_GetAttrString(pModule, "test_merge_model");//函数
    if(!PyCallable_Check(pFunc) || pFunc == NULL) {
        return -1;
    }
    PyObject* pResult = PyObject_CallObject(pFunc, pArgs1);

    if(pResult == NULL)
        return -1;

    for(int i =0; i< PyList_Size(pResult); ++i) {
        uchar *data = (uchar *)PyByteArray_AsString(PyList_GetItem(pResult, i));
        cv::Mat img( 256, 512, CV_8UC3, data);
        mat_list.push_back(img);
    }
    Py_Finalize();
    return 0;
}
