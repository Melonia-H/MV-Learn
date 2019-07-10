#include "lanenet_cluster.h"
#include <numpy/arrayobject.h>

LaneNetCluster::LaneNetCluster()
{

}
int LaneNetCluster::init_numpy()
{
    import_array();
}


int LaneNetCluster::GetLaneMask(long int binary_seg_ret[1][256][512], float instance_seg_ret[1][256][512][4])
{
    Py_Initialize();// 导入模块
    init_numpy();

    if (!Py_IsInitialized())
        return -1;

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('../python/')");

    PyObject* pModule = PyImport_ImportModule("test1"); // 文件名
    if (!pModule) {
        std::cout << "Cant open python file!" << std::endl;
        return -1;
    }

    PyObject* pArgs = PyTuple_New(3);
    npy_intp Dims[3] = {1, 256, 512}; //给定维度信息
    PyObject *PyArray = PyArray_SimpleNewFromData(3, Dims, NPY_INT, binary_seg_ret); //第一个参数2表示维度，第二个为维度数组Dims,第三个参数指出数组的类型，第四个参数为数组

    npy_intp Dims1[4] = {1, 256, 512, 4}; //给定维度信息
    PyObject *PyArray1 = PyArray_SimpleNewFromData(4, Dims1, NPY_INT, instance_seg_ret);

    std::string file_name = "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/data/0000.png";
    cv::Mat image = cv::imread(file_name);
    cv::resize(image, image, cv::Size(512, 256));
    uchar *data = image.data;
    npy_intp Dims2[3] = { 256, 512, 3};
    PyObject *PyArray2 = PyArray_SimpleNewFromData(3, Dims2, NPY_UBYTE, data);

    PyTuple_SetItem(pArgs, 0, PyArray);
    PyTuple_SetItem(pArgs, 1, PyArray1);
    PyTuple_SetItem(pArgs, 2, PyArray2);

    PyObject* pFunc = PyObject_GetAttrString(pModule, "test_merge_model");//函数名
    if(!PyCallable_Check(pFunc) || pFunc == NULL) { //
        std::cout << " can not find func " << std::endl;
        return -1;
    }
    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    if(pResult == NULL)
        return -1;

    //ConvertToArray(pResult);

    std::cout << " stop " << std::endl;
}

/*
I0709 16:37:19.122425 22397 test1.py:16] ----------input------------------
I0709 16:37:19.122578 22397 test1.py:17] (1, 256, 512)     binary_seg_image
I0709 16:37:19.122637 22397 test1.py:18] (1, 256, 512, 4)  instance_seg_image
I0709 16:37:19.122683 22397 test1.py:19] (256, 512, 3)     gt_image
I0709 16:37:19.122725 22397 test1.py:20] ----------------------------

I0709 16:37:19.137172 22397 test1.py:36] -------------output------------------
I0709 16:37:19.137296 22397 test1.py:37] (256, 512)        predict_binary    int32
I0709 18:14:06.098877 28443 test1.py:65] (256, 512, 3)     predict_lanenet   uint8
I0709 16:37:19.137406 22397 test1.py:39] (256, 512, 4)     predict_instance  uint8
I0709 16:37:19.137446 22397 test1.py:40] ----------------------------

*/

void LaneNetCluster::GetLanePts(PyObject *object)
{

}

void LaneNetCluster::ConvertToArray(PyObject *object)
{
    int predict_binary_data[256][512];
    uint8_t predict_lanenet_data[256][512][3];
    uint8_t predict_instance_data[256][512][4];

    if(PyList_Check(object)) {
        int SizeOfList = PyList_Size(object);
        if(SizeOfList == 3) {
            PyArrayObject *predict_binary_array = (PyArrayObject *)PyList_GetItem(object, 0);
            PyArrayObject *predict_lanenet_array = (PyArrayObject *)PyList_GetItem(object, 1);
            PyArrayObject *predict_instance_array = (PyArrayObject *)PyList_GetItem(object, 2);

            {
                int Rows = predict_binary_array->dimensions[0];
                int columns = predict_binary_array->dimensions[1];

                for(int Index_m = 0; Index_m < Rows; Index_m++) {
                    for(int Index_n = 0; Index_n < columns; Index_n++) {
                        predict_binary_data[Index_m][Index_n] = *(int *)(predict_binary_array->data + Index_m * predict_binary_array->strides[0] + Index_n * predict_binary_array->strides[1]);
                    }
                }
            }
            {
                int Rows = predict_lanenet_array->dimensions[0];
                int columns = predict_lanenet_array->dimensions[1];
                int channel = predict_lanenet_array->dimensions[2];

                for(int Index_m = 0; Index_m < Rows; Index_m++) {
                    for(int Index_n = 0; Index_n < columns; Index_n++) {
                        for(int Index_c = 0; Index_c < channel; Index_c++) {
                            predict_lanenet_data[Index_m][Index_n][Index_c] = *(uint8_t *)(predict_lanenet_array->data + Index_m * predict_lanenet_array->strides[0] +
                                    Index_n * predict_lanenet_array->strides[1] + Index_c * predict_lanenet_array->strides[2]);
                        }
                    }
                }
            }
            {
                int Rows = predict_instance_array->dimensions[0];
                int columns = predict_instance_array->dimensions[1];
                int channel = predict_instance_array->dimensions[2];

                for(int Index_m = 0; Index_m < Rows; Index_m++) {
                    for(int Index_n = 0; Index_n < columns; Index_n++) {
                        for(int Index_c = 0; Index_c < channel; Index_c++) {
                            predict_instance_data[Index_m][Index_n][Index_c] = *(uint8_t *)(predict_instance_array->data + Index_m * predict_instance_array->strides[0] +
                                    Index_n * predict_instance_array->strides[1] + Index_c * predict_instance_array->strides[2]);
                        }
                    }
                }
            }

        } else {
            std::cout<<" return error"<<std::endl;
        }
    } else {

        std::cout<<"Not a List"<<std::endl;
    }
    cv::Mat predict_binary(cv::Size(512, 256), CV_32S, predict_binary_data);
    cv::Mat predict_lanenet(cv::Size(512, 256), CV_8UC3, predict_lanenet_data);
    cv::Mat predict_instance(cv::Size(512, 256), CV_8UC4, predict_instance_data);

    cv::imwrite("predict_binary.png",predict_binary);
    cv::imwrite("predict_lanenet.png",predict_lanenet);
    cv::imwrite("predict_instance.png",predict_instance);
}
