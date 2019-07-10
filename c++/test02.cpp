#include <iostream>

#include <tensorflow/cc/ops/io_ops.h>
#include <tensorflow/cc/ops/parsing_ops.h>
#include <tensorflow/cc/ops/array_ops.h>
#include <tensorflow/cc/ops/math_ops.h>
#include <tensorflow/cc/ops/data_flow_ops.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace tensorflow;
using namespace tensorflow::ops;

int main()
{
    // set up your input paths
    const string pathToGraph = "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/tusimple_model/hnet/hnet-4000.meta";
    const string checkpointPath = "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/tusimple_model/hnet/hnet-4000";

    auto session = NewSession(SessionOptions());
    if (session == nullptr) {
        throw runtime_error("Could not create Tensorflow session.");
    }

    Status status;

    // Read in the protobuf graph we exported
    MetaGraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
    if (!status.ok()) {
        throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
    }

    // Add the graph to the session
    status = session->Create(graph_def.graph_def());
    if (!status.ok()) {
        throw runtime_error("Error creating graph: " + status.ToString());
    }

    // Read weights from the saved checkpoint
    Tensor checkpointPathTensor(DT_STRING, TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
    status = session->Run({{graph_def.saver_def().filename_tensor_name(), checkpointPathTensor},}, {},
    {graph_def.saver_def().restore_op_name()}, nullptr);
    if (!status.ok()) {
        throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
    }

    string image_path = "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/data/0001.png";
    cv::Mat image = cv::imread(image_path);
    cv::Mat image_vis;  //image
    cv::resize(image, image_vis, cv::Size(512, 256));

    cv::Mat image_hnet; //gt_image
    cv::resize(image, image_hnet, cv::Size(128, 64));

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({10, 64, 128, 3})); // gt_image
    tensorflow::Tensor lane_pts_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({10, 15*8, 3})); //pts
    string input1 = "Placeholder:0";
    string input2 = "Placeholder_1:0";


    auto input_tensor_mapped = input_tensor.tensor<float, 4>();
    for (int x = 0; x < 64; ++x) { //depth
        for (int y = 0; y < 128; ++y) {
            for (int z = 0; z < 3; ++z) {
                uchar *source_value = image_hnet.data;
                input_tensor_mapped(0, x, y, z) = source_value[ x * 128 * 3 + 3 * y + z];
//                input_tensor_mapped(1, x, y, z) = source_value[ x * 128 * 3 + 3 * y + z];
//                input_tensor_mapped(2, x, y, z) = source_value[ x * 128 * 3 + 3 * y + z];
//                input_tensor_mapped(3, x, y, z) = source_value[ x * 128 * 3 + 3 * y + z];
//                input_tensor_mapped(4, x, y, z) = source_value[ x * 128 * 3 + 3 * y + z];
//                input_tensor_mapped(5, x, y, z) = source_value[ x * 128 * 3 + 3 * y + z];
//                input_tensor_mapped(6, x, y, z) = source_value[ x * 128 * 3 + 3 * y + z];
//                input_tensor_mapped(7, x, y, z) = source_value[ x * 128 * 3 + 3 * y + z];
//                input_tensor_mapped(8, x, y, z) = source_value[ x * 128 * 3 + 3 * y + z];
//                input_tensor_mapped(9, x, y, z) = source_value[ x * 128 * 3 + 3 * y + z];
            }
        }
    }

    float labels[][3] = {{401, 260, 1}, {427, 270, 1}, {441, 280, 1}, {434, 290, 1}, {412, 300, 1}, {390, 310, 1}, {368, 320, 1}, {347, 330, 1},
        {325, 340, 1}, {303, 350, 1}, {277, 360, 1}, {247, 370, 1}, {216, 380, 1}, {185, 390, 1}, {154, 400, 1}, {124, 410, 1},
        {94, 420, 1},  {64, 430, 1},  {34, 440, 1},  {4, 450, 1  }, {507, 270, 2}, {521, 280, 2}, {530, 290, 2}, {539, 300, 2},
        {539, 310, 2}, {538, 320, 2}, {537, 330, 2}, {536, 340, 2}, {534, 350, 2}, {530, 360, 2}, {521, 370, 2}, {512, 380, 2},
        {504, 390, 2}, {495, 400, 2}, {486, 410, 2}, {478, 420, 2}, {469, 430, 2}, {460, 440, 2}, {452, 450, 2}, {443, 460, 2},
        {434, 470, 2}, {426, 480, 2}, {417, 490, 2}, {408, 500, 2}, {400, 510, 2}, {391, 520, 2}, {382, 530, 2}, {374, 540, 2},
        {365, 550, 2}, {355, 560, 2}, {346, 570, 2}, {337, 580, 2}, {328, 590, 2}, {318, 600, 2}, {309, 610, 2}, {300, 620, 2},
        {291, 630, 2}, {282, 640, 2}, {272, 650, 2}, {263, 660, 2}, {254, 670, 2}, {245, 680, 2}, {236, 690, 2}, {226, 700, 2},
        {217, 710, 2}, {709, 320, 3}, {729, 330, 3}, {748, 340, 3}, {764, 350, 3}, {780, 360, 3}, {795, 370, 3}, {811, 380, 3},
        {827, 390, 3}, {842, 400, 3}, {855, 410, 3}, {868, 420, 3}, {881, 430, 3}, {894, 440, 3}, {907, 450, 3}, {920, 460, 3},
        {933, 470, 3}, {946, 480, 3}, {959, 490, 3}, {972, 500, 3}, {985, 510, 3}, {999, 520, 3}, {1012, 530,3}, {1025, 540,3},
        {1039,550, 3}, {1053, 560,3}, {1066, 570,3}, {1080, 580,3}, {1094, 590,3}, {1108, 600,3}, {1122, 610,3}, {1135, 620,3},
        {1149, 630,3}, {1163, 640,3}, {1177, 650,3}, {1191, 660,3}, {1205, 670,3}, {1218, 680,3}, {1232, 690,3}, {1246, 700,3},
        {1260, 710,3}, {726, 290, 4}, {777, 300, 4}, {817, 310, 4}, {858, 320, 4}, {897, 330, 4}, {935, 340, 4}, {974, 350, 4},
        {1012, 360,4}, {1050, 370,4}, {1087, 380,4}, {1121, 390,4},{1155, 400, 4}, {1189, 410,4}, {1223, 420,4}, {1257, 430,4}
    };


    //    float labels[3][/*?*/][2] ;
    //    for(int label_index =0; label_index < 3; i++) {
    //        float pts[/*?*/][3];
    //        for(int i =0; i</*?*/; i++) {
    //            pts[i][0] = labels[label_index][1][i];
    //            pts[i][1] = labels[label_index][2][i];
    //            pts[i][3] = 1;
    //        }
    //        auto lane_pts_tensor_mapped = lane_pts_tensor.tensor<float, 2>();
    //        for (int y = 0; y < /*?*/; ++y) {
    //            for (int z = 0; z < 3; ++z) {
    //                lane_pts_tensor_mapped(x, y) = pts[x][y];
    //            }
    //        }

    //        vector<std::pair<string, Tensor> > inputs = {
    //            {input1,input_tensor},
    //            {input2,lane_pts_tensor}
    //        };

    //        std::vector<tensorflow::Tensor> finalOutput;
    //        std::string OutputName = "hnet/transfomation_coefficient/fc_output/output"; //
    //        session->Run(inputs, {OutputName}, {}, &finalOutput);
    //        std::cout << finalOutput.size()<<std::endl;

    //    }

    auto lane_pts_tensor_mapped = lane_pts_tensor.tensor<float, 3>();
    for (int x = 0; x < 15*8; ++x) {
        for (int y = 0; y < 3; ++y) {
            lane_pts_tensor_mapped(0, x, y) = labels[x][y];
//            lane_pts_tensor_mapped(1, x, y) = lane_pts_tensor_mapped(0, x, y);
//            lane_pts_tensor_mapped(2, x, y) = lane_pts_tensor_mapped(0, x, y);
//            lane_pts_tensor_mapped(3, x, y) = lane_pts_tensor_mapped(0, x, y);
//            lane_pts_tensor_mapped(4, x, y) = lane_pts_tensor_mapped(0, x, y);
//            lane_pts_tensor_mapped(5, x, y) = lane_pts_tensor_mapped(0, x, y);
//            lane_pts_tensor_mapped(6, x, y) = lane_pts_tensor_mapped(0, x, y);
//            lane_pts_tensor_mapped(7, x, y) = lane_pts_tensor_mapped(0, x, y);
//            lane_pts_tensor_mapped(8, x, y) = lane_pts_tensor_mapped(0, x, y);
//            lane_pts_tensor_mapped(9, x, y) = lane_pts_tensor_mapped(0, x, y);
        }
    }

    vector<std::pair<string, Tensor> > inputs = {
        {input1,input_tensor},
        {input2,lane_pts_tensor}
    };

    std::vector<tensorflow::Tensor> finalOutput;
    std::string OutputName = "hnet/transfomation_coefficient/fc_output/output"; //

    /*
    Tensor("hnet/transfomation_coefficient/fc_output/output:0", shape=(1, 6), dtype=float32)
    --------hnet_transformation---------
    Tensor("hnet/inference/MatMul_5:0", shape=(3, ?), dtype=float32)
    Tensor("hnet/inference/Reshape:0", shape=(3, 3), dtype=float32)
    */
    Status run_status  = session->Run(inputs, {OutputName}, {}, &finalOutput);

    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }

    std::cout << finalOutput.size()<<std::endl;
    tensorflow::Tensor ceof = finalOutput[0];
    std::cout << ceof.dims() << "," << ceof.dtype() << "," << ceof.dim_size(0)<<std::endl; //--> 10 *6

    auto ceof_mapped = ceof.tensor<float, 2>();
    for (int y = 0; y < 10; ++y) {
        for (int y = 0; y < 6; ++y) {
            std::cout << ceof_mapped(y, y) << "," ;
        }
        std::cout << std::endl;
    }

    return 0;
}

void hnet_transformation(float* gt_pts, tensorflow::Tensor transformation_coeffcient)
{
    auto root = tensorflow::Scope::NewRootScope();
    Scope scope = root.WithOpName("inference");

    //首先映射原始标签点对
    auto a = tensorflow::ops::Squeeze(scope, transformation_coeffcient);
    auto transformation_coeffcient1 = tensorflow::ops::Concat(scope, a, {1.0}, -1);

    auto multiplier = tensorflow::ops::Const({1.0, 1.0, 4.0, 1.0, 4.0, 0.25, 1.0});
    auto transformation_coeffcient2 = tensorflow::ops::Mul(scope, transformation_coeffcient1, multiplier);

    auto H_indices = tensorflow::ops::Const({{0}, {1}, {2}, {4}, {5},{7}, {8}});
    auto H_shape = tensorflow::ops::Const({9});

    auto H = tensorflow::ops::ScatterNd(scope, H_indicesm, transformation_coeffcient2, H_shape);
    auto H1 = tensorflow::ops::Reshape(scope, H, {3,3});

    auto gt_pts = tensorflow::ops::Transpose(scope, gt_pts);
    auto pts_projects = tensorflow::ops::MatMul(scope, H1, gt_pts);

    // 求解最小二乘二阶多项式拟合参数矩阵
    auto Y = tensorflow::ops::Transpose(pts_projects{1, /*:*/} / pts_projects{2, /*:*/});
    auto X = tensorflow::ops::Transpose(pts_projects{0, /*:*/} / pts_projects{2, /*:*/});

    auto Y_One = tensorflow::ops::OnesLike(scope, Y);
    auto Y_stack = tensorflow::ops::Stack(scope, {tensorflow::ops::Pow(scope, Y, 3), tensorflow::ops::Pow(scope,Y, 2), Y, Y_One}, 1);

    auto w = tensorflow::ops::MatMul(scope, tensorflow::ops::MatMul(scope, MatrixInverse(scope,tensorflow::ops::MatMul(scope, tensorflow::ops::Transpose(Y_stack),Y_stack)),
                                     tensorflow::ops::Transpose(scope, Y_stack)),tensorflow::ops::ExpandDims(scope, X, -1));

    // 利用二阶多项式参数求解拟合位置
    auto x_preds = tensorflow::ops::MatMul(scope, Y_stack, w);
    auto preds = tensorflow::ops::Transpose(scope, tensorflow::ops::Stack(scope, {
        tensorflow::ops::Squeeze(scope, x_preds, -1) * pts_projects{2,/*:*/},
        Y * pts_projects{2, /*:*/},
        pts_projects{2, /*:*/}
    }, 1) );

    auto x_transformation_back = tensorflow::ops::MatMul(scope, MatrixInverse(scope, H), preds);

    //return x_transformation_back, H
}

void MatrixInverse(Scope scope, Tensor* tensor)
{

}
