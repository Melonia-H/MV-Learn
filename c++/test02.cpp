void TestTensor::cvMat2tfTensor(cv::Mat input, tensorflow::Tensor& outputTensor)
{

    auto outputTensorMapped = outputTensor.tensor<float, 3>();

    int height = input.size().height;
    int width = input.size().width;
    int depth = input.channels();

    const float* data = (float*)input.data;
    for (int y = 0; y < height; ++y) {
        const float* dataRow = data + (y * width * depth);
        for (int x = 0; x < width; ++x) {
            const float* dataPixel = dataRow + (x * depth);
            for (int c = 0; c < depth; ++c) {
                const float* dataValue = dataPixel + c;
                outputTensorMapped(y, x, c) = *dataValue;
            }
        }
    }

}

cv::Mat TestTensor::MeanImage(cv::Mat image)
{

    cv::Mat result;
    int height = image.size().height;
    int width = image.size().width;

    result.create(cv::Size(width, height), CV_32FC3);
    float VGG_MEAN[3] = {103.939, 116.779, 123.68};

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int a = image.at<cv::Vec3b>(y,x)[0];
            int b = image.at<cv::Vec3b>(y,x)[1];
            int c = image.at<cv::Vec3b>(y,x)[2];
            float tmp1 = a-VGG_MEAN[0];
            float tmp2 = b-VGG_MEAN[2];
            float tmp3 = c-VGG_MEAN[3];
            result.at<cv::Vec3f>(y,x) = cv::Vec3f(tmp1, tmp2, tmp3);
        }
    }
    return result;
}


void TestTensor::test_lanenet()
{
    const std::string pathToGraph = "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/tusimple_model/tusimple_lanenet/tusimple_lanenet_enet_2019-06-26-14-54-26.ckpt-21000.meta";
    const std::string checkpointPath = "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/tusimple_model/tusimple_lanenet/tusimple_lanenet_enet_2019-06-26-14-54-26.ckpt-21000";

    tensorflow::Status status;
    tensorflow::Session* session = tensorflow::NewSession(tensorflow::SessionOptions());
    if (session == nullptr) {
        throw std::runtime_error("Could not create Tensorflow session.");
    }

    tensorflow::Scope root = tensorflow::Scope::NewRootScope();

    // 01-- 读入我们预先定义好的模型的计算图的拓扑结构
    tensorflow::MetaGraphDef graph_def;
    status = ReadBinaryProto(tensorflow::Env::Default(), pathToGraph, &graph_def);
    if (!status.ok()) {
        throw std::runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
    }

    // 02-- 利用读入的模型的图的拓扑结构构建一个session
    status = session->Create(graph_def.graph_def());
    if (!status.ok()) {
        throw std::runtime_error("Error creating graph: " + status.ToString());
    }

    // 03-- 读入预先训练好的模型的权重
    tensorflow::Tensor checkpointPathTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
    checkpointPathTensor.scalar<std::string>()() = checkpointPath;
    status = session->Run({{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor},}, {}, {graph_def.saver_def().restore_op_name()},nullptr);
    if (!status.ok()) {
        throw std::runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
    }

    std::string input_layer ="input_tensor";
    std::string input_layer1 ="net_phase";

    //04-- 构造模型的输入，相当与python版本中的feed

    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({256, 512, 3}));

    tensorflow::Tensor phase(tensorflow::DT_BOOL, tensorflow::TensorShape());
    auto matrix1 = phase.scalar<bool>();
    matrix1()= false;

    cv::Mat gt_image = cv::imread("/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/data/0000.png");
    cv::resize(gt_image, gt_image, cv::Size(512, 256), cv::INTER_LINEAR);
    cv::Mat laneNet_image = MeanImage(gt_image);
    this->cvMat2tfTensor(laneNet_image, tensor);

    tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({8,256, 512,3}));
    auto outputTensorMapped = input_tensor.tensor<float, 4>();
    auto matrix = tensor.tensor<float, 3>();

    for (int y = 0; y < 256; ++y) {
        for (int x = 0; x < 512; ++x) {
            for (int c = 0; c < 3; ++c) {
                outputTensorMapped(0, y, x, c) = matrix(y, x, c);
                outputTensorMapped(1, y, x, c) = matrix(y, x, c);
                outputTensorMapped(2, y, x, c) = matrix(y, x, c);
                outputTensorMapped(3, y, x, c) = matrix(y, x, c);
                outputTensorMapped(4, y, x, c) = matrix(y, x, c);
                outputTensorMapped(5, y, x, c) = matrix(y, x, c);
                outputTensorMapped(6, y, x, c) = matrix(y, x, c);
                outputTensorMapped(7, y, x, c) = matrix(y, x, c);
            }
        }
    }


    std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = {
        {input_layer, input_tensor},
        {input_layer1, phase}
    };

    std::vector<tensorflow::Tensor> output_tensor;
    std::string output_layer1 = "lanenet_model/pix_embedding_relu";
    std::vector<std::string> output_tensor_names = {output_layer1};

    status = session->Run({inputs}, output_tensor_names, {}, &output_tensor);
    if (!status.ok()) {
        std::cout << "\tRunning model failed: " << status << std::endl;
    } else {
        auto output_y = output_tensor[0].scalar<float>();
        std::cout << output_y() << "\n";
    }
}



error: 2019-07-05 14:03:34.010127: F tensorflow/core/framework/tensor.cc:639] Check failed: 1 == NumElements() (1 vs. 4194304)Must have a one element tensor

