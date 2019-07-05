#include <fstream>
#include <utility>
#include <vector>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;


static Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output)
{
    tensorflow::uint64 file_size = 0;
    TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

    string contents;
    contents.resize(file_size);

    std::unique_ptr<tensorflow::RandomAccessFile> file;
    TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

    tensorflow::StringPiece data;
    TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
    if (data.size() != file_size) {
        return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                            "' expected ", file_size, " got ",
                                            data.size());
    }
    output->scalar<string>()() = string(data);
    return Status::OK();
}

// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors)
{
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

    string input_name = "file_reader";
    string output_name = "normalized";

    // read file_name into a tensor named input
    Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
    TF_RETURN_IF_ERROR(
        ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

    // use a placeholder to read input data
    auto file_reader =
        Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
        {"input", input},
    };

    // Now try to figure out what kind of file it is and decode it.
    const int wanted_channels = 3;
    tensorflow::Output image_reader;
    if (tensorflow::str_util::EndsWith(file_name, ".png")) {
        image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                                 DecodePng::Channels(wanted_channels));
    } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
        // gif decoder returns 4-D tensor, remove the first dim
        image_reader =
            Squeeze(root.WithOpName("squeeze_first_dim"),
                    DecodeGif(root.WithOpName("gif_reader"), file_reader));
    } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
        image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
    } else {
        // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
        image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                                  DecodeJpeg::Channels(wanted_channels));
    }
    // Now cast the image data to float so we can do normal math on it.
    auto float_caster =
        Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
    // The convention for image ops in TensorFlow is that all images are expected
    // to be in batches, so that they're four-dimensional arrays with indices of
    // [batch, height, width, channel]. Because we only have a single image, we
    // have to add a batch dimension of 1 to the start with ExpandDims().
    auto dims_expander = ExpandDims(root, float_caster, 0);
    // Bilinearly resize the image to fit the required dimensions.
    auto resized = ResizeBilinear(
                       root, dims_expander,
                       Const(root.WithOpName("size"), {input_height, input_width}));

    // Subtract the mean and divide by the scale.
    Div(root.WithOpName(output_name), Sub(root, float_caster, {input_mean}),
    {input_std});

    // This runs the GraphDef network definition that we've just constructed, and
    // returns the results in the output tensor.
    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
        tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({inputs}, {output_name}, {}, out_tensors));
    return Status::OK();
}

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session)
{
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
        ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

int main(int argc, char* argv[])
{
    string image = "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/data/0001.png";
    string graph = "/home/maohui/all_files/a_programs/TestCar/laneDetect/programs/tensorflow/test/testTensorC/tusimple_model/tusimple_lanenet/other/lanenet.pb";
    int32 input_height = 256;
    int32 input_width = 512;
    float input_mean = 0;
    float input_std = 255;
    string input_layer1 = "input_tensor:0";
    string input_layer2 = "net_phase:0";
    string output_layer = "lanenet_model/pix_embedding_relu:0";
    string root_dir = "";

    auto root = tensorflow::Scope::NewRootScope();

    std::unique_ptr<tensorflow::Session> session;
    string graph_path = tensorflow::io::JoinPath(root_dir, graph);
    Status load_graph_status = LoadGraph(graph_path, &session);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return -1;
    }

    std::vector<Tensor> resized_tensors;
    string image_path = tensorflow::io::JoinPath(root_dir, image);
    Status read_tensor_status =
        ReadTensorFromImageFile(image_path, input_height, input_width, input_mean,
                                input_std, &resized_tensors);
    if (!read_tensor_status.ok()) {
        LOG(ERROR) << read_tensor_status;
        return -1;
    }
    const Tensor& resized_tensor = resized_tensors[0];

    Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({8,256, 512,3}));
    auto outputTensorMapped = input_tensor.tensor<float, 4>();
    auto matrix = resized_tensor.tensor<float, 3>();

    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
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

    Tensor phase(tensorflow::DT_BOOL, tensorflow::TensorShape());
    auto matrix1 = phase.scalar<bool>();
    matrix1()= false;

//    auto is_phase = tensorflow::ops::Const(root.WithOpName(input_layer2), false);

    std::cout << input_tensor.shape() <<std::endl;
    std::cout << phase.shape() <<std::endl;

    std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
        {input_layer1, input_tensor},
        {input_layer2, phase}
    };

    std::vector<Tensor> results;
    Status run_status = session->Run({inputs}, {output_layer}, {}, &results);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
        return -1;
    }

    return 0;
}

