#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <darknet.h>
#include <vector>

//using namespace std;

namespace po = boost::program_options;
namespace fs = boost::filesystem;


void test_detector(const char *datacfg,
                   const char *cfgfile,
                   const char *weightfile,
                   const char *filename,
                   float thresh,
                   float hier_thresh,
                   const char *outfile,
                   int fullscreen) {

    // create network
    network net = parse_network_cfg(cfgfile);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);

    srand(2222222);
    double time;
    int j;
    float nms = .3;

    // load, resize and relayout image
    image im = load_image_color(filename, 0, 0);
    image sized = letterbox_image(im, net.w, net.h);

    // allocate bounding boxs
    const layer &l = net.layers[net.n - 1];
    printf("Layer %d: width %d: height: %d n: %d\n", net.n - 1, l.w, l.h, l.n);

    std::vector<box> boxes(l.w * l.h * l.n, {0});
    float **probs = (float **) calloc(l.w * l.h * l.n, sizeof(float *));
    for (j = 0; j < l.w * l.h * l.n; ++j) probs[j] = (float *) calloc(l.classes + 1, sizeof(float *));

    // predict
    time = what_time_is_it_now();
    network_predict(net, sized.data);
    // extract network data to bounding boxes
    get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes.data(), 0, 0, 0, hier_thresh, 1);
    if (nms) do_nms_obj(boxes.data(), probs, l.w * l.h * l.n, l.classes, nms);
    printf("%s: Predicted in %f seconds.\n", filename, what_time_is_it_now() - time);

    // drawing directions
    auto options = read_data_cfg(datacfg);
    auto name_list = option_find_str(options, "names", "data/names.list");
    auto names = get_labels(name_list);
    image **alphabet = load_alphabet();
    draw_detections(im, l.w * l.h * l.n, thresh, boxes.data(), probs, 0, names, alphabet, l.classes);

    if (outfile) {
        save_image(im, outfile);
    } else {
        save_image(im, "predictions");
        cvNamedWindow("predictions", CV_WINDOW_NORMAL);
        if (fullscreen) {
            cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        }
        show_image(im, "predictions");
    }

    free_image(im);
    free_image(sized);
    //free(boxes);
    free_ptrs((void **) probs, l.w * l.h * l.n);
}

network createNetwork(const std::string& network_config, const std::string& network_weights) {
    network net = parse_network_cfg(network_config.c_str());
    load_weights(&net, network_weights.c_str());
    set_batch_network(&net, 1);
    return net;
}

image loadImage(const std::string& filename, int networkWidth, int networkHeight) {
    return letterbox_image(load_image_color(filename.c_str(), 0, 0), networkWidth, networkHeight);
}

image convert(const cv::Mat& cv_image, int networkWidth, int networkHeight) {
    auto data = cv_image.data;
    int h = cv_image.rows;
    int w = cv_image.cols;
    int c = cv_image.dims;
    int step = cv_image.step;
    int i, j, k;

    image out;
    out.h = h;
    out.w = w;
    out.c = step;

    out.data = (float*) calloc(h*w*c, sizeof(float));

    for(i = 0; i < h; ++i) {
        for(k= 0; k < c; ++k) {
            for(j = 0; j < w; ++j) {
                out.data[k*w*h + i*w + j] = data[i*step + j*c + k] / 255.f;
            }
        }
    }

    auto ret = letterbox_image(out, networkWidth, networkHeight);
    free(out.data);
    return ret;
}

int main(int ac, char *av[]) {
    std::string coco_cfg = "data/coco.data";
    std::string yolo_cfg = "data/yolo.cfg";
    std::string yolo_weights = "data/yolo.weights";
    std::string image_filename = "data/008100.jpg";

    auto yolo = createNetwork(yolo_cfg, yolo_weights);
    auto image = cv::imread(image_filename);
    auto out = convert(image, yolo.w, yolo.h);
    rgbgr_image(out);
    network_predict(yolo, out.data);
    printf("Predicted in seconds.\n");



//
//    test_detector(coco_cfg.c_str(), yolo_cfg.c_str(), yolo_weights.c_str(), image_filename.c_str(), 0.25, 0.5, NULL, 0);
//    auto image = cv::imread(image_filename);
//    cv::namedWindow("frame", cv::WINDOW_NORMAL);
//    cv::imshow("frame", image);
//    cv::waitKey(0);
//    cv::destroyAllWindows();
}


/*
int main(int ac, char* av[]) {
    try {
        fs::path input_file;
        po::options_description generic("Options");
        generic.add_options()
                ("help,h", "produce help message")
                ("input,i", po::value<fs::path>(&input_file), "input video file");
        po::options_description cmdline_options;
        cmdline_options.add(generic);

        po::options_description visible("Allowed options");
        visible.add(generic);

        po::variables_map vm;
        store(po::command_line_parser(ac, (const char* const*) av).options(cmdline_options).run(), vm);
        notify(vm);

        if (vm.count("help")) {
            cout << visible << '\n';
            return 0;
        }

        if (!vm.count("input")) {
            cout << "no input file given" << '\n';
            return 0;
        }

        if (!fs::exists(input_file)) {
            cout << "input file doesn't exist" << '\n';
            return 0;
        }

        cv::VideoCapture capture(cv::String(input_file.string()));

        if (!capture.isOpened()) {
            cout << "Could not open video file" << '\n';
            return 0;
        }

        const auto frameCount = capture.get(cv::CAP_PROP_FRAME_COUNT);
        const auto frameWidth = capture.get(cv::CAP_PROP_FRAME_WIDTH);
        const auto frameHeight = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
        const auto framePS = capture.get(cv::CAP_PROP_FPS);

        cout << "File: " << input_file.filename() << '\n'
             << frameCount << " frames " << frameWidth << "x" << frameHeight << "@" << framePS << "fps" << '\n';
        cv::Mat frame;
        // cv::namedWindow("frame",1);
        while (capture.read(frame)) {
            // cv::imshow("frame", frame);
            // if(cv::waitKey(1) >= 0) break;
        }
    }
    catch (exception& e) {
        cout << e.what();
        return 1;
    }

    return 0;
}
*/