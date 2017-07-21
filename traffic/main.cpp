#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <darknet.h>

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
                   int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.3;
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = letterbox_image(im, net.w, net.h);
        //image sized = resize_image(im, net.w, net.h);
        //image sized2 = resize_max(im, net.w);
        //image sized = crop_image(sized2, -((net.w - sized2.w)/2), -((net.h - sized2.h)/2), net.w, net.h);
        //resize_network(&net, sized.w, sized.h);
        layer l = net.layers[net.n-1];

        box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes + 1, sizeof(float *));
        float **masks = 0;
        if (l.coords > 4){
            masks = (float**)calloc(l.w*l.h*l.n, sizeof(float*));
            for(j = 0; j < l.w*l.h*l.n; ++j) masks[j] = (float *)calloc(l.coords-4, sizeof(float *));
        }

        float *X = sized.data;
        time=what_time_is_it_now();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
        get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, masks, 0, 0, hier_thresh, 1);
        if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        //else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, masks, names, alphabet, l.classes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
            cvNamedWindow("predictions", CV_WINDOW_NORMAL);
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions");
//            cvWaitKey(0);
//            cvDestroyAllWindows();
        }

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
        if (filename) break;
    }
}



int main(int ac, char* av[]) {
    std::string coco_cfg = "data/coco.data";
    std::string yolo_cfg = "data/yolo.cfg";
    std::string yolo_weights = "data/yolo.weights";
    std::string image_filename = "data/008100.jpg";

    test_detector(coco_cfg.c_str(), yolo_cfg.c_str(), yolo_weights.c_str(), image_filename.c_str(), 0.25, 0.5, NULL, 0);
    auto image = cv::imread(image_filename);
    cv::namedWindow("frame", cv::WINDOW_NORMAL);
    cv::imshow("frame", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
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