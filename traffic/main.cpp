#include <iostream>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

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
