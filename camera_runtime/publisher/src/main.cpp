#include "libobsensor/ObSensor.hpp"

#include <opencv2/opencv.hpp>
#include <zmq.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr const char *kTopicName          = "frames.rgbd.v1";
constexpr const char *kDefaultPubEndpoint = "tcp://127.0.0.1:5557";

std::atomic<bool> gStop{false};

struct RuntimeOptions {
    std::string endpoint    = kDefaultPubEndpoint;
    int         jpegQuality = 85;
    int         waitMs      = 100;
    enum class AlignPreference {
        Auto,
        Disable,
        Hw,
        Sw,
    } alignPreference = AlignPreference::Auto;
};

struct StreamConfigResult {
    std::shared_ptr<ob::Config>             config;
    std::shared_ptr<ob::VideoStreamProfile> colorProfile;
    std::shared_ptr<ob::VideoStreamProfile> depthProfile;
    OBAlignMode                             alignMode = ALIGN_DISABLE;
};

std::string readEnvString(const char *name, const std::string &defaultValue) {
    const char *v = std::getenv(name);
    if(!v || std::string(v).empty()) {
        return defaultValue;
    }
    return std::string(v);
}

int readEnvInt(const char *name, int defaultValue) {
    const char *v = std::getenv(name);
    if(!v) {
        return defaultValue;
    }
    try {
        return std::stoi(v);
    }
    catch(...) {
        return defaultValue;
    }
}

RuntimeOptions::AlignPreference parseAlignPreference(const std::string &raw) {
    std::string v = raw;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if(v == "disable" || v == "off" || v == "none") {
        return RuntimeOptions::AlignPreference::Disable;
    }
    if(v == "hw" || v == "d2c_hw") {
        return RuntimeOptions::AlignPreference::Hw;
    }
    if(v == "sw" || v == "d2c_sw") {
        return RuntimeOptions::AlignPreference::Sw;
    }
    return RuntimeOptions::AlignPreference::Auto;
}

RuntimeOptions loadOptions() {
    RuntimeOptions options;
    options.endpoint    = readEnvString("DABAI_PUB_ENDPOINT", kDefaultPubEndpoint);
    options.jpegQuality = std::max(50, std::min(95, readEnvInt("DABAI_JPEG_QUALITY", 85)));
    options.waitMs      = std::max(30, std::min(200, readEnvInt("DABAI_WAIT_MS", 100)));
    options.alignPreference = parseAlignPreference(readEnvString("DABAI_ALIGN_MODE", "auto"));
    return options;
}

void onSignal(int) {
    gStop.store(true);
}

std::string alignModeToString(OBAlignMode mode) {
    switch(mode) {
    case ALIGN_DISABLE:
        return "DISABLE";
    case ALIGN_D2C_HW_MODE:
        return "D2C_HW";
    case ALIGN_D2C_SW_MODE:
        return "D2C_SW";
    default:
        return "UNKNOWN";
    }
}

std::shared_ptr<ob::VideoStreamProfile> chooseColorProfile(ob::Pipeline &pipeline) {
    auto colorProfiles = pipeline.getStreamProfileList(OB_SENSOR_COLOR);
    if(!colorProfiles || colorProfiles->count() == 0) {
        return nullptr;
    }

    try {
        // Prefer square input for downstream YOLO preprocessing.
        return colorProfiles->getVideoStreamProfile(640, 640, OB_FORMAT_MJPG, 30);
    }
    catch(...) {
    }

    try {
        return colorProfiles->getVideoStreamProfile(640, 480, OB_FORMAT_MJPG, 30);
    }
    catch(...) {
    }

    try {
        return colorProfiles->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_MJPG, OB_FPS_ANY);
    }
    catch(...) {
    }

    try {
        auto profile = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
        return profile->as<ob::VideoStreamProfile>();
    }
    catch(...) {
        return nullptr;
    }
}

std::shared_ptr<ob::VideoStreamProfile> chooseDepthProfile(
    ob::Pipeline                                     &pipeline,
    const std::shared_ptr<ob::VideoStreamProfile>   &colorProfile,
    std::shared_ptr<ob::StreamProfileList>          &depthProfileList,
    OBAlignMode                                     &alignMode,
    RuntimeOptions::AlignPreference                  alignPreference
) {
    alignMode = ALIGN_DISABLE;

    auto tryHwAlign = [&]() -> bool {
        try {
            depthProfileList = pipeline.getD2CDepthProfileList(colorProfile, ALIGN_D2C_HW_MODE);
            if(depthProfileList && depthProfileList->count() > 0) {
                alignMode = ALIGN_D2C_HW_MODE;
                return true;
            }
        }
        catch(...) {
        }
        return false;
    };

    auto trySwAlign = [&]() -> bool {
        try {
            depthProfileList = pipeline.getD2CDepthProfileList(colorProfile, ALIGN_D2C_SW_MODE);
            if(depthProfileList && depthProfileList->count() > 0) {
                alignMode = ALIGN_D2C_SW_MODE;
                return true;
            }
        }
        catch(...) {
        }
        return false;
    };

    if(colorProfile && alignPreference != RuntimeOptions::AlignPreference::Disable) {
        bool found = false;
        switch(alignPreference) {
        case RuntimeOptions::AlignPreference::Hw:
            found = tryHwAlign();
            break;
        case RuntimeOptions::AlignPreference::Sw:
            found = trySwAlign();
            break;
        case RuntimeOptions::AlignPreference::Auto:
            found = tryHwAlign();
            if(!found) {
                found = trySwAlign();
            }
            break;
        case RuntimeOptions::AlignPreference::Disable:
            break;
        }
        if(!found) {
            depthProfileList = nullptr;
            alignMode        = ALIGN_DISABLE;
        }
    }

    if(!depthProfileList || depthProfileList->count() == 0) {
        depthProfileList = pipeline.getStreamProfileList(OB_SENSOR_DEPTH);
        alignMode        = ALIGN_DISABLE;
    }

    if(!depthProfileList || depthProfileList->count() == 0) {
        return nullptr;
    }

    try {
        if(colorProfile) {
            auto matched = depthProfileList->getVideoStreamProfile(OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_Y16, colorProfile->fps());
            if(matched) {
                return matched;
            }
        }
    }
    catch(...) {
    }

    try {
        auto defaultProfile = depthProfileList->getProfile(OB_PROFILE_DEFAULT);
        return defaultProfile->as<ob::VideoStreamProfile>();
    }
    catch(...) {
        return nullptr;
    }
}

StreamConfigResult makeStreamConfig(ob::Pipeline &pipeline, RuntimeOptions::AlignPreference alignPreference) {
    StreamConfigResult result;
    result.config = std::make_shared<ob::Config>();

    result.colorProfile = chooseColorProfile(pipeline);
    if(!result.colorProfile) {
        throw std::runtime_error("Color stream is required for YOLO inference, but no color profile is available.");
    }
    result.config->enableStream(result.colorProfile);

    std::shared_ptr<ob::StreamProfileList> depthProfileList;
    result.depthProfile = chooseDepthProfile(pipeline, result.colorProfile, depthProfileList, result.alignMode, alignPreference);
    if(!result.depthProfile) {
        throw std::runtime_error("No depth profile available.");
    }
    result.config->enableStream(result.depthProfile);
    result.config->setAlignMode(result.alignMode);

    return result;
}

bool depthFrameHasSignal(const std::shared_ptr<ob::DepthFrame> &depthFrame) {
    if(!depthFrame || depthFrame->dataSize() < sizeof(uint16_t)) {
        return false;
    }
    const auto *raw = reinterpret_cast<const uint16_t *>(depthFrame->data());
    const int   cnt = static_cast<int>(depthFrame->dataSize() / sizeof(uint16_t));
    if(cnt <= 0) {
        return false;
    }
    const int step = std::max(1, cnt / 2048);
    for(int i = 0; i < cnt; i += step) {
        if(raw[i] > 0) {
            return true;
        }
    }
    return false;
}

bool colorFrameToBgr(const std::shared_ptr<ob::ColorFrame> &colorFrame, cv::Mat &outBgr) {
    const int width  = static_cast<int>(colorFrame->width());
    const int height = static_cast<int>(colorFrame->height());

    switch(colorFrame->format()) {
    case OB_FORMAT_BGR: {
        outBgr = cv::Mat(height, width, CV_8UC3, const_cast<void *>(colorFrame->data())).clone();
        return true;
    }
    case OB_FORMAT_RGB: {
        cv::Mat raw(height, width, CV_8UC3, const_cast<void *>(colorFrame->data()));
        cv::cvtColor(raw, outBgr, cv::COLOR_RGB2BGR);
        return true;
    }
    case OB_FORMAT_BGRA: {
        cv::Mat raw(height, width, CV_8UC4, const_cast<void *>(colorFrame->data()));
        cv::cvtColor(raw, outBgr, cv::COLOR_BGRA2BGR);
        return true;
    }
    case OB_FORMAT_RGBA: {
        cv::Mat raw(height, width, CV_8UC4, const_cast<void *>(colorFrame->data()));
        cv::cvtColor(raw, outBgr, cv::COLOR_RGBA2BGR);
        return true;
    }
    case OB_FORMAT_YUYV:
    case OB_FORMAT_YUY2: {
        cv::Mat raw(height, width, CV_8UC2, const_cast<void *>(colorFrame->data()));
        cv::cvtColor(raw, outBgr, cv::COLOR_YUV2BGR_YUY2);
        return true;
    }
    case OB_FORMAT_UYVY: {
        cv::Mat raw(height, width, CV_8UC2, const_cast<void *>(colorFrame->data()));
        cv::cvtColor(raw, outBgr, cv::COLOR_YUV2BGR_UYVY);
        return true;
    }
    case OB_FORMAT_NV12: {
        cv::Mat raw(height * 3 / 2, width, CV_8UC1, const_cast<void *>(colorFrame->data()));
        cv::cvtColor(raw, outBgr, cv::COLOR_YUV2BGR_NV12);
        return true;
    }
    case OB_FORMAT_NV21: {
        cv::Mat raw(height * 3 / 2, width, CV_8UC1, const_cast<void *>(colorFrame->data()));
        cv::cvtColor(raw, outBgr, cv::COLOR_YUV2BGR_NV21);
        return true;
    }
    case OB_FORMAT_MJPG: {
        cv::Mat raw(1, static_cast<int>(colorFrame->dataSize()), CV_8UC1, const_cast<void *>(colorFrame->data()));
        outBgr = cv::imdecode(raw, cv::IMREAD_COLOR);
        return !outBgr.empty();
    }
    default:
        return false;
    }
}

bool encodeColorJpeg(const std::shared_ptr<ob::ColorFrame> &colorFrame, int jpegQuality, std::vector<uint8_t> &jpeg) {
    if(!colorFrame) {
        return false;
    }

    if(colorFrame->format() == OB_FORMAT_MJPG) {
        jpeg.resize(colorFrame->dataSize());
        std::memcpy(jpeg.data(), colorFrame->data(), colorFrame->dataSize());
        return true;
    }

    cv::Mat bgr;
    if(!colorFrameToBgr(colorFrame, bgr)) {
        return false;
    }

    std::vector<int> encodeParams{cv::IMWRITE_JPEG_QUALITY, jpegQuality};
    return cv::imencode(".jpg", bgr, jpeg, encodeParams);
}

bool zmqSendMultiPart(void *socket, const std::string &topic, const std::string &meta, const std::vector<uint8_t> &rgbJpeg, const void *depthData, int depthSize) {
    if(zmq_send(socket, topic.data(), static_cast<int>(topic.size()), ZMQ_SNDMORE) < 0) {
        return false;
    }
    if(zmq_send(socket, meta.data(), static_cast<int>(meta.size()), ZMQ_SNDMORE) < 0) {
        return false;
    }
    if(zmq_send(socket, rgbJpeg.data(), static_cast<int>(rgbJpeg.size()), ZMQ_SNDMORE) < 0) {
        return false;
    }
    if(zmq_send(socket, depthData, depthSize, 0) < 0) {
        return false;
    }
    return true;
}

std::string buildMetaJson(
    uint64_t frameId,
    uint64_t tsUs,
    int      rgbW,
    int      rgbH,
    int      depthW,
    int      depthH,
    float    depthScale,
    float    fx,
    float    fy,
    float    cx,
    float    cy,
    const std::string &alignMode
) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << "{"
        << "\"frame_id\":" << frameId << ","
        << "\"ts_us\":" << tsUs << ","
        << "\"rgb_w\":" << rgbW << ","
        << "\"rgb_h\":" << rgbH << ","
        << "\"depth_w\":" << depthW << ","
        << "\"depth_h\":" << depthH << ","
        << "\"depth_scale\":" << depthScale << ","
        << "\"fx\":" << fx << ","
        << "\"fy\":" << fy << ","
        << "\"cx\":" << cx << ","
        << "\"cy\":" << cy << ","
        << "\"align_mode\":\"" << alignMode << "\""
        << "}";
    return oss.str();
}

}  // namespace

int main() try {
    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    const RuntimeOptions options = loadOptions();
    std::cout << "[publisher] endpoint=" << options.endpoint << ", jpeg_quality=" << options.jpegQuality << ", wait_ms=" << options.waitMs << std::endl;

    void *zmqContext = zmq_ctx_new();
    if(!zmqContext) {
        std::cerr << "[publisher] failed to create ZeroMQ context." << std::endl;
        return EXIT_FAILURE;
    }

    void *publisher = zmq_socket(zmqContext, ZMQ_PUB);
    if(!publisher) {
        std::cerr << "[publisher] failed to create ZeroMQ PUB socket." << std::endl;
        zmq_ctx_destroy(zmqContext);
        return EXIT_FAILURE;
    }

    const int sndHwm = 2;
    const int linger = 0;
    zmq_setsockopt(publisher, ZMQ_SNDHWM, &sndHwm, sizeof(sndHwm));
    zmq_setsockopt(publisher, ZMQ_LINGER, &linger, sizeof(linger));

    if(zmq_bind(publisher, options.endpoint.c_str()) != 0) {
        std::cerr << "[publisher] bind failed: " << zmq_strerror(zmq_errno()) << std::endl;
        zmq_close(publisher);
        zmq_ctx_destroy(zmqContext);
        return EXIT_FAILURE;
    }

    ob::Context::setLoggerSeverity(OB_LOG_SEVERITY_WARN);
    uint64_t frameId = 0;
    bool     forceAlignDisable = false;

    while(!gStop.load()) {
        try {
            ob::Pipeline pipeline;
            auto         preferredAlign = forceAlignDisable ? RuntimeOptions::AlignPreference::Disable : options.alignPreference;
            auto         streamConfig   = makeStreamConfig(pipeline, preferredAlign);
            pipeline.start(streamConfig.config);

            try {
                pipeline.enableFrameSync();
            }
            catch(...) {
                std::cout << "[publisher] frame sync unavailable, continue without sync." << std::endl;
            }

            const auto cameraParam = pipeline.getCameraParam();
            const float fx = (cameraParam.rgbIntrinsic.fx > 1.0f) ? cameraParam.rgbIntrinsic.fx : cameraParam.depthIntrinsic.fx;
            const float fy = (cameraParam.rgbIntrinsic.fy > 1.0f) ? cameraParam.rgbIntrinsic.fy : cameraParam.depthIntrinsic.fy;
            const float cx = (cameraParam.rgbIntrinsic.cx > 1.0f) ? cameraParam.rgbIntrinsic.cx : cameraParam.depthIntrinsic.cx;
            const float cy = (cameraParam.rgbIntrinsic.cy > 1.0f) ? cameraParam.rgbIntrinsic.cy : cameraParam.depthIntrinsic.cy;

            std::cout << "[publisher] started. align_mode=" << alignModeToString(streamConfig.alignMode) << std::endl;
            std::cout << "[publisher] color_profile="
                      << (streamConfig.colorProfile ? std::to_string(streamConfig.colorProfile->width()) + "x" + std::to_string(streamConfig.colorProfile->height()) : "none")
                      << ", depth_profile=" << streamConfig.depthProfile->width() << "x" << streamConfig.depthProfile->height() << std::endl;

            auto lastLogTs = std::chrono::steady_clock::now();
            int  sentCount = 0;
            int  zeroDepthStreak = 0;
            bool shouldRestartWithoutAlign = false;

            while(!gStop.load()) {
                auto frameset = pipeline.waitForFrames(options.waitMs);
                if(!frameset) {
                    continue;
                }

                auto colorFrame = frameset->colorFrame();
                auto depthFrame = frameset->depthFrame();
                if(!colorFrame || !depthFrame) {
                    continue;
                }

                std::vector<uint8_t> jpeg;
                if(!encodeColorJpeg(colorFrame, options.jpegQuality, jpeg)) {
                    continue;
                }

                const int depthW     = static_cast<int>(depthFrame->width());
                const int depthH     = static_cast<int>(depthFrame->height());
                const int depthBytes = static_cast<int>(depthFrame->dataSize());
                const float depthScale = depthFrame->getValueScale();
                const uint64_t tsUs = depthFrame->timeStampUs();

                if(depthFrameHasSignal(depthFrame)) {
                    zeroDepthStreak = 0;
                }
                else {
                    zeroDepthStreak += 1;
                    if(zeroDepthStreak >= 45 && streamConfig.alignMode != ALIGN_DISABLE) {
                        std::cout << "[publisher] depth appears all-zero under " << alignModeToString(streamConfig.alignMode)
                                  << ", restarting with ALIGN_DISABLE fallback." << std::endl;
                        forceAlignDisable       = true;
                        shouldRestartWithoutAlign = true;
                        break;
                    }
                }

                frameId += 1;
                auto meta = buildMetaJson(
                    frameId,
                    tsUs,
                    static_cast<int>(colorFrame->width()),
                    static_cast<int>(colorFrame->height()),
                    depthW,
                    depthH,
                    depthScale,
                    fx,
                    fy,
                    cx,
                    cy,
                    alignModeToString(streamConfig.alignMode)
                );

                if(!zmqSendMultiPart(publisher, kTopicName, meta, jpeg, depthFrame->data(), depthBytes)) {
                    std::cerr << "[publisher] send failed: " << zmq_strerror(zmq_errno()) << std::endl;
                    break;
                }

                sentCount += 1;
                auto now = std::chrono::steady_clock::now();
                if(std::chrono::duration_cast<std::chrono::seconds>(now - lastLogTs).count() >= 1) {
                    std::cout << "[publisher] sent_fps~" << sentCount << ", last_frame_id=" << frameId << std::endl;
                    sentCount = 0;
                    lastLogTs = now;
                }
            }

            pipeline.stop();
            if(shouldRestartWithoutAlign) {
                std::this_thread::sleep_for(std::chrono::milliseconds(400));
                continue;
            }
        }
        catch(const ob::Error &e) {
            std::cerr << "[publisher] SDK error function=" << e.getName() << ", args=" << e.getArgs() << ", message=" << e.getMessage() << ", type=" << e.getExceptionType()
                      << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
        }
        catch(const std::exception &e) {
            std::cerr << "[publisher] exception: " << e.what() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(800));
        }
    }

    zmq_close(publisher);
    zmq_ctx_destroy(zmqContext);
    std::cout << "[publisher] stopped." << std::endl;
    return EXIT_SUCCESS;
}
catch(const std::exception &e) {
    std::cerr << "[publisher] fatal: " << e.what() << std::endl;
    return EXIT_FAILURE;
}
