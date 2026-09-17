#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

constexpr int kStatusHeight = 148;
constexpr const char* kWindowName = "Drone annotation editor";

enum class Format { Auto, Json, Csv, Yolo };

struct Box {
    int class_id = 0;
    double x = 0;
    double y = 0;
    double width = 0;
    double height = 0;
};

struct FrameAnnotation {
    bool exists = false;
    std::vector<Box> boxes;
};

struct FrameItem {
    fs::path path;
    std::size_t annotation_index = 0;
};

struct CsvData {
    fs::path path;
    std::vector<std::string> headers;
    std::vector<std::vector<std::string>> rows;
    int frame_column = -1;
    int exists_column = -1;
    int class_column = -1;
    int x_column = -1;
    int y_column = -1;
    int width_column = -1;
    int height_column = -1;
    int x2_column = -1;
    int y2_column = -1;
};

struct Sequence {
    fs::path directory;
    fs::path frames_directory;
    std::vector<FrameItem> frames;
    std::vector<FrameAnnotation> annotations;
    Format format = Format::Json;
    fs::path annotation_path;
    std::optional<CsvData> csv;
    int image_width = 0;
    int image_height = 0;
};

struct Change {
    std::size_t sequence_index;
    std::size_t annotation_index;
    FrameAnnotation before;
    FrameAnnotation after;
};

struct Options {
    fs::path root;
    Format format = Format::Auto;
    int class_id = 0;
    int max_width = 1600;
    int max_height = 950;
    bool check_only = false;
};

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool is_image(const fs::path& path) {
    static const std::vector<std::string> extensions = {
        ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"};
    const auto extension = lower(path.extension().string());
    return std::find(extensions.begin(), extensions.end(), extension) != extensions.end();
}

std::optional<std::size_t> frame_number(const fs::path& path) {
    static const std::regex pattern(R"(^frame_(\d+))", std::regex::icase);
    std::smatch match;
    const std::string name = path.filename().string();
    if (!std::regex_search(name, match, pattern)) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(std::stoull(match[1].str()));
}

std::vector<std::string> parse_csv_row(const std::string& line) {
    std::vector<std::string> values;
    std::string value;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (character == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                value.push_back('"');
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (character == ',' && !quoted) {
            values.push_back(value);
            value.clear();
        } else if (character != '\r') {
            value.push_back(character);
        }
    }
    values.push_back(value);
    return values;
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string escaped = "\"";
    for (const char character : value) {
        escaped += character == '"' ? "\"\"" : std::string(1, character);
    }
    return escaped + "\"";
}

int find_column(const std::vector<std::string>& headers,
                const std::vector<std::string>& aliases) {
    for (std::size_t index = 0; index < headers.size(); ++index) {
        const auto name = lower(headers[index]);
        if (std::find(aliases.begin(), aliases.end(), name) != aliases.end()) {
            return static_cast<int>(index);
        }
    }
    return -1;
}

void atomic_replace(const fs::path& temporary, const fs::path& destination) {
    if (std::rename(temporary.string().c_str(), destination.string().c_str()) != 0) {
        fs::remove(temporary);
        throw std::runtime_error("Could not replace annotation file: " + destination.string());
    }
}

std::optional<fs::path> exact_or_first(const fs::path& directory,
                                       const std::string& extension) {
    const fs::path exact = directory / (directory.filename().string() + extension);
    if (fs::is_regular_file(exact)) {
        return exact;
    }
    std::vector<fs::path> candidates;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && lower(entry.path().extension().string()) == extension) {
            candidates.push_back(entry.path());
        }
    }
    std::sort(candidates.begin(), candidates.end());
    return candidates.empty() ? std::nullopt : std::optional<fs::path>(candidates.front());
}

CsvData load_csv(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Could not open CSV: " + path.string());
    }
    CsvData csv;
    csv.path = path;
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Empty CSV: " + path.string());
    }
    csv.headers = parse_csv_row(line);
    while (std::getline(input, line)) {
        auto row = parse_csv_row(line);
        row.resize(csv.headers.size());
        csv.rows.push_back(std::move(row));
    }
    csv.frame_column = find_column(csv.headers, {"frame", "frame_index", "index"});
    csv.exists_column = find_column(csv.headers, {"drone_exist", "exists", "exist", "present"});
    csv.class_column = find_column(csv.headers, {"class", "class_id", "classid"});
    csv.x_column = find_column(csv.headers, {"x", "left", "x1"});
    csv.y_column = find_column(csv.headers, {"y", "top", "y1"});
    csv.width_column = find_column(csv.headers, {"width", "w"});
    csv.height_column = find_column(csv.headers, {"height", "h"});
    csv.x2_column = find_column(csv.headers, {"x2", "right"});
    csv.y2_column = find_column(csv.headers, {"y2", "bottom"});
    if (csv.frame_column < 0) {
        throw std::runtime_error("CSV needs a frame column: " + path.string());
    }
    return csv;
}

void add_csv_column(CsvData& csv, const std::string& name, int& column) {
    if (column >= 0) {
        return;
    }
    column = static_cast<int>(csv.headers.size());
    csv.headers.push_back(name);
    for (auto& row : csv.rows) {
        row.emplace_back();
    }
}

void write_csv(const CsvData& csv) {
    const fs::path temporary = csv.path.string() + ".tmp";
    std::ofstream output(temporary);
    if (!output) {
        throw std::runtime_error("Could not write CSV: " + temporary.string());
    }
    for (std::size_t column = 0; column < csv.headers.size(); ++column) {
        if (column) output << ',';
        output << csv_escape(csv.headers[column]);
    }
    output << '\n';
    for (const auto& row : csv.rows) {
        for (std::size_t column = 0; column < csv.headers.size(); ++column) {
            if (column) output << ',';
            output << csv_escape(column < row.size() ? row[column] : "");
        }
        output << '\n';
    }
    output.close();
    atomic_replace(temporary, csv.path);
}

std::vector<FrameAnnotation> load_json(const fs::path& path) {
    cv::FileStorage storage(path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) {
        throw std::runtime_error("Could not open JSON: " + path.string());
    }
    const cv::FileNode exists_node = storage["exist"];
    const cv::FileNode boxes_node = storage["gt_rect"];
    if (exists_node.empty() || boxes_node.empty() || !exists_node.isSeq() || !boxes_node.isSeq()) {
        throw std::runtime_error("JSON must contain exist and gt_rect arrays: " + path.string());
    }
    if (exists_node.size() != boxes_node.size()) {
        throw std::runtime_error("JSON annotation arrays differ in length: " + path.string());
    }
    std::vector<FrameAnnotation> annotations(exists_node.size());
    auto exists_iterator = exists_node.begin();
    auto boxes_iterator = boxes_node.begin();
    for (std::size_t index = 0; index < annotations.size();
         ++index, ++exists_iterator, ++boxes_iterator) {
        annotations[index].exists = static_cast<int>(*exists_iterator) != 0;
        const cv::FileNode box_node = *boxes_iterator;
        if (box_node.isSeq() && box_node.size() == 4) {
            auto value = box_node.begin();
            Box box;
            box.x = static_cast<double>(*value++);
            box.y = static_cast<double>(*value++);
            box.width = static_cast<double>(*value++);
            box.height = static_cast<double>(*value++);
            annotations[index].boxes.push_back(box);
        }
    }
    return annotations;
}

void write_json(const Sequence& sequence) {
    const fs::path temporary = sequence.annotation_path.string() + ".tmp";
    cv::FileStorage storage(
        temporary.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) {
        throw std::runtime_error("Could not write JSON: " + temporary.string());
    }
    storage << "exist" << "[";
    for (const auto& annotation : sequence.annotations) {
        storage << static_cast<int>(annotation.exists && !annotation.boxes.empty());
    }
    storage << "]" << "gt_rect" << "[";
    for (const auto& annotation : sequence.annotations) {
        storage << "[";
        if (!annotation.boxes.empty()) {
            const auto& box = annotation.boxes.front();
            storage << box.x << box.y << box.width << box.height;
        }
        storage << "]";
    }
    storage << "]";
    storage.release();
    atomic_replace(temporary, sequence.annotation_path);
}

std::vector<FrameAnnotation> load_csv_annotations(CsvData& csv, std::size_t count) {
    std::vector<FrameAnnotation> annotations(count);
    const bool has_size_columns = csv.width_column >= 0 && csv.height_column >= 0;
    const bool has_corner_columns = csv.x2_column >= 0 && csv.y2_column >= 0;
    const bool has_box_columns =
        csv.x_column >= 0 && csv.y_column >= 0 && (has_size_columns || has_corner_columns);
    for (const auto& row : csv.rows) {
        const std::size_t index = static_cast<std::size_t>(std::stoull(row[csv.frame_column]));
        if (index >= annotations.size()) continue;
        const bool exists = csv.exists_column < 0 || row[csv.exists_column] == "1" ||
                            lower(row[csv.exists_column]) == "true";
        annotations[index].exists = exists;
        if (exists && has_box_columns && !row[csv.x_column].empty()) {
            Box box;
            box.class_id = csv.class_column >= 0 && !row[csv.class_column].empty()
                               ? std::stoi(row[csv.class_column])
                               : 0;
            box.x = std::stod(row[csv.x_column]);
            box.y = std::stod(row[csv.y_column]);
            box.width = has_size_columns ? std::stod(row[csv.width_column])
                                         : std::stod(row[csv.x2_column]) - box.x;
            box.height = has_size_columns ? std::stod(row[csv.height_column])
                                          : std::stod(row[csv.y2_column]) - box.y;
            annotations[index].boxes.push_back(box);
        }
    }
    return annotations;
}

void update_csv(CsvData& csv, const Sequence& sequence, std::size_t index, bool primary) {
    if (primary) {
        add_csv_column(csv, "drone_exist", csv.exists_column);
        add_csv_column(csv, "class_id", csv.class_column);
        add_csv_column(csv, "x", csv.x_column);
        add_csv_column(csv, "y", csv.y_column);
        if (csv.x2_column < 0 || csv.y2_column < 0) {
            add_csv_column(csv, "width", csv.width_column);
            add_csv_column(csv, "height", csv.height_column);
        }
    }
    for (auto& row : csv.rows) {
        if (static_cast<std::size_t>(std::stoull(row[csv.frame_column])) != index) continue;
        const auto& annotation = sequence.annotations[index];
        if (csv.exists_column >= 0) {
            row[csv.exists_column] = annotation.exists && !annotation.boxes.empty() ? "1" : "0";
        }
        if (primary) {
            if (annotation.boxes.empty()) {
                row[csv.class_column].clear();
                row[csv.x_column].clear();
                row[csv.y_column].clear();
                if (csv.width_column >= 0) row[csv.width_column].clear();
                if (csv.height_column >= 0) row[csv.height_column].clear();
                if (csv.x2_column >= 0) row[csv.x2_column].clear();
                if (csv.y2_column >= 0) row[csv.y2_column].clear();
            } else {
                const auto& box = annotation.boxes.front();
                row[csv.class_column] = std::to_string(box.class_id);
                row[csv.x_column] = std::to_string(box.x);
                row[csv.y_column] = std::to_string(box.y);
                if (csv.width_column >= 0) row[csv.width_column] = std::to_string(box.width);
                if (csv.height_column >= 0) row[csv.height_column] = std::to_string(box.height);
                if (csv.x2_column >= 0) row[csv.x2_column] = std::to_string(box.x + box.width);
                if (csv.y2_column >= 0) row[csv.y2_column] = std::to_string(box.y + box.height);
            }
        }
        break;
    }
    write_csv(csv);
}

void load_yolo(Sequence& sequence) {
    sequence.annotations.resize(sequence.frames.back().annotation_index + 1);
    for (const auto& frame : sequence.frames) {
        fs::path label = sequence.annotation_path / (frame.path.stem().string() + ".txt");
        if (!fs::is_regular_file(label)) continue;
        std::ifstream input(label);
        int class_id;
        double center_x, center_y, width, height;
        while (input >> class_id >> center_x >> center_y >> width >> height) {
            Box box;
            box.class_id = class_id;
            box.width = width * sequence.image_width;
            box.height = height * sequence.image_height;
            box.x = center_x * sequence.image_width - box.width / 2.0;
            box.y = center_y * sequence.image_height - box.height / 2.0;
            sequence.annotations[frame.annotation_index].boxes.push_back(box);
        }
        sequence.annotations[frame.annotation_index].exists =
            !sequence.annotations[frame.annotation_index].boxes.empty();
    }
}

void write_yolo(const Sequence& sequence, const FrameItem& frame) {
    fs::create_directories(sequence.annotation_path);
    const fs::path destination =
        sequence.annotation_path / (frame.path.stem().string() + ".txt");
    const fs::path temporary = destination.string() + ".tmp";
    std::ofstream output(temporary);
    if (!output) {
        throw std::runtime_error("Could not write YOLO label: " + temporary.string());
    }
    output << std::fixed << std::setprecision(8);
    for (const auto& box : sequence.annotations[frame.annotation_index].boxes) {
        const double center_x = (box.x + box.width / 2.0) / sequence.image_width;
        const double center_y = (box.y + box.height / 2.0) / sequence.image_height;
        output << box.class_id << ' ' << center_x << ' ' << center_y << ' '
               << box.width / sequence.image_width << ' '
               << box.height / sequence.image_height << '\n';
    }
    output.close();
    atomic_replace(temporary, destination);
}

std::vector<fs::path> discover_frames_directories(const fs::path& root) {
    std::vector<fs::path> directories;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (entry.is_directory() && lower(entry.path().filename().string()) == "frames") {
            directories.push_back(entry.path());
        }
    }
    std::sort(directories.begin(), directories.end());
    return directories;
}

Sequence load_sequence(const fs::path& frames_directory, Format requested_format) {
    Sequence sequence;
    sequence.frames_directory = frames_directory;
    sequence.directory = frames_directory.parent_path();
    std::vector<fs::path> image_paths;
    for (const auto& entry : fs::directory_iterator(frames_directory)) {
        if (entry.is_regular_file() && is_image(entry.path())) image_paths.push_back(entry.path());
    }
    std::sort(image_paths.begin(), image_paths.end(), [](const fs::path& left, const fs::path& right) {
        const auto left_number = frame_number(left);
        const auto right_number = frame_number(right);
        if (left_number && right_number && *left_number != *right_number) {
            return *left_number < *right_number;
        }
        return left.filename().string() < right.filename().string();
    });
    if (image_paths.empty()) return sequence;
    for (std::size_t position = 0; position < image_paths.size(); ++position) {
        sequence.frames.push_back({image_paths[position], frame_number(image_paths[position]).value_or(position)});
    }
    const cv::Mat first = cv::imread(sequence.frames.front().path.string(), cv::IMREAD_COLOR);
    if (first.empty()) {
        throw std::runtime_error("Could not read image: " + sequence.frames.front().path.string());
    }
    sequence.image_width = first.cols;
    sequence.image_height = first.rows;
    const std::size_t minimum_size = sequence.frames.back().annotation_index + 1;

    const auto json_path = exact_or_first(sequence.directory, ".json");
    const auto csv_path = exact_or_first(sequence.directory, ".csv");
    const fs::path labels_directory = sequence.directory / "labels";
    bool has_adjacent_yolo = false;
    for (const auto& frame : sequence.frames) {
        if (fs::is_regular_file(sequence.frames_directory /
                                (frame.path.stem().string() + ".txt"))) {
            has_adjacent_yolo = true;
            break;
        }
    }
    const bool has_yolo = fs::is_directory(labels_directory) || has_adjacent_yolo;
    if (requested_format == Format::Auto) {
        sequence.format = json_path ? Format::Json : (has_yolo ? Format::Yolo :
                          (csv_path ? Format::Csv : Format::Json));
    } else {
        sequence.format = requested_format;
    }

    if (csv_path) sequence.csv = load_csv(*csv_path);
    if (sequence.format == Format::Json) {
        sequence.annotation_path = json_path.value_or(
            sequence.directory / (sequence.directory.filename().string() + ".json"));
        sequence.annotations = json_path ? load_json(*json_path)
                                         : std::vector<FrameAnnotation>(minimum_size);
    } else if (sequence.format == Format::Csv) {
        sequence.annotation_path = csv_path.value_or(
            sequence.directory / (sequence.directory.filename().string() + ".csv"));
        if (!sequence.csv) {
            CsvData csv;
            csv.path = sequence.annotation_path;
            csv.headers = {"frame", "drone_exist", "class_id", "x", "y", "width", "height"};
            csv.frame_column = 0;
            csv.exists_column = 1;
            csv.class_column = 2;
            csv.x_column = 3;
            csv.y_column = 4;
            csv.width_column = 5;
            csv.height_column = 6;
            for (std::size_t index = 0; index < minimum_size; ++index) {
                csv.rows.push_back({std::to_string(index), "0", "", "", "", "", ""});
            }
            sequence.csv = std::move(csv);
        }
        sequence.annotations = load_csv_annotations(*sequence.csv, minimum_size);
    } else {
        sequence.annotation_path = fs::is_directory(labels_directory) || !has_adjacent_yolo
                                       ? labels_directory
                                       : sequence.frames_directory;
        load_yolo(sequence);
    }
    if (sequence.annotations.size() < minimum_size) {
        throw std::runtime_error("Annotations do not cover every frame in " +
                                 sequence.directory.string());
    }
    return sequence;
}

Format parse_format(const std::string& value) {
    const auto normalized = lower(value);
    if (normalized == "auto") return Format::Auto;
    if (normalized == "json") return Format::Json;
    if (normalized == "csv") return Format::Csv;
    if (normalized == "yolo" || normalized == "txt") return Format::Yolo;
    throw std::runtime_error("Unknown annotation format: " + value);
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) {
        throw std::runtime_error(
            "Usage: drone_annotation_editor ROOT [--format auto|json|csv|yolo] "
            "[--class-id N] [--max-width N] [--max-height N]");
    }
    Options options;
    options.root = fs::absolute(argv[1]);
    for (int index = 2; index < argc; ++index) {
        const std::string argument = argv[index];
        auto next = [&]() -> std::string {
            if (++index >= argc) throw std::runtime_error("Missing value after " + argument);
            return argv[index];
        };
        if (argument == "--format") options.format = parse_format(next());
        else if (argument == "--class-id") options.class_id = std::stoi(next());
        else if (argument == "--max-width") options.max_width = std::stoi(next());
        else if (argument == "--max-height") options.max_height = std::stoi(next());
        else if (argument == "--check") options.check_only = true;
        else throw std::runtime_error("Unknown argument: " + argument);
    }
    if (!fs::is_directory(options.root)) {
        throw std::runtime_error("Dataset root does not exist: " + options.root.string());
    }
    if (options.class_id < 0 || options.max_width <= 0 || options.max_height <= kStatusHeight) {
        throw std::runtime_error("Invalid class ID or window dimensions");
    }
    return options;
}

class Editor {
public:
    Editor(Options options, std::vector<Sequence> sequences)
        : options_(std::move(options)), sequences_(std::move(sequences)) {}

    void run() {
        cv::namedWindow(kWindowName, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(kWindowName, mouse_callback, this);
        render();
        while (true) {
            const int key = cv::waitKeyEx(0);
            if (key == 27 || key == 'q' || key == 'Q') break;
            message_.clear();
            dragging_ = false;
            if (key == 2 || key == 81 || key == 63234 || key == 65361 || key == 2424832 ||
                key == 'a' || key == 'A') move_frame(-1);
            else if (key == 3 || key == 83 || key == 63235 || key == 65363 ||
                     key == 2555904 || key == 'd' || key == 'D') move_frame(1);
            else if (key == 0 || key == 82 || key == 63232 || key == 65362 ||
                     key == 2490368 || key == 'w' || key == 'W') move_sequence(-1);
            else if (key == 1 || key == 84 || key == 63233 || key == 65364 ||
                     key == 2621440 || key == 's' || key == 'S') move_sequence(1);
            else if (key == ',' || key == '<') move_frame(-10);
            else if (key == '.' || key == '>') move_frame(10);
            else if (key == 'n' || key == 'N') next_unannotated();
            else if (key == 'c' || key == 'C') copy_previous();
            else if (key == 'x' || key == 'X' || key == 8 || key == 127) clear_current();
            else if (key == 'z' || key == 'Z') undo();
            else if (key == 'y' || key == 'Y') redo();
            else if (key == 'b' || key == 'B') boxes_visible_ = !boxes_visible_;
            else if (key == '+' || key == '=') zoom_by(1.5);
            else if (key == '-' || key == '_') zoom_by(1.0 / 1.5);
            else if (key == '0') {
                zoom_ = 1.0;
                view_center_ = {-1.0, -1.0};
            }
            render();
        }
        cv::destroyAllWindows();
    }

private:
    Options options_;
    std::vector<Sequence> sequences_;
    std::size_t sequence_index_ = 0;
    std::vector<std::size_t> positions_ = std::vector<std::size_t>(sequences_.size(), 0);
    std::vector<Change> undo_stack_;
    std::vector<Change> redo_stack_;
    cv::Mat canvas_;
    int display_width_ = 0;
    int display_height_ = 0;
    int original_width_ = 0;
    int original_height_ = 0;
    bool boxes_visible_ = true;
    bool dragging_ = false;
    cv::Point drag_start_;
    cv::Rect view_roi_;
    double zoom_ = 1.0;
    cv::Point2d view_center_{-1.0, -1.0};
    std::string message_;

    Sequence& sequence() { return sequences_[sequence_index_]; }
    FrameItem& frame() { return sequence().frames[positions_[sequence_index_]]; }
    FrameAnnotation& annotation() { return sequence().annotations[frame().annotation_index]; }

    static void mouse_callback(int event, int x, int y, int flags, void* data) {
        static_cast<Editor*>(data)->on_mouse(event, x, y, flags);
    }

    cv::Point to_original(int x, int y) const {
        x = std::clamp(x, 0, display_width_ - 1);
        y = std::clamp(y, 0, display_height_ - 1);
        return {
            view_roi_.x + static_cast<int>(std::round(
                              x * (view_roi_.width - 1.0) / std::max(1, display_width_ - 1))),
            view_roi_.y + static_cast<int>(std::round(
                              y * (view_roi_.height - 1.0) / std::max(1, display_height_ - 1)))};
    }

    void on_mouse(int event, int x, int y, int flags) {
        const bool inside = x >= 0 && y >= 0 && x < display_width_ && y < display_height_;
        if (event == cv::EVENT_MOUSEWHEEL) {
            const int direction = cv::getMouseWheelDelta(flags) > 0 ? 1 : -1;
            if (flags & cv::EVENT_FLAG_CTRLKEY) zoom_by(direction > 0 ? 1.5 : 1.0 / 1.5);
            else move_frame(direction > 0 ? -1 : 1);
            render();
        } else if (event == cv::EVENT_MBUTTONDOWN && inside) {
            view_center_ = to_original(x, y);
            message_ = "View recentered";
            render();
        } else if (event == cv::EVENT_LBUTTONDOWN && inside) {
            dragging_ = true;
            drag_start_ = {x, y};
        } else if (event == cv::EVENT_MOUSEMOVE && dragging_) {
            cv::Mat preview = canvas_.clone();
            cv::rectangle(preview, drag_start_,
                          {std::clamp(x, 0, display_width_ - 1),
                           std::clamp(y, 0, display_height_ - 1)},
                          {0, 220, 255}, 2);
            cv::imshow(kWindowName, preview);
        } else if (event == cv::EVENT_LBUTTONUP && dragging_) {
            dragging_ = false;
            if (!inside) {
                message_ = "Edit cancelled: release inside the frame";
                render();
                return;
            }
            const cv::Point first = to_original(drag_start_.x, drag_start_.y);
            const cv::Point second = to_original(x, y);
            const int left = std::min(first.x, second.x);
            const int top = std::min(first.y, second.y);
            const int width = std::abs(first.x - second.x);
            const int height = std::abs(first.y - second.y);
            if (width < 2 || height < 2) {
                message_ = "Edit cancelled: box is too small";
                render();
                return;
            }
            auto updated = annotation();
            Box box{options_.class_id, static_cast<double>(left), static_cast<double>(top),
                    static_cast<double>(width), static_cast<double>(height)};
            const bool add = (flags & cv::EVENT_FLAG_SHIFTKEY) && sequence().format == Format::Yolo;
            if (add) updated.boxes.push_back(box);
            else updated.boxes = {box};
            updated.exists = true;
            apply(updated, add ? "Added box" : "Saved box");
            render();
        } else if (event == cv::EVENT_RBUTTONDOWN && inside) {
            const cv::Point point = to_original(x, y);
            auto updated = annotation();
            const auto match = std::find_if(updated.boxes.begin(), updated.boxes.end(),
                                            [&](const Box& box) {
                return point.x >= box.x && point.y >= box.y &&
                       point.x <= box.x + box.width && point.y <= box.y + box.height;
            });
            if (match != updated.boxes.end()) {
                updated.boxes.erase(match);
                updated.exists = !updated.boxes.empty();
                apply(updated, "Deleted box");
                render();
            }
        }
    }

    void save_current() {
        auto& current = sequence();
        if (current.format == Format::Json) write_json(current);
        else if (current.format == Format::Csv) update_csv(*current.csv, current,
                                                           frame().annotation_index, true);
        else write_yolo(current, frame());
        if (current.format != Format::Csv && current.csv) {
            update_csv(*current.csv, current, frame().annotation_index, false);
        }
    }

    void apply(const FrameAnnotation& updated, const std::string& message) {
        const auto index = frame().annotation_index;
        undo_stack_.push_back({sequence_index_, index, annotation(), updated});
        redo_stack_.clear();
        annotation() = updated;
        save_current();
        message_ = message + " (autosaved)";
    }

    void clear_current() {
        FrameAnnotation empty;
        apply(empty, "Cleared frame");
    }

    void copy_previous() {
        if (positions_[sequence_index_] == 0) {
            message_ = "No previous frame";
            return;
        }
        const auto previous = sequence().frames[positions_[sequence_index_] - 1].annotation_index;
        apply(sequence().annotations[previous], "Copied previous box");
    }

    void undo() {
        if (undo_stack_.empty()) {
            message_ = "Nothing to undo";
            return;
        }
        const Change change = undo_stack_.back();
        undo_stack_.pop_back();
        sequence_index_ = change.sequence_index;
        auto& frames = sequence().frames;
        const auto position = std::find_if(frames.begin(), frames.end(), [&](const FrameItem& item) {
            return item.annotation_index == change.annotation_index;
        });
        if (position != frames.end()) positions_[sequence_index_] = position - frames.begin();
        sequence().annotations[change.annotation_index] = change.before;
        save_current();
        redo_stack_.push_back(change);
        message_ = "Undo autosaved";
    }

    void redo() {
        if (redo_stack_.empty()) {
            message_ = "Nothing to redo";
            return;
        }
        const Change change = redo_stack_.back();
        redo_stack_.pop_back();
        sequence_index_ = change.sequence_index;
        auto& frames = sequence().frames;
        const auto position = std::find_if(frames.begin(), frames.end(), [&](const FrameItem& item) {
            return item.annotation_index == change.annotation_index;
        });
        if (position != frames.end()) positions_[sequence_index_] = position - frames.begin();
        sequence().annotations[change.annotation_index] = change.after;
        save_current();
        undo_stack_.push_back(change);
        message_ = "Redo autosaved";
    }

    void move_frame(int offset) {
        auto& position = positions_[sequence_index_];
        const auto maximum = static_cast<long long>(sequence().frames.size() - 1);
        position = static_cast<std::size_t>(std::clamp(
            static_cast<long long>(position) + offset, 0LL, maximum));
    }

    void move_sequence(int offset) {
        sequence_index_ = static_cast<std::size_t>(std::clamp(
            static_cast<long long>(sequence_index_) + offset, 0LL,
            static_cast<long long>(sequences_.size() - 1)));
        zoom_ = 1.0;
        view_center_ = {-1.0, -1.0};
    }

    void zoom_by(double factor) {
        zoom_ = std::clamp(zoom_ * factor, 1.0, 8.0);
        if (view_center_.x < 0 || view_center_.y < 0) {
            view_center_ = {original_width_ / 2.0, original_height_ / 2.0};
        }
        std::ostringstream text;
        text << std::fixed << std::setprecision(1) << "Zoom " << zoom_ << "x";
        message_ = text.str();
    }

    void next_unannotated() {
        auto& position = positions_[sequence_index_];
        for (std::size_t candidate = position + 1; candidate < sequence().frames.size(); ++candidate) {
            const auto index = sequence().frames[candidate].annotation_index;
            if (!sequence().annotations[index].exists || sequence().annotations[index].boxes.empty()) {
                position = candidate;
                message_ = "Next unannotated frame";
                return;
            }
        }
        message_ = "No later unannotated frame in this sequence";
    }

    void render() {
        cv::Mat image = cv::imread(frame().path.string(), cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Could not read image: " + frame().path.string());
        original_width_ = image.cols;
        original_height_ = image.rows;
        if (original_width_ != sequence().image_width || original_height_ != sequence().image_height) {
            throw std::runtime_error("Image dimensions changed within sequence: " + frame().path.string());
        }
        if (boxes_visible_) {
            for (const auto& box : annotation().boxes) {
                const cv::Rect rectangle(
                    static_cast<int>(std::round(box.x)), static_cast<int>(std::round(box.y)),
                    static_cast<int>(std::round(box.width)), static_cast<int>(std::round(box.height)));
                cv::rectangle(image, rectangle, {0, 255, 0}, 2);
                cv::putText(image, "class " + std::to_string(box.class_id),
                            {rectangle.x, std::max(18, rectangle.y - 6)},
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 255, 0}, 2, cv::LINE_AA);
            }
        }
        if (view_center_.x < 0 || view_center_.y < 0) {
            view_center_ = {original_width_ / 2.0, original_height_ / 2.0};
        }
        const int roi_width = std::max(2, static_cast<int>(std::round(original_width_ / zoom_)));
        const int roi_height = std::max(2, static_cast<int>(std::round(original_height_ / zoom_)));
        const int roi_x = std::clamp(static_cast<int>(std::round(view_center_.x - roi_width / 2.0)),
                                     0, original_width_ - roi_width);
        const int roi_y = std::clamp(static_cast<int>(std::round(view_center_.y - roi_height / 2.0)),
                                     0, original_height_ - roi_height);
        view_roi_ = {roi_x, roi_y, roi_width, roi_height};
        view_center_ = {roi_x + roi_width / 2.0, roi_y + roi_height / 2.0};

        const double base_scale = std::min(
            {1.0, options_.max_width / static_cast<double>(original_width_),
             (options_.max_height - kStatusHeight) / static_cast<double>(original_height_)});
        display_width_ = std::max(2, static_cast<int>(std::round(original_width_ * base_scale)));
        display_height_ = std::max(2, static_cast<int>(std::round(original_height_ * base_scale)));
        cv::Mat view = image(view_roi_).clone();
        if (view.cols != display_width_ || view.rows != display_height_) {
            cv::resize(view, view, {display_width_, display_height_}, 0, 0,
                       zoom_ > 1.0 ? cv::INTER_LINEAR : cv::INTER_AREA);
        }
        cv::copyMakeBorder(view, canvas_, 0, kStatusHeight, 0, 0,
                           cv::BORDER_CONSTANT, {22, 22, 22});
        const int y = display_height_ + 25;
        const auto& current = sequence();
        const std::string format = current.format == Format::Json ? "JSON" :
                                   current.format == Format::Csv ? "CSV" : "YOLO TXT";
        std::vector<std::string> lines = {
            "Folder " + std::to_string(sequence_index_ + 1) + "/" +
                std::to_string(sequences_.size()) + " | Frame " +
                std::to_string(positions_[sequence_index_] + 1) + "/" +
                std::to_string(current.frames.size()) + " | boxes=" +
                std::to_string(annotation().boxes.size()) + " | " + format + " | zoom=" +
                std::to_string(zoom_).substr(0, 3) + "x",
            current.directory.filename().string() + " / " + frame().path.filename().string()};
        if (message_.empty()) {
            lines.push_back("Drag: replace | Shift-drag: add YOLO | Right-click: delete");
            lines.push_back("Wheel/arrows: move | Ctrl-wheel: zoom | Middle-click: recenter");
            lines.push_back("N: next empty | C: copy | X: clear | Z/Y: undo/redo | Q: quit");
        } else {
            lines.push_back(message_);
        }
        for (std::size_t index = 0; index < lines.size(); ++index) {
            const cv::Scalar color = !message_.empty() && index == lines.size() - 1
                                         ? cv::Scalar(80, 220, 255)
                                         : cv::Scalar(235, 235, 235);
            cv::putText(canvas_, lines[index], {12, y + static_cast<int>(index) * 27},
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        }
        cv::imshow(kWindowName, canvas_);
    }
};

int main(int argc, char** argv) {
    try {
        if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            std::cout
                << "Usage: drone_annotation_editor ROOT [--format auto|json|csv|yolo]\n"
                << "       [--class-id N] [--max-width N] [--max-height N] [--check]\n";
            return 0;
        }
        const Options options = parse_options(argc, argv);
        std::vector<Sequence> sequences;
        for (const auto& directory : discover_frames_directories(options.root)) {
            Sequence sequence = load_sequence(directory, options.format);
            if (!sequence.frames.empty()) sequences.push_back(std::move(sequence));
        }
        if (sequences.empty()) {
            throw std::runtime_error("No frames directories found under " + options.root.string());
        }
        std::size_t frame_count = 0;
        for (const auto& sequence : sequences) frame_count += sequence.frames.size();
        std::cout << "Loaded " << sequences.size() << " sequences and " << frame_count
                  << " frames.\n";
        if (options.check_only) {
            std::cout << "Annotation workspace check passed.\n";
            return 0;
        }
        std::cout << "Changes autosave; Q exits.\n";
        Editor(options, std::move(sequences)).run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
