//#undef slots
#include <torch/torch.h>
#include <torch/script.h> 
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp> 
#include <stdio.h> 
#include <filesystem>
namespace fs = std::filesystem; 
int main() {
    
    std::string mode = "test dataset"; // "test dataset" or "single image"
    std::string file_path = "../class_list.txt"; 
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return 1;
    }

    std::vector<std::string> class_names;
    //std::unordered_map<std::string, int> class_labels;

    std::string class_name;
    int index = 0;
    while (std::getline(file, class_name)) {
        class_names.push_back(class_name);
    }
    
    
    // Load the TorchScript model
    torch::jit::script::Module module;
    module = torch::jit::load("../models/torch_script_model.pt");

    if(mode == "single image") {
        
        //std::string filename = "../testimages/cricket.jpg";
        std::string filename = "../testimages/Swimming.jpeg";

        // Read the image
        cv::Mat img = cv::imread(filename);

        // Resize the image
        cv::resize(img, img, cv::Size(224, 224));

        // Convert the image from BGR to RGB
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        // Convert the image to float and normalize it
        img.convertTo(img, CV_32F, 1.0 / 255);

        // Perform inference and print the predicted label
        // Convert the image to a tensor
        torch::Tensor input_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}).permute({2, 0, 1});

        // Add a batch dimension
        input_tensor.unsqueeze_(0);

        // Perform inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor); 
        
        auto output = module.forward(inputs).toGenericDict();
        
        auto output_tensor = output.at("logits").toTensor();
        
        // Get the predicted label
        auto predicted_label = output_tensor.argmax(1);
        auto predicted_class = class_names[predicted_label.item<int>()];
        std::cout <<"Image File: " <<filename << " -> Predicted label: " << predicted_class<< std::endl;
    }

    else if(mode == "test dataset") {
        int correct = 0;
        int total = 0;
        
        //test folder path
        std::string folder_path = "../dataset/sportsdata/test";
        std:: cout<<"Processing images and predicting classes.....\n"<< std::endl;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        

        if (entry.is_directory()) {
            std::string class_name = entry.path().filename().string();
            for (const auto& img_entry : fs::directory_iterator(entry.path())) {
                if (!img_entry.is_directory()) {
                    std::string img_path = img_entry.path().string();
                    // Read the image using OpenCV
                    cv::Mat img = cv::imread(img_path);
                    if (img.empty()) {
                        std::cerr << "Failed to read image: " << img_path << std::endl;
                        continue;
                    }

                    cv::resize(img, img, cv::Size(224, 224));

                    // Convert the image from BGR to RGB
                    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

                    // Convert the image to float and normalize it
                    img.convertTo(img, CV_32F, 1.0 / 255);

                    torch::Tensor input_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}).permute({2, 0, 1});

                    // Add a batch dimension
                    input_tensor.unsqueeze_(0);
                    std::vector<torch::jit::IValue> inputs;
                    // Perform inference
                    inputs.push_back(input_tensor); 
                    
                    auto output = module.forward(inputs).toGenericDict();
                    
                    auto output_tensor = output.at("logits").toTensor();
                    
                    // Get the predicted label
                    auto predicted_label = output_tensor.argmax(1);
                    auto predicted_class = class_names[predicted_label.item<int>()];
                    if(predicted_class == class_name) {
                        correct++;
                    }
                    total++;
                
                }
            }
        }
    }
    std::cout << "Total Test Images: " << total << std::endl;
    std::cout << "Correct Predictions: " << correct << std::endl;
    std::cout << "Incorrect Predictions: " << total - correct << std::endl;
    std::cout <<std::endl;
    float accu = (static_cast<float>(correct) / total) * 100;
    std::cout << "Test Accuracy: " << accu << "%" << std::endl;
        
}
 return 0;

}   