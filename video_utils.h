#ifndef VIDEO_UTILS_H
#define VIDEO_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

///@brief Creates a video from a 3D grid.
///@param grid3D The 3D grid containing simulation steps.
///@param outputPath The file path to save the video.
///@param cellSize The size of each cell in pixels.
///@param fps Frames per second for the video.
///@throws std::invalid_argument if the grid3D is empty or invalid.
///@throws std::runtime_error if the video file cannot be opened.
void createVideo(const std::vector<std::vector<std::vector<int> > > &grid3D,
		 const std::string &outputPath, int cellSize = 20,
		 int fps = 10);

#endif // VIDEO_UTILS_H
