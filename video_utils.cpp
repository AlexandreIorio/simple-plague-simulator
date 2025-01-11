#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include "video_utils.h"

cv::Scalar stateToColor(int state)
{
	switch (state) {
	case 0:
		return cv::Scalar(255, 255, 255); // Blanc
	case 1:
		return cv::Scalar(0, 128, 0); // Vert
	case 2:
		return cv::Scalar(0, 0, 128); // Rouge
	case 3:
		return cv::Scalar(0, 0, 0); // Noir
	case 4:
		return cv::Scalar(128, 128, 0);
	default:
		return cv::Scalar(255, 255, 255); // Par défaut blanc
	}
}

cv::Mat createFrame(int *grid, size_t w, size_t h, int cellSize)
{
	cv::Mat frame(h * cellSize, w * cellSize, CV_8UC3,
		      cv::Scalar(255, 255, 255));

	for (size_t i = 0; i < h; ++i) {
		for (size_t j = 0; j < w; ++j) {
			cv::Point center(j * cellSize + cellSize / 2,
					 i * cellSize + cellSize / 2);
			int radius = cellSize / 2;
			cv::Scalar color = stateToColor(grid[i * w + j]);
			cv::circle(frame, center, radius, color, cv::FILLED);
		}
	}

	return frame;
}

int create_video(int **grids, size_t rounds, size_t w, size_t h,
		 const char *outputPath, int cellSize, int fps)
{
	if (grids == NULL) {
		return -1;
	}

	cv::Size videoSize(w * cellSize, h * cellSize);

	cv::VideoWriter writer(outputPath,
			       cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
			       videoSize);
	if (!writer.isOpened()) {
		return -1;
	}

	// Générer chaque frame
	for (size_t i = 0; i < rounds; ++i) {
		cv::Mat frame = createFrame(grids[i], w, h, cellSize);
		writer.write(frame);
	}

	writer.release();
	return 0;
}
