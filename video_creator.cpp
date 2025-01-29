#include <cstdint>
#include <iostream>
#include "timeline.h"
#include <opencv4/opencv2/opencv.hpp>
#include <ostream>

const int CELL_SIZE = 2;
cv::Scalar stateToColor(uint8_t state)
{
	switch (state) {
	case 0:
		return cv::Scalar(200, 200, 200, 255);
		break;
	case 1:
		return cv::Scalar(0, 255, 0, 255);
		break;
	case 2:
		return cv::Scalar(0, 0, 255, 255);
		break;
	case 3:
		return cv::Scalar(0, 0, 0, 255);
		break;
	case 4:
		return cv::Scalar(255, 0, 0, 255);
		break;
	default:
		return cv::Scalar(255, 255, 255, 255);
		break;
	}
}

cv::Mat createFrame(uint8_t *grid, size_t w, size_t h, int cellSize)
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

int main(int argc, char **argv)
{
	if (argc < 3) {
		std::cerr << "Usage " << argv[0]
			  << " <timeline_file.bin> <output.avi>\n";
		return EXIT_FAILURE;
	}

	timeline_t tl;

	timeline_error_t err = timeline_load(&tl, argv[1]);
	if (err != TL_OK) {
		std::cerr << "Failed to Load Timeline " << argv[1] << '\n';
		return EXIT_FAILURE;
	}

	std::cout << "------------------------------------\n";
	std::cout << "Parameters\n";
	std::cout << "------------------------------------\n";
	std::cout
		<< "Population                  : "
		<< tl.params.populationPercent << "%\n"
		<< "World height                : " << tl.params.worldHeight
		<< "\n"
		<< "World Width                 : " << tl.params.worldWidth
		<< "\n"
		<< "World size                  : "
		<< tl.params.worldHeight * tl.params.worldWidth << "\n"
		<< "Proximity                   : " << tl.params.proximity
		<< "\n"
		<< "Infection duration          : "
		<< tl.params.infectionDuration << " turns\n"
		<< "Healthy infection probability:"
		<< tl.params.healthyInfectionProbability << " % \n"
		<< "Immune infection probability: "
		<< tl.params.immuneInfectionProbability << " % \n"
		<< "Death probability           : "
		<< tl.params.deathProbability << "%\n"
		<< "Initial infected            : " << tl.params.initialInfected
		<< "\n"
		<< "Population immunized        : " << tl.params.initialImmune
		<< "%\n"
		<< "Number of Rounds            : " << tl.saved_rounds << '\n';
	std::cout << "\n";

	uint8_t *grid =
		new uint8_t[tl.params.worldHeight * tl.params.worldWidth];

	cv::Size videoSize{ (int)tl.params.worldWidth * CELL_SIZE,
			    (int)tl.params.worldHeight * CELL_SIZE };

	cv::VideoWriter writer(argv[2],
			       cv::VideoWriter::fourcc('H', '2', '6', '4'), 60,
			       videoSize);

	if (!writer.isOpened()) {
		return -1;
	}

	for (size_t i = 0; i < tl.saved_rounds; ++i) {
		err = timeline_get_round(&tl, grid);
		if (err == TL_END) {
			break;
		} else if (err != TL_OK) {
			std::cerr << "Failed to Read Round\n";
			break;
		}
		cv::Mat frame = createFrame(grid, tl.params.worldWidth,
					    tl.params.worldHeight, CELL_SIZE);
		writer.write(frame);
	}

	writer.release();
	timeline_close(&tl);
	delete[] grid;
	return 0;
}
