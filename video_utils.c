#include "video_utils.h"

cv::Scalar stateToColor(int state)
{
	switch (state) {
	case 0:
		return cv::Scalar(255, 255, 255); // Blanc
	case 1:
		return cv::Scalar(0, 255, 0); // Vert
	case 2:
		return cv::Scalar(0, 0, 255); // Rouge
	case 3:
		return cv::Scalar(0, 0, 0); // Noir
	case 4:
		return cv::Scalar(255, 255, 0); // Jaune
	default:
		return cv::Scalar(255, 255, 255); // Par défaut blanc
	}
}

cv::Mat createFrame(const std::vector<std::vector<int> > &grid, int cellSize)
{
	int rows = grid.size();
	int cols = grid[0].size();

	// Créer une image de taille adaptée
	cv::Mat frame(rows * cellSize, cols * cellSize, CV_8UC3,
		      cv::Scalar(255, 255, 255));

	// Remplir les cellules avec les couleurs correspondantes
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			cv::Rect cellRect(j * cellSize, i * cellSize, cellSize,
					  cellSize);
			cv::Scalar color = stateToColor(grid[i][j]);
			cv::rectangle(frame, cellRect, color, cv::FILLED);
		}
	}

	return frame;
}

void createVideo(const std::vector<std::vector<std::vector<int> > > &grid3D,
		 const std::string &outputPath, int cellSize, int fps)
{
	if (grid3D.empty() || grid3D[0].empty() || grid3D[0][0].empty()) {
		throw std::invalid_argument("Grid 3D is empty or invalid");
	}

	int rows = grid3D[0].size();
	int cols = grid3D[0][0].size();

	// Définir les dimensions de la vidéo
	cv::Size videoSize(cols * cellSize, rows * cellSize);

	// Initialiser le writer vidéo
	cv::VideoWriter writer(outputPath,
			       cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
			       videoSize);
	if (!writer.isOpened()) {
		throw std::runtime_error(
			"Could not open the video file for writing");
	}

	// Générer chaque frame
	for (const auto &grid : grid3D) {
		cv::Mat frame = createFrame(grid, cellSize);
		writer.write(frame);
	}

	writer.release();
}