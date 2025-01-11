#ifndef VIDEO_UTILS_H
#define VIDEO_UTILS_H

#include <stddef.h>

///@brief Creates a video from a 3D grid.
///@param rounds a pointer containing simulation steps.
///	  Each pointer points to w * h grid
///@param nb_rounds The number of rounds
///@param w the grid width
///@param h the grid height
///@param path The file path to save the video.
///@param cellSize The size of each cell in pixels.
///@param fps Frames per second for the video.
int create_video(int **rounds, size_t nb_rounds, size_t w, size_t h,
		 const char *path, int cellSize = 20, int fps = 10);

#endif // VIDEO_UTILS_H
