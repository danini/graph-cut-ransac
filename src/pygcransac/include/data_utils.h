#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>
#include <filesystem>
#include "dirent.h"

namespace fs = std::filesystem;

namespace datareading
{
	template <typename _Type>
	bool writeToText(
		const std::string &filename_,
		const cv::Mat &data_)
	{
		std::ofstream file(filename_);
		if (!file.is_open())
			return false;

		file << data_.rows << " " << data_.cols << std::endl;

		for (size_t row = 0; row < data_.rows; ++row)	
		{
			for (size_t col = 0; col < data_.cols; ++col)	
				file << data_.at<_Type>(row, col) << " ";
			file << std::endl;
		}
	}
	
	template <typename _Type>
	bool readFromText(
		const std::string &filename_,
		const int &dataType_,
		cv::Mat &data_)
	{
		std::ifstream file(filename_);
		if (!file.is_open())
			return false;

		int rows, cols;
		file >> rows >> cols;
		data_.create(rows, cols, dataType_);

		for (size_t row = 0; row < rows; ++row)	
			for (size_t col = 0; col < cols; ++col)	
				file >> data_.at<_Type>(row, col);
	}

	bool writeResults(
		const std::string &filename_,
		const cv::Mat &data_,
		const std::string &label_)
	{
		static std::mutex saving_mutex;
		saving_mutex.lock();
		cv::Ptr<cv::hdf::HDF5> h5io;

		h5io = cv::hdf::open(filename_);

		if (h5io.empty())
		{
			h5io->close();
			saving_mutex.unlock();
			return false;
		}

		h5io->dswrite(data_, label_);
		h5io->close();
		saving_mutex.unlock();
		return true;
	}

	bool openDatabase(
		const std::string &filename_,
		cv::Ptr<cv::hdf::HDF5> &h5io_)
	{
		h5io_ = cv::hdf::open(filename_);

		if (h5io_.empty())
			return false;
		return true;
	}

	bool readData(
		const std::string &filename_,
		const std::string &label_,
		cv::Mat &data_)
	{
		static std::mutex reading_mutex;
		reading_mutex.lock();

		// Check if file exists first
		std::ifstream ifile;
		ifile.open(filename_);
		if (!ifile.is_open())
		{
			reading_mutex.unlock();
			return false;
		}
		ifile.close();

		cv::Ptr<cv::hdf::HDF5> h5io = cv::hdf::open(filename_);

		if (h5io.empty())
		{
			reading_mutex.unlock();
			return false;
		}

		if (!h5io->hlexists(label_))
		{
			reading_mutex.unlock();
			return false;
		}

		h5io->dsread(data_, label_);
		h5io->close();
		reading_mutex.unlock();
		return true;
	}

	bool readData(
		const std::string &filename_,
		const std::string &label_,
		const cv::Ptr<cv::hdf::HDF5> h5io_,
		cv::Mat &data_)
	{
		static std::mutex reading_mutex;
		reading_mutex.lock();

		if (h5io_.empty())
		{
			reading_mutex.unlock();
			return false;
		}

		if (!h5io_->hlexists(label_))
		{
			reading_mutex.unlock();
			return false;
		}

		h5io_->dsread(data_, label_);
		data_.convertTo(data_, CV_64F);
		reading_mutex.unlock();
		return true;
	}

	std::vector<std::string> readFilesInFolder(
		const std::string &path_)
	{		
		std::vector<std::string> files;
		DIR *dir;
		struct dirent *ent;
		if ((dir = opendir(path_.c_str())) != NULL) {
			/* print all the files and directories within directory */
			while ((ent = readdir (dir)) != NULL) {
				std::istringstream tokenStream(ent->d_name);
				std::string token;
				while (std::getline(tokenStream, token, '\\'));
				files.emplace_back(token);
			}
			closedir (dir);
		} else {
		}
		/*std::cout << path_ << std::endl;
		std::vector<std::string> files;
		for (const auto & entry : fs::directory_iterator(path_))
		{
			std::cout << entry.path().string() << std::endl;
			if (!entry.is_directory())
			{
				std::istringstream tokenStream(entry.path().string());
				std::string token;
				while (std::getline(tokenStream, token, '\\'));
				files.emplace_back(token);
			}
		}*/
		return files;
	}
}