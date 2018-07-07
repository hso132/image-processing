#include "tools.h"

#include "include/CImg.h"
#include <iostream>

Image::Image(const char* const filepath)
{
	cimg_library::CImg<byte_t> img(filepath);
	m_height = img.height();
	m_width = img.width();
	m_num_channels = img.spectrum();
	m_data = new byte_t[m_height*m_width*m_num_channels];
	std::copy_n(img.data(), m_height*m_width*m_num_channels, m_data);
}

Image::~Image()
{
	delete m_data;
}


std::size_t Image::num_channels() const 
{
	return m_num_channels;
}

std::size_t Image::width() const
{
	return m_width;
}

std::size_t Image::height() const
{
	return m_height;
}

const byte_t* Image::data() const
{
	return m_data;
}

void Image::save(const char* const filepath)
{
	cimg_library::CImg<byte_t> img(m_data, m_width, m_height, 1, m_num_channels);
	img.save_png(filepath);
}

Image::Image(byte_t* data, const std::size_t width, const size_t height, const size_t channels)
{
	m_data = data;
	m_height = height;
	m_width = width;
	m_num_channels = channels;
}
