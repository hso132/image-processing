#ifndef TOOLS_H_HEADER
#define TOOLS_H_HEADER
#include <cstddef>
typedef unsigned char byte_t;
class Image
{
	public:
	Image(const char* const filepath);
	Image(byte_t* data, const size_t width, const size_t height, const size_t channels);
	~Image();
	
	const byte_t* data() const;
	size_t height() const;
	size_t width() const;
	size_t num_channels() const;

	void save(const char* const filepath);

	protected:

	byte_t* m_data;
	size_t m_height;
	size_t m_width;
	size_t m_num_channels;

};
#endif
