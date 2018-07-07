#include <iostream>
#include "tools.h"
#include "functions.h"

int main()
{
	Image img("test.png");
	Image new_img = to_edges(img);
	std::cout<<std::hex;
	/*
	for(int k = 0; k<img.num_channels(); k++)
	{
		for(int i = 0; i<img.height(); i++)
		{
			for(int j = 0; j<img.width(); j++)
			{
				std::cout << (int)img.data()[k*img.width()*img.height()+i*img.width()+j] << " ";
			}
			std::cout << std::endl;
		}

		std::cout << std::endl;
	}//*/
	/*
	for(int k = 0; k<new_img.num_channels(); k++)
	{
		for(int i = 0; i<new_img.height(); i++)
		{
			for(int j = 0; j<new_img.width(); j++)
			{
				std::cout << (int)new_img.data()[k*new_img.width()*new_img.height()+i*new_img.width()+j] << " ";
			}
			std::cout << std::endl;
		}

		std::cout << std::endl;
	}//*/
	std::cout << std::dec;
	std::cout << new_img.width() << " " << new_img.height() << " " << new_img.num_channels() << std::endl;
	new_img.save("grey.png");
	return 0;
}
