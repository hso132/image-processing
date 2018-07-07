#include "functions.h"
#include "functions.hu"
#include <iostream>
#include <cassert>


__global__ void non_max_suppress(const float* G, const float* angles, const size_t width, const size_t height, float* result)
{
	size_t x = blockDim.x*blockIdx.x+threadIdx.x;
	size_t y = blockDim.y*blockIdx.y+threadIdx.y;
	size_t idx = y*width+x;
	if(x<width && y<height)
	{
		//first point is always to the left
		int64_t x1, y1;
		int64_t x2, y2;
		//convert the radian angle into degrees
		double angle = int(360*(angles[idx]/(2*M_PI)));
		//make the angle positive, and in [0,180)
		angle = fmod(angle+360, 180.0);

		//the gradient is East - West
		if(angle >= 157.5 || angle < 22.5)
		{
			//check left and right
			x1 = int64_t(x)-1;
			x2 = int64_t(x)+1;
			y1 = y;
			y2 = y;
		}
		//the gradient is South-East - North-West
		else if(angle >= 22.5 && angle < 67.5)
		{
			//check top right and bottom left
			x1= int64_t(x)-1;
			x2= int64_t(x)+1;
			y1= int64_t(y)+1;
			y2= int64_t(y)-1;
		}
		//the gradient is North - South
		else if(angle >= 67.5 && angle < 112.5)
		{
			//check top and bottom
			x1 = x;
			x2 = x;
			y1 = int64_t(y)-1;
			y2 = int64_t(y)+1;
		}
		//the gradient is South-West - North-East
		else
		{
			//check bottom right and top left
			x1= int64_t(x)-1;
			x2= int64_t(x)+1;
			y1= int64_t(y)-1;
			y2= int64_t(y)+1;
		}

		float str1;
		float str2;
		//if the points are within the image, we consider them, otherwise we ignore them
		if(x1 < width && x1 >= 0 && y1 < height && y1 >= 0)
		{
			str1=G[x1+width*y1];
		}
		else
		{
			str1 = 0;
		}
		if(x2 < width && x2 >= 0 && y2 < height && y2 >= 0)
		{
			str2=G[x2+width*y2];
		}
		else
		{
			str2=0;
		}
		float str3 = G[idx];
		//if str3 is a local maximum
		if(str3 > str1 && str3 > str2)
		{
			result[idx] = str3;
		}
		else
		{
			result[idx] = 0;
		}
	}
	
}
__global__ void apply_dual_threshold(const float* im, size_t width, size_t height, float T1, float T2, float* out)
{
	size_t x = blockDim.x*blockIdx.x+threadIdx.x;
	size_t y = blockDim.y*blockIdx.y+threadIdx.y;
	size_t idx = y*width+x;
	if(x<width && y<height)
	{
		if(im[idx] >= T1 && im[idx] <= T2)
		{
			out[idx] = im[idx];
		}
		else
		{
			out[idx] = 0;
		}
	}
}
__global__ void compute_theta(const float* G_x, const float* G_y, size_t width, size_t height, float* theta)
{
	size_t x = blockDim.x*blockIdx.x+threadIdx.x;
	size_t y = blockDim.y*blockIdx.y+threadIdx.y;
	size_t idx = y*width+x;
	if(x<width && y<height)
	{
		theta[idx] = atan2(G_y[idx], G_x[idx]);
	}
}
__global__ void compute_G(const float* G_x, const float* G_y, size_t width, size_t height, float* G)
{
	size_t x = blockDim.x*blockIdx.x+threadIdx.x;
	size_t y = blockDim.y*blockIdx.y+threadIdx.y;
	size_t idx = x+width*y;
	if(x<width && y<height)
	{
		G[idx]= sqrt((G_x[idx]*G_x[idx])+(G_y[idx]*G_y[idx]));
	}
}
void generate_gauss_kernel(float* dest, size_t size, float sigma)
{
	const double sigma_squared = sigma*sigma;
	assert(size%2==1);
	size_t k = size/2;
	for(int i = 0; i<size; i++)
	{
		for(int j = 0; j<size; j++)
		{
			int i_0 = i-k;
			int j_0 = j-k;
			double arg = -(i_0*i_0 + j_0*j_0)/(2*sigma_squared);
			double result = std::exp(arg)/(2*M_PI*sigma_squared);
			dest[i*size+j]=result;
		}
	}
}
const int MAX_THREADS_PER_SIDE = 32;
__global__ void greyscale(const byte_t* channels, const size_t width, const size_t height, float* output)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int idx = x+width*y;
	int channel_width = width*height;

	if(x<width && y < height)
	{
		float avg = (channels[idx] + channels[idx+channel_width] + channels[idx+channel_width+channel_width])/768.0f;
		output[idx] = avg;
	}
	
}
Image to_edges(const Image& rgb)
{
	//makes more sense
	assert(rgb.num_channels() == 3);

	//some copies; better performance
	auto height = rgb.height();
	auto width = rgb.width();
	auto num_channels = rgb.num_channels();
	auto channel_width = height*width;


	//allocate memory on gpu
	byte_t* d_channels;
	auto err = cudaMalloc(&d_channels, sizeof(byte_t)*channel_width*num_channels);
	handle_cuda_error(err);

	//copy memory to gpu
	err = cudaMemcpy(d_channels, rgb.data(), channel_width*num_channels*sizeof(byte_t), cudaMemcpyHostToDevice);
	handle_cuda_error(err);

	//allocate output
	float* d_float_greyscale;
	err = cudaMalloc(&d_float_greyscale, sizeof(float)*channel_width);
	handle_cuda_error(err);

	
	//launch threads
	dim3 num_blocks = get_optimal_blocks(width, height);

	dim3 num_threads{MAX_THREADS_PER_SIDE, MAX_THREADS_PER_SIDE};
	greyscale<<<num_blocks, num_threads>>>(d_channels, width, height, d_float_greyscale);


	size_t blur_side = 9;
	size_t blur_size = blur_side*blur_side;
	float h_blur[blur_size];
	
	float sigma = 1.2;
	generate_gauss_kernel(h_blur, blur_side, sigma);

	float *d_blur;
	err = cudaMalloc(&d_blur, blur_size*sizeof(float));
	handle_cuda_error(err);
	cudaMemcpy(d_blur, h_blur, blur_size*sizeof(float), cudaMemcpyHostToDevice);

	float *d_blurred;
	err=cudaMalloc(&d_blurred, channel_width*sizeof(float));
	handle_cuda_error(err);
	//apply gaussian kernel
	apply_kernel<<<num_blocks, num_threads>>>(d_float_greyscale, width, height, d_blur, blur_side, blur_side, d_blurred);

	const size_t sobel_side = 3;
	const size_t sobel_size = sobel_side*sobel_side;
	
	float h_hor_sobel[]={	1,0,-1,
							2,0,-2,
							1,0,-1,};
						
	float h_vert_sobel[]={	1,2,1,
							0,0,0,
							-1,-2,-1};



	float *d_hor_sobel;
	err = cudaMalloc(&d_hor_sobel, sobel_size*sizeof(float));
	handle_cuda_error(err);
	cudaMemcpy(d_hor_sobel, h_hor_sobel, sobel_size*sizeof(float), cudaMemcpyHostToDevice);

	float *d_vert_sobel;
	err = cudaMalloc(&d_vert_sobel, sobel_size*sizeof(float));
	handle_cuda_error(err);
	cudaMemcpy(d_vert_sobel, h_vert_sobel, sobel_size*sizeof(float), cudaMemcpyHostToDevice);



	float* d_Gx;
	err = cudaMalloc(&d_Gx, channel_width*sizeof(float));
	handle_cuda_error(err);

	//compute G_x
	apply_kernel<<<num_blocks, num_threads>>>(d_blurred, width, height, d_hor_sobel, sobel_side, sobel_side, d_Gx);

	float* d_Gy;
	err = cudaMalloc(&d_Gy, channel_width*sizeof(float));
	handle_cuda_error(err);

	//compute G_y
	apply_kernel<<<num_blocks, num_threads>>>(d_blurred, width, height, d_vert_sobel, sobel_side, sobel_side, d_Gy);




	float* d_G;
	err = cudaMalloc(&d_G, channel_width*sizeof(float));
	handle_cuda_error(err);
	
	//compute G as the hypotenuse
	compute_G<<<num_blocks, num_threads>>>(d_Gx, d_Gy, width, height, d_G);

	float* d_theta;
	err = cudaMalloc(&d_theta, channel_width*sizeof(float));
	handle_cuda_error(err);

	//compute the angle of the gradient
	compute_theta<<<num_blocks, num_threads>>>(d_Gx, d_Gy, width, height, d_theta);


	float* d_non_max_supp;
	err = cudaMalloc(&d_non_max_supp, channel_width*sizeof(float));
	handle_cuda_error(err);
	non_max_suppress<<<num_blocks, num_threads>>>(d_G, d_theta, width, height, d_non_max_supp);


	float* d_thresholded;
	err = cudaMalloc(&d_thresholded, channel_width*sizeof(float));
	handle_cuda_error(err);

	apply_dual_threshold<<<num_blocks, num_threads>>>(d_non_max_supp, width, height, 0.4, 2, d_thresholded);
	

	/* debugger
	float* h_G = new float[channel_width];
	cudaMemcpy(h_G, d_non_max_supp, channel_width*sizeof(float), cudaMemcpyDeviceToHost);
	for(int64_t i=0; i<height; i++)
	{
		for(int64_t j=0; j<width; j++)
		{
			std::cout << h_G[i*width+j] << " ";
		}
		std::cout << std::endl;
	}
	delete[] h_G;
	//*/

	byte_t* d_byte_out;
	err = cudaMalloc(&d_byte_out, channel_width*sizeof(byte_t));
	handle_cuda_error(err);

	//convert to bytes
	float_to_byte<<<num_blocks, num_threads>>>(d_thresholded, width, height, d_byte_out);

	//copy to out
	byte_t* h_out = new byte_t[channel_width];
	cudaMemcpy(h_out, d_byte_out, channel_width*sizeof(byte_t), cudaMemcpyDeviceToHost);

	cudaFree(d_byte_out);
	cudaFree(d_non_max_supp);
	cudaFree(d_G);
	cudaFree(d_blur);
	cudaFree(d_channels);
	cudaFree(d_float_greyscale);
	cudaFree(d_blurred);
	cudaFree(d_hor_sobel);
	cudaFree(d_vert_sobel);
	cudaFree(d_Gx);
	cudaFree(d_Gy);
	return {h_out, width, height, 1};

}

__global__ void float_to_byte(const float* float_greyscale, size_t width, size_t height, byte_t* byte_greyscale)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int idx = x+width*y;
	if(x<width && y<height)
	{
		float f_grey = float_greyscale[idx];
		if(f_grey >= 1)
		{
			byte_greyscale[idx] = 255;
		}
		else if(f_grey < 0)
		{
			byte_greyscale[idx] = 0;
		}
		else
		{
			byte_greyscale[idx] = byte_t(256*float_greyscale[idx]);
		}
	}
}

__global__ void apply_kernel(const float* image, size_t width, size_t height, const float* kernel, size_t kernel_width, size_t kernel_height, float* out)
{
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;
	int idx = x+width*y;
	unsigned height_offset = kernel_height/2;
	unsigned width_offset = kernel_width/2;
	if(x<width && y<height)
	{
		float sum = 0.0f;
		for(int i = 0; i<kernel_height; i++)
		{
			int src_y = i-height_offset+y;
			for(int j = 0; j<kernel_width; j++)
			{
				int src_x = j-width_offset+x;
				if(src_x >= 0 && src_x < width && src_y >= 0 && src_y < height)
				{
					sum+=image[src_x + width*src_y]*kernel[i*kernel_width+j];
				}
			}
		}
		out[idx] = sum;
	}
}

dim3 get_optimal_blocks(size_t width, size_t height)
{
	//have enough blocks to cover the entire thing
	unsigned blocks_height = height/MAX_THREADS_PER_SIDE;
	if(height%MAX_THREADS_PER_SIDE != 0)
	{
		blocks_height++;
	}

	unsigned blocks_width = width/MAX_THREADS_PER_SIDE;
	if(width%MAX_THREADS_PER_SIDE != 0)
	{
		blocks_width++;
	}

	return {blocks_width,blocks_height};
}

void handle_cuda_error(cudaError_t err)
{
	if(err != cudaSuccess)
	{
		std::cout << cudaGetErrorName(err) << std::endl;
	}
}
