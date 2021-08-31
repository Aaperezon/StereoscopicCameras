// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
#include <iostream>
#include <lodepng.h>
#include <chrono>
#include <math.h>
#include <algorithm>
#include <numeric>  
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;

using namespace std;

texture<unsigned char, cudaTextureType2D> tex8u_left;
texture<unsigned char, cudaTextureType2D> tex8u_right;

__global__ void box_filter_kernel_8u_c1(unsigned char* output,const int width, const int height, const size_t pitch, const int patch_width, const int patch_height)
{
  int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  int filter_offset_x = patch_width/2;
  int filter_offset_y = patch_height/2;
  //Make sure the current thread is inside the image bounds
  if(xIndex<width && yIndex<height){
    int index = yIndex * pitch + xIndex;
    if(xIndex%4 == 0){
      double min_normalized_diff = INT_MAX;
      int displacement = 0;
      for(int block_shift = 0; block_shift < 16; block_shift++){
        if(xIndex-block_shift*4-filter_offset_x*4 < 0 || yIndex-filter_offset_y < 0){
          output[index+3] = 0;
        
          break;
        }
        output[index+3] = 255;
        double right_sum = 0;
        double left_sum = 0;
        double right_sum_sq = 0;
        double left_sum_sq = 0;
        double left_right_sum = 0;
        double normalized_diff = 0;
        for(int i = -filter_offset_x; i<=filter_offset_x; i++){
          for(int j=-filter_offset_y; j<=filter_offset_y; j++)
          {
            int left =  tex2D(tex8u_left, xIndex+i*4, yIndex+j)+
                        tex2D(tex8u_left, xIndex+i*4+1, yIndex+j)+
                        tex2D(tex8u_left, xIndex+i*4+2, yIndex+j);
            left /= 3;
            int right = tex2D(tex8u_right, xIndex+i*4-block_shift*4, yIndex+j)+
                        tex2D(tex8u_right, xIndex+i*4-block_shift*4+1, yIndex+j)+
                        tex2D(tex8u_right, xIndex+i*4-block_shift*4+2, yIndex+j);
            right /= 3;
            left_right_sum += left*right;
            right_sum_sq += right*right;
            left_sum_sq += left*left;
          }
        }
        normalized_diff = sqrt(left_sum_sq * right_sum_sq )/left_right_sum;
        if(normalized_diff < min_normalized_diff){
          min_normalized_diff = normalized_diff;
          displacement = block_shift;
        }
      }
      if(min_normalized_diff<1.01){
        int res = displacement;
        output[index] = static_cast<unsigned char>(res);
        output[index+1] = static_cast<unsigned char>(res);
        output[index+2] = static_cast<unsigned char>(res);
      } else {
        output[index] = static_cast<unsigned char>(10);
        output[index+1] = static_cast<unsigned char>(0);
        output[index+2] = static_cast<unsigned char>(0);
      }
    }
    else{
      
    }
  }else{
  }
}

void box_filter_8u_c1(unsigned char* left_image, unsigned char* right_image, unsigned char* CPUoutput, const int width, const int height, const int widthStep, const int filterWidth, const int filterHeight)
{

    /*
     * 2D memory is allocated as strided linear memory on GPU.
     * The terminologies "Pitch", "WidthStep", and "Stride" are exactly the same thing.
     * It is the size of a row in bytes.
     * It is not necessary that width = widthStep.
     * Total bytes occupied by the image = widthStep x height.
     */

    //Declare GPU pointer
    unsigned char *GPU_left, *GPU_right, *GPU_output;

    //Allocate 2D memory on GPU. Also known as Pitch Linear Memory
    size_t gpu_image_pitch=0;
    cout<<"box_filter_8u_c1: width "<<width<<endl;
    cout<<"box_filter_8u_c1: height "<<height<<endl;
    cudaMallocPitch<unsigned char>(&GPU_left,&gpu_image_pitch,width,height);
    cudaMallocPitch<unsigned char>(&GPU_right,&gpu_image_pitch,width,height);
    cudaMallocPitch<unsigned char>(&GPU_output,&gpu_image_pitch,width,height);
    //Copy data from host to device.
    cudaMemcpy2D(GPU_left,gpu_image_pitch,left_image,widthStep,width,height,cudaMemcpyHostToDevice);
    cudaMemcpy2D(GPU_right,gpu_image_pitch,right_image,widthStep,width,height,cudaMemcpyHostToDevice);

    //Bind the image to the texture. Now the kernel will read the input image through the texture cache.
    //Use tex2D function to read the image
    cudaBindTexture2D(NULL,tex8u_left,GPU_left,width,height,gpu_image_pitch);
    cudaBindTexture2D(NULL,tex8u_right,GPU_right,width,height,gpu_image_pitch);

    /*
     * Set the behavior of tex2D for out-of-range image reads.
     * cudaAddressModeBorder = Read Zero
     * cudaAddressModeClamp  = Read the nearest border pixel
     * We can skip this step. The default mode is Clamp.
     */
    tex8u_left.addressMode[0] = tex8u_left.addressMode[1] = cudaAddressModeBorder;
    tex8u_right.addressMode[0] = tex8u_right.addressMode[1] = cudaAddressModeBorder;

    /*
     * Specify a block size. 256 threads per block are sufficient.
     * It can be increased, but keep in mind the limitations of the GPU.
     * Older GPUs allow maximum 512 threads per block.
     * Current GPUs allow maximum 1024 threads per block
     */

    dim3 block_size(16,16);

    /*
     * Specify the grid size for the GPU.
     * Make it generalized, so that the size of grid changes according to the input image size
     */

    dim3 grid_size;
    grid_size.x = (width + block_size.x - 1)/block_size.x;  /*< Greater than or equal to image width */
    grid_size.y = (height + block_size.y - 1)/block_size.y; /*< Greater than or equal to image height */

    //Launch the kernel
    box_filter_kernel_8u_c1<<<grid_size,block_size>>>(GPU_output,width,height,gpu_image_pitch,filterWidth,filterHeight);

    //Copy the results back to CPU
    cudaMemcpy2D(CPUoutput,widthStep,GPU_output,gpu_image_pitch,width,height,cudaMemcpyDeviceToHost);

    //Release the texture
    cudaUnbindTexture(tex8u_left);
    cudaUnbindTexture(tex8u_right);

    //Free GPU memory
    cudaFree(GPU_left);
    cudaFree(GPU_right);
    cudaFree(GPU_output);
}
 
void printHistogram(vector<unsigned char> left_image_vector){
  int histogram_steps = 8;
  int* histogram_r{ new int[histogram_steps]{} }; 
  int* histogram_g{ new int[histogram_steps]{} }; 
  int* histogram_b{ new int[histogram_steps]{} }; 
  //Generate Hisotgram
  cout<<"Size of LEFT IMAGE: "<<left_image_vector.size()<<endl;
  for(int i = 0; i < left_image_vector.size(); i+=4){
    histogram_r[left_image_vector[i]/histogram_steps]++;
    histogram_g[(int)(left_image_vector[i+1])/histogram_steps]++;
    histogram_b[(int)(left_image_vector[i+2])/histogram_steps]++;
  }
  cout<<"RED"<<endl;
  for(int i = 0; i<histogram_steps;i++){
    printf("%3i %6i ",i,histogram_r[i]);
    for(int j = 0; j < histogram_r[i]/800; j++){
      cout<<"*";
    }
    cout<<endl;
  }
  cout<<"GREEN"<<endl;

  for(int i = 0; i<histogram_steps;i++){
    printf("%3i %6i ",i,histogram_g[i]);
    for(int j = 0; j < histogram_g[i]/800; j++){
      cout<<"*";
    }
    cout<<endl;
  }
  cout<<"BLUE"<<endl;

  for(int i = 0; i<histogram_steps;i++){
    printf("%3i %6i ",i,histogram_b[i]);
    for(int j = 0; j < histogram_b[i]/800; j++){
      cout<<"*";
    }
    cout<<endl;
  }
  cout<<"histogram ends"<<endl;
  //Done printing histogram
}
template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  stable_sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;



}






void preprocessing(vector<unsigned char> left_image_vector,vector<unsigned char> right_image_vector, int methodc, char* methodv[]){
  cout<<"Received parameter count: "<<methodc<<"\n";
  for(int i = 4; i < methodc; i++){
    if(strcmp(methodv[i],"histogram") == 0){
      printHistogram(left_image_vector);
    }
    else if(strcmp(methodv[i],"option2") == 0){
      printHistogram(left_image_vector);
    }
    else{
      cout<<"No preprocessing: "<<methodv[i]<<".\n";
    }

  }
}


int main(int argc, char** argv) {
  if(argc < 4) {
    cout << "Run with left, right and output image filenames." << endl;
    return 0;
  }
  
  const char* left_filename = argv[1];
  const char* right_filename = argv[2];
  const char* out_filename = argv[3];
  vector<unsigned char> left_image_vector;
  vector<unsigned char> right_image_vector;
  vector<unsigned char> out_image_vector;
  unsigned int width, height;

  // Load images 
  cout << "Load images" << endl;
  cout << lodepng_error_text(lodepng::decode(left_image_vector, width, height, left_filename)) << endl;
  cout << lodepng_error_text(lodepng::decode(right_image_vector, width, height, right_filename)) << endl;
  cout << "done loading" << endl;

  // Prepare the data
  unsigned char* left_image = new unsigned char[left_image_vector.size()];
  unsigned char* right_image = new unsigned char[right_image_vector.size()];
  unsigned char* out_image = new unsigned char[left_image_vector.size()];
  for(int i = 0; i < left_image_vector.size(); i++) {
    left_image[i] = left_image_vector.at(i);
    right_image[i] = right_image_vector.at(i);
    out_image[left_image_vector.size()-i] = left_image_vector.at(i);
  } 
  cout << "done preparing"<<endl;


  
  if(argc>4){
    preprocessing(left_image_vector,right_image_vector,argc,argv);
  }else{
      cout<<"No preprocessing.\n";
  }



  /*

  int* reds{ new int[left_image_vector.size()/4]{} }; 
  int* greens{ new int[left_image_vector.size()/4]{} }; 
  int* blues{ new int[left_image_vector.size()/4]{} }; 

  for(int i = 0; i < left_image_vector.size()/4; i++){
    reds[i] = left_image_vector[i*4];
    greens[i] = left_image_vector[i*4+1];
    blues[i]= left_image_vector[i*4+2];
  }
  cout<<"got separate channels"<<endl;
  vector<size_t> reds_vector;
  vector<size_t> greens_vector;
  vector<size_t> blues_vector;
  reds_vector.assign(reds, reds+left_image_vector.size()/4);
  greens_vector.assign(greens, greens+left_image_vector.size()/4);
  blues_vector.assign(blues, blues+left_image_vector.size()/4);
  cout<<"Going to sort indexes"<<endl;
  
  auto vsorted_r = sort_indexes(reds_vector);
  auto vsorted_g = sort_indexes(greens_vector);
  auto vsorted_b = sort_indexes(blues_vector);
  for (int i = 0; i< vsorted_r.size(); i++) {
    left_image_vector[4*i] =  vsorted_r[i]*255/(vsorted_r.size());
    left_image_vector[4*i+1] =  vsorted_g[i]*255/(vsorted_g.size());
    left_image_vector[4*i+2] =  vsorted_b[i]*255/(vsorted_b.size());

  }
  printHistogram(left_image_vector);
  */









  auto time = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  int block_size = 2;
  box_filter_8u_c1(left_image, right_image, out_image, 384*4, 288, left_image_vector.size()/288, block_size, block_size);
  cout<<"Time after disparity map: "<<(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - time) <<endl;
  
  int max_displacement = -1;
  int max_displacement_i = -1;
  for(int i = 0; i < left_image_vector.size(); i += 4){
    if(out_image[i] > max_displacement){
      max_displacement = out_image[i];
      max_displacement_i = i;
    }
  }
  cout << "Max displacement: " << max_displacement << " at " << max_displacement_i%(384*4)/4 << ", " << max_displacement_i/(384*4) << endl;
  out_image[max_displacement_i-8] = 0;
  out_image[max_displacement_i-7] = 255;
  out_image[max_displacement_i-6] = 0;
  out_image[max_displacement_i-4] = 0;
  out_image[max_displacement_i-3] = 255;
  out_image[max_displacement_i-2] = 0;
  out_image[max_displacement_i] = 0;
  out_image[max_displacement_i+1] = 255;
  out_image[max_displacement_i+2] = 0;
  out_image[max_displacement_i+4] = 0;
  out_image[max_displacement_i+5] = 255;
  out_image[max_displacement_i+6] = 0;
  out_image[max_displacement_i+8] = 0;
  out_image[max_displacement_i+9] = 255;
  out_image[max_displacement_i+10] = 0;
  for(int i = 0; i < left_image_vector.size(); i++) {
    if(i%4 == 3)
      out_image_vector.push_back(out_image[i]);
    else
      out_image_vector.push_back(out_image[i]*255/max_displacement);
  }

  cout<<"Time after augmenting and preparing vector to save: "<<(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - time) <<endl;
  
  cout << "Saving image" << endl;
  cout << lodepng_error_text( lodepng::encode(out_filename, out_image_vector, width, height));

  //if there's an error, display it
  // if(error) cout << "encoder error " << error << ": "<< lodepng_error_text(error) << endl;

  delete[] left_image;
  delete[] right_image;
  delete[] out_image;

  return 0;
}
/*
nvcc copy.cu lodepng.cpp  -I . -o copy && copy.exe tsukuba_L.png tsukuba_R.png out.png
*/
