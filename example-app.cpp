#include <torch/script.h> // One-stop header.
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <typeinfo>
#include <string>
#include <stdio.h>
#include <stdlib.h>

/* int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
} */
//int main(int argc, const char* argv[]) {

using namespace cv; 
using namespace std; 

void use_model(const char* in_filename,const char* out_filename,const char* in_bmp_name,const char* out_bmp_name,torch::jit::script::Module module)
{

 //读取txt中的input像素值
	FILE *fp_input;
	
	//格式化读
	if((fp_input=fopen(in_filename,"r"))==NULL)
	{
		printf("打开文件in_filename失败ljd \n");
		exit(EXIT_FAILURE);
	}
	int q;
	
	


   //输入图像
    //cv::Mat image = cv::imread("/home/lzh412/super-resolution-master/total_RaceHorses_832x480_30_off1.bmp",cv::ImreadModes::IMREAD_GRAYSCALE);
    //cv::Mat image = cv::imread(PATH,cv::ImreadModes::IMREAD_GRAYSCALE);
	
	//cout<<image.rows<<endl<<image.cols<<endl<<image.type()<<endl;

	//从lowinput中读取图像宽和高
	fscanf(fp_input,"%d",&q);
	int lowinput_height=q;
	fscanf(fp_input,"%d",&q);
	int lowinput_width=q;

	//从lowinput.txt中读取像素值，对input图像像素赋值
	cv::Mat input_Mat;
    	input_Mat=cv::Mat::zeros(lowinput_height,lowinput_width,CV_8UC1);
        for(int i=0;i<input_Mat.rows;i++)
        for(int j=0;j<input_Mat.cols;j++)
        {
	   fscanf(fp_input,"%d",&q);
	   //printf("%d\n",q);
           input_Mat.at<uchar>(i,j)=uchar(q);

        }
	
	fclose(fp_input);

	cv::imwrite(in_bmp_name, input_Mat);



    cv::Mat image_transfomed;
    //cv::resize(image, image_transfomed, cv::Size(832, 480));
    cv::resize(input_Mat, image_transfomed, cv::Size(lowinput_width, lowinput_height));

   // 图像转换为Tensor
    torch::Tensor tensor_image = torch::from_blob(image_transfomed.data, {image_transfomed.rows, image_transfomed.cols,1},torch::kByte);
    tensor_image = tensor_image.permute({2,0,1});
    tensor_image = tensor_image.toType(torch::kFloat);
    tensor_image = tensor_image.div(255);//[ Variable[CPUType]{1,1,480,832} ]

    tensor_image = tensor_image.unsqueeze(0);

/*  
   //输出tensor_image中的前10个元素看一看
    for(int i=0;i<10;i++)
       {cout<<*(tensor_image[0][0][0][i].data<float>())<<"  ";}
    cout<<endl;
*/
  
    // 网络前向计算
    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward({tensor_image}).toTensor();
    output=output.mul(255.0).clamp(0,255);

/*    //输出tensor_image中的前10个元素看一看
    for(int i=0;i<10;i++)
       {cout<<*(output[0][0][0][i].data<float>())<<"  ";}
    cout<<endl;
*/ 
    
  

	FILE *fp_highoutput;
	if((fp_highoutput=fopen(out_filename,"w"))==NULL)
	{

		printf("打开文件out_filename.txt失败 \n");
		exit(EXIT_FAILURE);
	}



    //把输出tensor转化成Mat对象
    cv::Mat output_Mat;
    output_Mat=cv::Mat::zeros(input_Mat.size(),input_Mat.type());
    for(int i=0;i<input_Mat.rows;i++)
       for(int j=0;j<input_Mat.cols;j++)
       {
           output_Mat.at<uchar>(i,j)=*(output[0][0][i][j].data_ptr<float>());
	   int pixel_highoutput=int(output_Mat.at<uchar>(i,j));
	   fprintf(fp_highoutput,"%d\n",pixel_highoutput);

       }
	fclose(fp_highoutput);
	
	//这里会报错，有待优化
    /* cv::imwrite(out_bmp_name, output_Mat);
     //创建一个窗口
	namedWindow("output_mat", WINDOW_AUTOSIZE);
    // 显示图像
        imshow("output_mat", output_Mat);//第一个参数是显示图像窗口的名字，第二个参数是个Mat型对象
 */
  
}

int main(void) {
 
   
for(int i=0;i<3;i++)//控制对YUV三通道的滤波
{
  cout<<"/////////////// ";
    
  //输出YUV三通道像素txt,字符串拼接
  const char* in_filename=NULL;
  const char* out_filename=NULL;
  const char* in_bmp_name=NULL;
  const char* out_bmp_name=NULL;
  char ch[2]={0};
  //itoa(int(compID),ch,10);//int转char数组
  string src1_in="lowinput_";
  string src1_out="highoutput_";
  string src2=ch;
  string src3=".txt";
  string src3_bmp=".bmp";

  string in_channel_name=src1_in+src2+src3;
  in_filename=in_channel_name.c_str();
  string out_channel_name=src1_out+src2+src3;
  out_filename=out_channel_name.c_str();

  string in_bmp_channel_name=src1_in+src2+src3_bmp;
  in_bmp_name=in_bmp_channel_name.c_str();
  string out_bmp_channel_name=src1_out+src2+src3_bmp;
  out_bmp_name=out_bmp_channel_name.c_str();

	torch::jit::script::Module module;
	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		module = torch::jit::load("/home/li/Desktop/LZH/HM-16.15-4995-jiabeizhu/HEVC_CNN/160epoch_dense_DLVC_QP=37_model2_4block.pt");
	   }
	   catch (const c10::Error& e) {
		std::cerr << "error loading the model\n";
		return -1;
	  }  
	  

	  
		use_model(in_filename,out_filename,in_bmp_name,out_bmp_name,module);
		std::cout << "ok\n";
	}
    return 0;

   


}

