
#define mCV 1                                         //if use opencv

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>

#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/string.h>
#include <dlib/clustering.h>
#if mCV 
#include "opencv2/imgproc/imgproc.hpp"
#else 
#include <dlib/gui_widgets.h>
#endif

using namespace dlib;
using namespace std;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

cv::VideoCapture cap(0);                              //turn on camera

frontal_face_detector detector;
shape_predictor sp;
anet_type net;
std::vector<matrix<float, 0, 1>> face_descriptors;

std::vector<string> name;
std::vector<int>label;

const int numP = 5;                                   //The number of pictures each person needs to collect

void addPeople(void)                                  //add a person
{
	string nowName;
	cout << "Input the name of people: ";
	cin >> nowName;

	bool isExist = false;

	for (unsigned int i = 0; i < name.size(); i++)
	{
		if (strcmp(nowName.c_str(), name[i].c_str()) == 0)
		{
			isExist = true;
			break;
		}
	}
	if (isExist)                                      //the person is already exist
	{
		cout << nowName << " is already exist!" << endl;
		return;
	}

	string dirName = "./data/" + nowName;             //create folder to save pictures
	struct stat status;
	if(stat(dirName.c_str(), &status) != 0)
		//if (access(dirName.c_str(), 0) == -1)
	{
		int cf = mkdir(dirName.c_str(), 0777);
		if (cf == -1)
		{
			cout << "Can not create folder " + dirName + "!"<<endl;
			return;
		}
	}

	int nowNum = 1;                                   //now photo number
	char faceNum[10];

	std::vector<matrix<rgb_pixel>> chip_faces;        //faces

	while (nowNum <= numP)                            //Until numP photos are collected
	{
		cv::Mat mat;
		while (cap.read(mat))
		{
			cv::Mat temp = mat;
			cv_image<bgr_pixel> cimg(mat);

			std::vector<rectangle> faces = detector(cimg);    //Detect faces 

			unsigned int num = faces.size();          //The number of faces in the current image

			if (num < 1) continue;

			full_object_detection shape = sp(cimg, faces[0]); // Find the pose of the first face.

			int key = cv::waitKey(10);

			if ((key == 'p' || key == 'P') && num == 1)
			{
				matrix<rgb_pixel> face_chip;
				extract_image_chip(cimg, get_face_chip_details(shape,150, 0.25), face_chip);
				chip_faces.push_back(move(face_chip));

				//itoa_s(nowNum, faceNum, 10);
				sprintf(faceNum, "%d", nowNum); 
				cv::imwrite(dirName + "/" + faceNum + ".jpg", temp);

				cout << "Register " + nowName + faceNum + " sucessfully!" << endl;

				nowNum = nowNum + 1;                  //register 1 photo sucessfully

				break;
			}

			cv::Rect face_rect;
			face_rect.x = shape.get_rect().left();
			face_rect.y = shape.get_rect().top();
			face_rect.width = shape.get_rect().width();
			face_rect.height = shape.get_rect().height();
			cv::rectangle(mat, face_rect, CV_RGB(0, 0, 255), 2, 8, 0);

			for (unsigned long j = 0; j < shape.num_parts(); j++)
			{
				int x = shape.part(j)(0);
				int y = shape.part(j)(1);
				cv::circle(mat, cvPoint(x, y), 2, CV_RGB(0, 255, 0), CV_FILLED);
			}
			cv::imshow("Test", mat);
		}
	}

	std::vector<matrix<float,0,1>> now_face_descriptors = net(chip_faces);

	for(unsigned int i = 0; i < now_face_descriptors.size(); i++)
	{
		face_descriptors.push_back(now_face_descriptors[i]);
		name.push_back(nowName);
	}

	cout << nowName << " register sucessfully!" << endl;

	ofstream mFile("./data/people.txt", ios::app);    //add the person to file
	mFile << endl;
	mFile << nowName << endl;
	mFile.close();
}

int main()
{
	try
	{
		//cv::VideoCapture cap(0);
		if (!cap.isOpened())
		{
			cerr << "Unable to connect to camera" << endl;
			return 1;
		}
#if mCV
		cv::namedWindow("Test", cv::WINDOW_AUTOSIZE);
#else 
		image_window win;
#endif
		// Load face detection and pose estimation models.
		detector = get_frontal_face_detector();
		deserialize("./data/model/shape_predictor_68_face_landmarks.dat") >> sp;
		deserialize("./data/model/dlib_face_recognition_resnet_model_v1.dat") >> net;

		//if (access("./data/people.txt", 0) != -1)
		struct stat status;
		if(stat("./data/people.txt", &status) == 0)
		{
			ifstream mFile("./data/people.txt");      //people already exist
			string mName = "";
			mFile >> mName; 
			char faceNum[10];
			while (!mFile.fail())
			{
				bool ifR = false;
				std::vector<matrix<rgb_pixel>> chip_faces; //faces

				for (int i = 1; i <= numP; i++)
				{
					//itoa_s(i, faceNum, 10);
					sprintf(faceNum, "%d", i); 
					cv::Mat mat = cv::imread("./data/" + mName + "/" + faceNum + ".jpg");
					if (mat.empty()) continue;

					cv_image<bgr_pixel> cimg(mat);

					std::vector<rectangle> faces = detector(cimg); //Detect faces 
					unsigned int num = faces.size(); //The number of faces in the current image
					if (num < 1) continue;

					full_object_detection shape = sp(cimg, faces[0]); // Find the pose of the first face.
					matrix<rgb_pixel> face_chip;
					extract_image_chip(cimg, get_face_chip_details(shape,150, 0.25), face_chip);
					chip_faces.push_back(move(face_chip));
				}

				std::vector<matrix<float,0,1>> now_face_descriptors = net(chip_faces);
				for(unsigned int i = 0; i < now_face_descriptors.size(); i++)
				{
					face_descriptors.push_back(now_face_descriptors[i]);
					name.push_back(mName);
					ifR = true;
				}

				if(ifR) cout << mName << " register sucessfully!" << endl;

				mFile >> mName;                       //read the next person
			}

			mFile.close();
		}

		if (name.size() < 1) addPeople();             //add a person if empty

		// Grab and process frames until the main window is closed by the user.
#if mCV
		while (true)
#else 
		while (!win.is_closed())
#endif
		{
			// Grab a frame
			cv::Mat temp;
			if (!cap.read(temp))
			{
				break;
			}
			// Turn OpenCV's Mat into something dlib can deal with.  Note that this just
			// wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
			// long as temp is valid.  Also don't do anything to temp that would cause it
			// to reallocate the memory which stores the image as that will make cimg
			// contain dangling pointers.  This basically means you shouldn't modify temp
			// while using cimg.
			cv_image<bgr_pixel> cimg(temp);

			std::vector<rectangle> faces = detector(cimg); //Detect faces 
			std::vector<full_object_detection> shapes; //Find the pose of each face.
			std::vector<matrix<rgb_pixel>> chip_faces;
			std::vector<matrix<float,0,1>> now_face_descriptors;

			for (unsigned long i = 0; i < faces.size(); i++)
			{
				full_object_detection shape = sp(cimg, faces[i]);
				shapes.push_back(shape);
				matrix<rgb_pixel> face_chip;
				extract_image_chip(cimg, get_face_chip_details(shape,150, 0.25), face_chip);
				chip_faces.push_back(move(face_chip));
			}

			if(chip_faces.size() > 0) now_face_descriptors = net(chip_faces);
#if mCV
			for (unsigned long i = 0; i < shapes.size(); i++) // Display it all on the screen(opencv)
			{
				cv::Rect face_rect;
				face_rect.x = shapes[i].get_rect().left();
				face_rect.y = shapes[i].get_rect().top();
				face_rect.width = shapes[i].get_rect().width();
				face_rect.height = shapes[i].get_rect().height();
				cv::rectangle(temp, face_rect, CV_RGB(0, 0, 255), 2, 8, 0);
/*
				for (unsigned long j = 0; j < shapes[i].num_parts(); j++)
				{
				    int x = shapes[i].part(j)(0);
				    int y = shapes[i].part(j)(1);
				    cv::circle(temp, cvPoint(x, y), 2, CV_RGB(0, 255, 0), CV_FILLED);
				}
*/
				int index = 0;
				double mindata = length(now_face_descriptors[i]-face_descriptors[0]);
				for (unsigned long j = 1; j < face_descriptors.size(); j++)
				{
					if(length(now_face_descriptors[i]-face_descriptors[j]) < mindata)
					{
						index = j;
						mindata = length(now_face_descriptors[i]-face_descriptors[j]);
					}
				}

				cv::Point text_lp = cv::Point(face_rect.x, face_rect.y);
				if (mindata < 0.6)
					cv::putText(temp, name[index], text_lp, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
				else
					cv::putText(temp, "Unknown", text_lp, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0, 0, 255));
			}

			imshow("Test", temp);

			int mKey = cv::waitKey(1);
			if (mKey == 'a' || mKey == 'A') addPeople();
			if (mKey == 'q' || mKey == 'Q') break;
#else 
			win.clear_overlay();                      // Display it all on the screen(dlib)
			win.set_image(cimg);
			win.add_overlay(render_face_detections(shapes));
#endif 
		}
	}
	catch(serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch(exception& e)
	{
		cout << e.what() << endl;
	}

#if mCV 

	cv::destroyAllWindows();

#endif

}