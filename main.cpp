#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdio.h>
#include <time.h>

// SIZE = number of xmls for testing smile detection
// name of the xmls are provided below in string array 'smile_array'

#define SIZE 5

using namespace std;
using namespace cv;

float scale_factor = 1.1;
int min_neighbor = 2;
CascadeClassifier smile_cascade;
string smile_array[SIZE] = {"smiled_01.xml", "smiled_02.xml", "smiled_03.xml", "smiled_04.xml", "smiled_05.xml"};

int main(int argc, char *argv[]) {
	clock_t tStart = clock();	
    for (int i = 1; i < argc; i++) {
		if (string(argv[i]) == "-sf") {
	    	scale_factor = (float)atof(argv[++i]);
		} else if (string(argv[i]) == "-mn") {
	    	min_neighbor = atoi(argv[++i]);
		}
    }

    for (int i = 0; i < SIZE; i++) {
		string smile_cascade_name = smile_array[i];

		if (!smile_cascade.load(smile_cascade_name)) {
	    	printf("--(!) Error loading smile cascade: %s\n", smile_cascade_name.c_str());
			return -1;
		} else {
	    	string line = "";
	    	printf("Testing smile detection for negative dataset using xml: %s\n", smile_cascade_name.c_str());
	    	printf("Scale Factor: %f, Minimum Neighbor: %d\n", scale_factor, min_neighbor);
	    	ifstream myfile("smiles_01_neg.idx");
	    	if (myfile.is_open()) {
				int found = 0; int not_found = 0; int total = 0;
				while (getline(myfile, line)) {
		    		Mat img = imread(line, 0);
		    		vector<Rect> smile;
					smile_cascade.detectMultiScale(img, smile, scale_factor, min_neighbor, 0|2, Size (20,20));
					if (smile.size() > 0) {
						found++;
					} else {
						not_found++;
					}
					total++;
				}
				printf("Smiles found: %d\n", found);
				printf("Smiles missing: %d\n", not_found);
				printf("Total sample: %d\n", total);
				printf("Accuracy: %f\n", ((double)not_found/total));
				printf("\n");
				myfile.close();
			}

			printf("Testing smile detection for positive dataset using xml: %s\n", smile_cascade_name.c_str());
			printf("Scale Factor: %f, Minimum Neighbor: %d\n", scale_factor, min_neighbor);
        	myfile.open("smiles_01_pos.idx");
        	if (myfile.is_open()) {
        		int found = 0; int not_found = 0; int total = 0;
        		while (getline(myfile, line)) {
					line = line.substr(0, line.find(" "));
            	    Mat img = imread(line, 0);
            	    vector<Rect> smile;
            	    smile_cascade.detectMultiScale(img, smile, scale_factor, min_neighbor, 0|2, Size (20,20));
					if (smile.size() > 0) {
            	    	found++;
					} else {
            	    	not_found++;
					}
					total++;
				}
            	printf("Smiles found: %d\n", found);
            	printf("Smiles missing: %d\n", not_found);
            	printf("Total sample: %d\n", total);
            	printf("Accuracy: %f\n", ((double)found/total));
            	printf("\n");
            	myfile.close();
			}
		}
	}
	printf("Time taken: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
	return 0;
}
