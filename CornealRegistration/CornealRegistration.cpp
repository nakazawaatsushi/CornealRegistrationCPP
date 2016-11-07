#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

int main(int argc, char **argv)
{
	if( argc < 2 ){
		printf("usage: %s [file1] [file2]\n", argv[0]);
		exit(0);
	}

    Mat img1 = imread(argv[1], IMREAD_GRAYSCALE);
    Mat img2 = imread(argv[2], IMREAD_GRAYSCALE);

    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> nn_matches;

	matcher.match(desc1, desc2, nn_matches);

	for( int i = 0; i < nn_matches.size(); i++ ){
		printf("%03d: matched %d - %d : %f\n", i, nn_matches[i].queryIdx, nn_matches[i].trainIdx, nn_matches[i].distance);
	}

    Mat res;
    drawMatches(img1, kpts1, img2, kpts2, nn_matches, res);
    imwrite("res.png", res);

    cout << "A-KAZE Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << nn_matches.size() << endl;
    cout << endl;

    return 0;
}

