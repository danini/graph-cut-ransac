# Graph-Cut RANSAC

The Graph-Cut RANSAC algorithm proposed in paper: Daniel Barath and Jiri Matas; Graph-Cut RANSAC, Conference on Computer Vision and Pattern Recognition, 2018. 
It is available at http://openaccess.thecvf.com/content_cvpr_2018/papers/Barath_Graph-Cut_RANSAC_CVPR_2018_paper.pdf

# Installation

To build and install `GraphCutRANSAC`, clone or download this repository and then build the project by CMAKE. 

# Example project

To build the sample project showing examples of fundamental matrix, homography and essential matrix fitting, set variable `CREATE_SAMPLE_PROJECT = ON` when creating the project in CMAKE. 

Next to the executable, copy the `data` folder and, also, create a `results` folder. 

# Requirements

- Eigen 3.0 or higher
- CMake 2.8.12 or higher
- OpenCV 3.0 or higher
- A modern compiler with C++17 support


# Acknowledgements

When using the algorithm, please cite `Barath, Daniel, and Matas, Jiří. "Graph-cut RANSAC." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018`.
