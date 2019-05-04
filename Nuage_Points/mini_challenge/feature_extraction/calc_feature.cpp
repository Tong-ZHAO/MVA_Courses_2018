#if defined (_MSC_VER) && !defined (_WIN64)
#pragma warning(disable:$@$$) // boost::number_distance_distance()
                              // converts 64 to 32 bits integers
#endif

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <CGAL/Simple_cartesian.h>
#include <CGAL/Classification.h>
#include <CGAL/Point_set_3.h>
#include <CGAL/Point_set_3/IO.h>
#include <CGAL/Real_timer.h>

typedef CGAL::Simple_cartesian<double>  Kernel;
typedef Kernel::Point_3                 Point;
typedef CGAL::Point_set_3<Point>        Point_set;
typedef Kernel::Iso_cuboid_3            Iso_cuboid_3;
typedef std::vector<Point>              Point_range;

typedef Point_set::Point_map            Pmap;
typedef Point_set::Property_map<int>    Imap;
typedef Point_set::Property_map<double> Dmap;

namespace Classification = CGAL::Classification;

typedef Classification::Label_handle                                            Label_handle;
typedef Classification::Feature_handle                                          Feature_handle;
typedef Classification::Label_set                                               Label_set;
typedef Classification::Feature_set                                             Feature_set;
typedef Classification::Point_set_feature_generator<Kernel, Point_set, Pmap>    Feature_generator;


void usage(){
    std::cout << "Usage: ./calc_feature file.ply" << std::endl;
    std::cout << "file.ply: the file containing the point cloud" << std::endl;
}


int main(int argc, char** argv){
    // Check number of params
    if (argc != 2){
        usage();
        return EXIT_FAILURE;
    }

    // Read training dataset
    std::string filename(argv[1]);
    std::ifstream in(filename.c_str(), std::ios::binary);
    Point_set pts;
    std::cerr << "Reading input..." << std::endl;
    in >> pts;

    Feature_set features;

    std::cerr << "Generating features..." << std::endl;
    CGAL::Real_timer t;
    t.start();
    Feature_generator generator(pts, pts.point_map(), 4); // using 4 scales
    generator.generate_point_based_features(features);
    t.stop();
    std::cerr << features.size() << " feature(s) generated in " << t.time() << " second(s)" << std::endl;

    for(std::size_t i = 0; i < features.size(); i++){
        Feature_handle ft = features[i];
        Dmap ft_map;
        bool success = false;
        boost::tie(ft_map, success) = pts.add_property_map<double>(ft->name(), 0.);

        for(std::size_t j = 0; j < pts.size(); j++)
            ft_map[j] = ft-> value(j);
    }

    std::cerr << "Writing file..." << std::endl;
    std::ofstream f (filename.substr(0, filename.size() - 4) + "_fd.ply", std::ofstream::binary);
    f.precision(10);
    f << pts;
    std::cerr << "All done!" << std::endl;

    return EXIT_SUCCESS;
}