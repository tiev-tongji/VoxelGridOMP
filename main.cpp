
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include "tic_toc.h"
#include "voxel_grid_omp.h"

std::string pcd_file = "/home/hk/CLionProjects/pcl-test/map.pcd";

int
main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZINormal>());

  // Fill in the cloud data
  pcl::PCDReader reader;
  // Replace the path below with the path where you saved your file
  reader.read (pcd_file, *cloud); // Remember to download the file first!
  std::cerr << "PointCloud before filtering: " << cloud->size() << std::endl;

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_PCL_filtered(new pcl::PointCloud<pcl::PointXYZINormal>());
  // Create the filtering object
  pcl::VoxelGrid<pcl::PointXYZINormal> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (1.0, 1.0, 1.0);

    TicToc t_vg;
    sor.filter (*cloud_PCL_filtered);
    printf("voxel grid PCL cost: %fms\n", t_vg.toc());
    std::cerr << "PointCloud after filtering: " << cloud_PCL_filtered->size() << std::endl;


    pcl::VoxelGridOMP vg_omp;
    vg_omp.setInputCloud (cloud);
    vg_omp.setNumberOfThreads(6);
    vg_omp.setLeafSize (1.0, 1.0, 1.0);
    vg_omp.setSaveLeafLayout(false);

    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_OPM_filtered(new pcl::PointCloud<pcl::PointXYZINormal>());
    TicToc t_vg_omp;
//    vg_omp.downSample(*cloud_OPM_filtered);
    vg_omp.setFinalFilter(true);
    vg_omp.filter(*cloud_OPM_filtered);
    printf("voxel grid OMP cost: %fms\n", t_vg_omp.toc());
    std::cerr << "PointCloud after filtering: " << cloud_OPM_filtered->size() << std::endl;

//    for (int i = 0; i < (int)cloud_PCL_filtered->size(); ++i) {
//        const pcl::PointXYZINormal& p1 = cloud_PCL_filtered->points[i];
//        const pcl::PointXYZINormal& p2 = cloud_OPM_filtered->points[i];
//        if (p1.getVector3fMap() != p2.getVector3fMap() )
//        {
//            printf("i = %d xyz:\n", i);
//            std::cout << p1.getVector3fMap().transpose() <<std::endl;
//            std::cout << p2.getVector3fMap().transpose() <<std::endl;
//        }
//        if (p1.getNormalVector3fMap() != p2.getNormalVector3fMap() )
//        {
//            printf("i = %d normal:\n", i);
//            std::cout << p1.getNormalVector3fMap().transpose() <<std::endl;
//            std::cout << p2.getNormalVector3fMap().transpose() <<std::endl;
//        }
//    }
//    printf("check done.\n");

//    pcl::io::savePCDFileASCII("/tmp/cloud_PCL_filtered.pcd", *cloud_PCL_filtered);
//    pcl::io::savePCDFileASCII("/tmp/cloud_OPM_filtered.pcd", *cloud_OPM_filtered);

    return (0);
}