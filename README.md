# VoxelGridOMP

Voxel grid in parallel, using the OpenMP standard, based on PCL.
Tested on PCL-1.7, 1.8, 1.10.

In our experiment, 2,500,000 points cost 68 ms with 10 threads compared to 217 ms for PCL voxelgrid.

It is recommended to use VoxelGripOMP when the size of points exceeds 100000 to compensate for the cost of multithreading.

use:

Modify the line 24 and 28 in voxel_grid_omp.h, setting "PointXYZINormal" to the point TYPE you need.

...

    class VoxelGridOMP : public VoxelGrid<PointXYZINormal>
    {
        public:
            typedef typename pcl::PointXYZINormal PointT;
            typedef typename Filter<PointT>::PointCloud PointCloud;
...
