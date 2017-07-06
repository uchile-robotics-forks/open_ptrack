#include "ros/ros.h"
#include "opt_msgs/Onoff.h"
#include <cstdlib>

int main(int argc, char **argv)
{
  ros::init(argc, argv, "following_client");
  if (argc != 2)
  {
    ROS_INFO("usage: following_client 1 <or> following_client 0");
    return 1;
  }

  ros::NodeHandle n;
  ros::ServiceClient client = n.serviceClient<opt_msgs::Onoff>("follower/Active");
  opt_msgs::Onoff srv;
 
  srv.request.select = (atoll(argv[1]) == 1);
  if (client.call(srv))
  {
    ROS_INFO("Ready");
  }
  else
  {
    ROS_ERROR("Failed to call service following_client");
    return 1;
  }

  return 0;
}
