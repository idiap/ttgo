# TTGO: Tensor Train for Global Optimization Problems in Robotics

A PyTorch implementation of TTGO algorithm and the applications presented in the paper "Tensor Train for Global Optimization Problems in Robotics "

Website: https://sites.google.com/view/ttgo/home

Paper: https://arxiv.org/pdf/2206.05077.pdf

### Pre-requistes
- Install the tntorch library from: https://github.com/SuhanNShetty/tntorch
- Pybullet (only for visualization of robotics applications): https://pypi.org/project/pybullet/

### Overview
- *./ttgo.py*: the TTGO algorithm is defined in this class
- *./function_optimization/*: includes the application of ttgo for optimization of several benchmark functions
  - Recommendation: try these notebooks first to understand the approach
- *./toy_robots/*: application of ttgo for simple toy models of robotics problems (planar manipulator IK and reaching tasks)
- *./manipulator/*: application of ttgo for IK and reaching tasks with some standard manipulators

Note: All the implementations are fully compatible for use with GPU. For faster computation, it is highly recommended to use GPU

For any questions, contact the author Suhan Shetty <suhan.shetty@idiap.ch>
