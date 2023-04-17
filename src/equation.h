// Copyright (c) 2023 xfw <xfwahss@qq.com>
#include <string>
#include <vector>
using std::string;
using std ::vector;
class NSEquation {
  private:
    int dimension;
    vector<float> u;
    string var;

  public:
    NSEquation();
    void setDimension(int);
};
