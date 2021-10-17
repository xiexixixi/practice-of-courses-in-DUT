//
// Created by xinzhi on 2020/4/14.
//
#include <iostream>
#include <string>
using namespace std;
int main(int argc, char *argv[]){
    int a = 0;
    int b = 0;
    char x[10] = {};
    char y[10] = {};
    cout << "please input two number such as 1 enter 2 enter\n";
    cin.getline(x, 10);
    cin.getline(y, 10);
    sscanf(x, "%d", &a);
    sscanf(y, "%d", &b);
    cout << "your input is " << x << " and " << y << "\n";
    cout << "min is " << (a < b ? a : b) << "\n";
    return 0;
}