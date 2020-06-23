//
// Created by xinzhi on 2020/4/14.
//
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
using namespace std;
int main(int argc, char* argv[])
{
    int ans = 0;
    char x[10] = {};
    char y[10] = {};
    char* args[] = {"", NULL};
    char* path[] = {NULL};
    char* max = "max";
    char* min = "min";
    char* average = "average";
    cout << "请选择你想执行的程序(min, max, average)\n";
    string ins = "";
    cin >> ins;
    if(ins == "min") {
        args[0] = min;
    } else if(ins == "max") {
        args[0] = max;
    } else if(ins == "average") {
        args[0] = average;
    } else {
        cout << "指令错误";
    }
    ans = execve(args[0],args,path);
    cout << ans;
    return 0;
}
