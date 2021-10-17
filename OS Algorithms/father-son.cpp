//
// Created by xinzhi on 2020/4/14.
//
#include <unistd.h>
#include <iostream>
#include <stdlib.h>
using namespace std;
int main()
{
    pid_t spid = -1;
    spid = fork();
    if(spid < 0) {
        // 创建进程失败
        cout << "error create son process failed\n";
    } 
	else if(spid == 0) 
	{
        // 子进程
        cout << "i am son process1 my pid is " << getpid() << "\n\n";
    } 
	else 
	{
        pid_t sspid = -1;
        sspid = fork();
        if(sspid < 0) 
		{
            // 创建进程失败
            cout << "error create son process failed\n";
        } 
		else if(sspid == 0)
		{
            // 子进程
            cout << "i am son process2 my pid is " << getpid() << "\n\n";
        }
		else
		{
            // 最高父进程
            cout << "i am main process my pid is " << getpid() << "\n";
            cout << "my son process1 pid is " << spid << "\n";
            cout << "my son process2 pid is " << sspid << "\n\n";
        }    
    }
    return 0;
}
