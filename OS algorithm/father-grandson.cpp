
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
    } else if(spid == 0) {
        // 当前进程为子进程
        pid_t sspid = -1;
        sspid = fork();
        if(sspid < 0) {
            // 创建进程失败
            cout << "error create grandson\n";
        } else if(sspid == 0) {
            // 当前进程为孙子进程打印自己的进程号
            cout << "i am grandson process2 my pid is " << getpid() << "\n\n";
        } else {
            // 子进程打印自己和孙子进程号
            cout << "i am process1 my pid is " << getpid() << "  ";
            cout << "my son process2 pid is " << sspid << "\n\n";
        }
    } else {
        // 当前进程为最高父进程打印自己和子进程的进程号
        cout << "i am main process my pid is " << getpid() << "  ";
        cout << "my son process1 pid is " << spid << "\n\n";
    }
    return 0;
}
