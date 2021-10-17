#include <iostream>
using namespace std;
int main(int argc, char *argv[])
{
    int a = 0;
    int b = 0;
    int c = 0;
    double ans = 0;
    char x[10] = {};
    char y[10] = {};
    char z[10] = {};
    cout << "please input three number such as 1 enter 2 enter 3 enter\n";
    cin.getline(x, 10);
    cin.getline(y, 10);
    cin.getline(z, 10);
    sscanf(x, "%d", &a);
    sscanf(y, "%d", &b);
    sscanf(z, "%d", &c);
    cout << "your input is " << x << " and " << y << " and " << z << "\n";
    ans = (double)(a + b + c)/3;
    cout << "average is " << ans << "\n";
    return 0;
}
