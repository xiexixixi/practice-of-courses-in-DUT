#include<stdlib.h>
#include<stdio.h>
#include<windows.h>
#include<math.h>
#include"UI.h"
#include"translate.h"
int ifread = 0;
void hideCursor() {
	CONSOLE_CURSOR_INFO cursor_info = { 1,0 };
	SetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE), &cursor_info);
}
void gotoxy(int x, int y)//移动光标
{
	COORD coord;
	coord.X = x;
	coord.Y = y;
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);
}
void gotoxyAndPutchar(int x, int y,const char* s, int color)//移动光标至x,y并输出字符，有15种颜色，10种进制
{
	if (x >= 0 && x <= 200 && y >= 0 && y <= 200)
	{
		gotoxy(x, y);
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color);
		printf("%s", s);
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
	}
}
void SetColor(char* s, int color)//改变字体颜色 
{
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color);
	printf("%s", s);
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
}
void showboard() {
	int i = 0;
	gotoxyAndPutchar(20, 13, "            >>>>>>>欢迎来到电子地图管理系统<<<<<<<          ", 0x0b);
	gotoxyAndPutchar(20, 18, "                                            made by Group 3 \n", 0x09);
	Sleep(2000);
	system("cls");
}
void clearline(int linenumber) {//清除某行 
	int i = 0;
	for (i = 0; i<100; i++) {
		gotoxyAndPutchar(i, linenumber, " ", 0x07);
	}
}
void announce(const char *s) {//在通知行显示通知  
	clearline(24);
	gotoxyAndPutchar(30, 24, s, 0x07);
}
void subAnnounce(const char *s) {//在副通知行定位光标 
	clearline(26);
	gotoxyAndPutchar(30, 26, s, 0x07);
}
void basicInterface()
{
	gotoxyAndPutchar(30, 7, "请选择您要进行的操作：\n", 0x0f);
	if (!ifread) {
		gotoxyAndPutchar(30, 10, "1.读取文件(./data/GBLT.dat)\n", 0x0c);
	}
	else {
		clearline(10);
	}
	gotoxyAndPutchar(30, 13, "2.排序并输出排序结果(./data/GBLT.dat)\n", 0x0a);
	gotoxyAndPutchar(30, 16, "3.检索\n\n", 0x0e);
	gotoxyAndPutchar(30, 19, "4.更新\n\n", 0x0d);
	gotoxyAndPutchar(30, 22, "0.退出\n\n", 0x0b);
	clearline(26); subAnnounce("");

}
void searchInterface()
{
	gotoxyAndPutchar(30, 7, "请选择检索方式：\n\n", 0x0f);
	gotoxyAndPutchar(30, 10, "1:按LinkID检索\n\n", 0x0c);
	gotoxyAndPutchar(30, 13, "2:按交叉Link列表示Class番号检索\n\n", 0x09);
	gotoxyAndPutchar(30, 16, "3:按岔路数检索\n\n", 0x0a);
	gotoxyAndPutchar(30, 19, "4:按道路名称检索\n\n", 0x0b);
	subAnnounce(" ");
}
