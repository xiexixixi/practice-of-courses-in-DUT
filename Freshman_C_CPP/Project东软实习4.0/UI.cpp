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
void gotoxy(int x, int y)//�ƶ����
{
	COORD coord;
	coord.X = x;
	coord.Y = y;
	SetConsoleCursorPosition(GetStdHandle(STD_OUTPUT_HANDLE), coord);
}
void gotoxyAndPutchar(int x, int y,const char* s, int color)//�ƶ������x,y������ַ�����15����ɫ��10�ֽ���
{
	if (x >= 0 && x <= 200 && y >= 0 && y <= 200)
	{
		gotoxy(x, y);
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color);
		printf("%s", s);
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
	}
}
void SetColor(char* s, int color)//�ı�������ɫ 
{
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color);
	printf("%s", s);
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 0x07);
}
void showboard() {
	int i = 0;
	gotoxyAndPutchar(20, 13, "            >>>>>>>��ӭ�������ӵ�ͼ����ϵͳ<<<<<<<          ", 0x0b);
	gotoxyAndPutchar(20, 18, "                                            made by Group 3 \n", 0x09);
	Sleep(2000);
	system("cls");
}
void clearline(int linenumber) {//���ĳ�� 
	int i = 0;
	for (i = 0; i<100; i++) {
		gotoxyAndPutchar(i, linenumber, " ", 0x07);
	}
}
void announce(const char *s) {//��֪ͨ����ʾ֪ͨ  
	clearline(24);
	gotoxyAndPutchar(30, 24, s, 0x07);
}
void subAnnounce(const char *s) {//�ڸ�֪ͨ�ж�λ��� 
	clearline(26);
	gotoxyAndPutchar(30, 26, s, 0x07);
}
void basicInterface()
{
	gotoxyAndPutchar(30, 7, "��ѡ����Ҫ���еĲ�����\n", 0x0f);
	if (!ifread) {
		gotoxyAndPutchar(30, 10, "1.��ȡ�ļ�(./data/GBLT.dat)\n", 0x0c);
	}
	else {
		clearline(10);
	}
	gotoxyAndPutchar(30, 13, "2.�������������(./data/GBLT.dat)\n", 0x0a);
	gotoxyAndPutchar(30, 16, "3.����\n\n", 0x0e);
	gotoxyAndPutchar(30, 19, "4.����\n\n", 0x0d);
	gotoxyAndPutchar(30, 22, "0.�˳�\n\n", 0x0b);
	clearline(26); subAnnounce("");

}
void searchInterface()
{
	gotoxyAndPutchar(30, 7, "��ѡ�������ʽ��\n\n", 0x0f);
	gotoxyAndPutchar(30, 10, "1:��LinkID����\n\n", 0x0c);
	gotoxyAndPutchar(30, 13, "2:������Link�б�ʾClass���ż���\n\n", 0x09);
	gotoxyAndPutchar(30, 16, "3:����·������\n\n", 0x0a);
	gotoxyAndPutchar(30, 19, "4:����·���Ƽ���\n\n", 0x0b);
	subAnnounce(" ");
}
