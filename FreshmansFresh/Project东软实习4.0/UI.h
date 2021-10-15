#pragma once
#include<stdlib.h>
#include<stdio.h>
#include<windows.h>
#include<math.h>
void hideCursor();
void gotoxy(int x, int y);
void gotoxyAndPutchar(int x, int y, const char* s, int color);
void SetColor(char* s, int color);
void showboard();
void clearline(int linenumber);
void announce(const char *s);
void subAnnounce(const char *s);
void basicInterface();
void searchInterface();